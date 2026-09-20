"""STEP 7 — GPS parsing and conversion to local metric coordinates.

This module reads the drone flight-log GPS fixes recorded by the capture
pipeline (``data/gps/gps.csv``), validates them, and converts the WGS84
geodetic coordinates to a local metric frame that the SfM backend can use to
provide real-world scale (STEP 8).  Two metric representations are supported:

* an east-north-up (ENU) tangent-plane frame anchored at the first fix, and
* a UTM projection (zone either explicit or derived from the data).

All geodetic input is WGS84.  Conversions never treat latitude/longitude
degrees as metres.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_GPS_COLUMNS = ("timestamp_s", "latitude", "longitude", "altitude_m")

#: Case-insensitive aliases accepted for each canonical GPS field, so the
#: capture pipeline can emit timestamp, lat, lng, alt, ... without rework.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp_s": ("timestamp", "timestamp_s", "time", "t", "epoch_time", "seconds"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lng", "lon", "long"),
    "altitude_m": ("altitude", "altitude_m", "alt", "height", "alt_m"),
}


@dataclass(frozen=True)
class GPSFix:
    """A single validated GPS fix from the flight log.

    Timestamps are seconds since the flight started (positive and finite).
    Coordinates use the WGS84 datum: latitude/longitude in decimal degrees,
    altitude in metres above the WGS84 ellipsoid.
    """

    timestamp_s: float
    latitude: float
    longitude: float
    altitude_m: float


@dataclass(frozen=True)
class MetricFix:
    """A GPS fix projected into a local metric frame.

    ``easting_m/northing_m/up_m`` are metres in the resolved CRS (ENU tangent
    plane or UTM).  The geodetic fields are preserved so writing the metric
    CSV back to disk keeps full provenance.
    """

    timestamp_s: float
    latitude: float
    longitude: float
    altitude_m: float
    easting_m: float
    northing_m: float
    up_m: float


@dataclass(frozen=True)
class GpsResult:
    """Everything produced by one :func:`run_gps_conversion` call."""

    source: str
    n_fixes: int
    crs: str
    zone: Optional[str]
    origin: GPSFix
    extent_m: float
    metric_csv: str
    report_json: str


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:
    """Map the actual CSV header to canonical field names via aliases.

    Returns a dict ``canonical -> actual-header``.  Raises ``ValueError`` when
    any canonical field cannot be resolved or when two headers resolve to the
    same canonical field.
    """
    lowered = {h.strip().lower(): h for h in fieldnames if h.strip()}
    resolved: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        matches = [lowered[a] for a in aliases if a in lowered]
        if not matches:
            raise ValueError(
                f"missing GPS column '{canonical}' — expected one of "
                f"{', '.join(aliases)} in header {fieldnames}")
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous GPS header: multiple columns resolve to "
                f"'{canonical}' -> {matches}")
        resolved[canonical] = matches[0]
    return resolved


def read_gps_csv(path: str | Path) -> list[GPSFix]:
    """Read and parse a GPS fix CSV into validated :class:`GPSFix` rows.

    Accepts any header layout whose columns can be resolved through the
    :data:`_COLUMN_ALIASES` table.  Validation of the numeric ranges is left
    to :func:`validate_fixes`.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"GPS log not found: {csv_path} — run the capture pipeline first")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = _resolve_columns(reader.fieldnames or [])
        fixes: list[GPSFix] = []
        for row in reader:
            if not any((row.get(c) or "").strip() for c in columns.values()):
                continue
            fixes.append(GPSFix(
                timestamp_s=float(row[columns["timestamp_s"]]),
                latitude=float(row[columns["latitude"]]),
                longitude=float(row[columns["longitude"]]),
                altitude_m=float(row[columns["altitude_m"]]),
            ))
    if not fixes:
        raise ValueError(f"no GPS fixes found in {csv_path}")
    return fixes


def validate_fixes(fixes: list[GPSFix]) -> list[GPSFix]:
    """Range-check every fix; raise ``ValueError`` with offending row indices.

    WGS84 constraints enforced: latitude in [-90, 90], longitude in
    [-180, 180], altitude >= -500 m (below the Dead Sea trench, comfortably
    below any drone flight), and finite non-negative timestamps.
    """
    bad: list[str] = []
    for i, fix in enumerate(fixes):
        if not math.isfinite(fix.timestamp_s) or fix.timestamp_s < 0:
            bad.append(f"row {i}: timestamp {fix.timestamp_s!r}")
        if not (-90.0 <= fix.latitude <= 90.0):
            bad.append(f"row {i}: latitude {fix.latitude!r}")
        if not (-180.0 <= fix.longitude <= 180.0):
            bad.append(f"row {i}: longitude {fix.longitude!r}")
        if not math.isfinite(fix.altitude_m) or fix.altitude_m < -500.0:
            bad.append(f"row {i}: altitude {fix.altitude_m!r}")
    if bad:
        joined = "; ".join(bad)
        raise ValueError(f"invalid GPS fixes ({len(bad)}): {joined}")
    return fixes


def wgs84_distance_m(fix_a: GPSFix, fix_b: GPSFix) -> float:
    """Great-circle surface distance between two fixes, in metres (WGS84)."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    _, _, distance = geod.inv(
        lons1=fix_a.longitude, lats1=fix_a.latitude,
        lons2=fix_b.longitude, lats2=fix_b.latitude,
    )
    return abs(distance)


def enu_origin(fixes: list[GPSFix]) -> GPSFix:
    """Return the tangent-plane origin: the first fix of the flight.

    The ENU frame is anchored at the start of the flight log so all metric
    coordinates share a stable, interpretable reference point.
    """
    return fixes[0]


_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _geodetic_to_ecef(latitude_deg: float, longitude_deg: float,
                      altitude_m: float) -> tuple[float, float, float]:
    """Convert WGS84 geodetic to Earth-Centred Earth-Fixed XYZ (metres).

    Uses the closed-form oblate-spheroid solution (Bowring-style through the
    prime-vertical radius of curvature); no iterations required.
    """
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    n_radius = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n_radius + altitude_m) * cos_lat * math.cos(lon)
    y = (n_radius + altitude_m) * cos_lat * math.sin(lon)
    z = (n_radius * (1.0 - _WGS84_E2) + altitude_m) * sin_lat
    return x, y, z


def _ecef_to_enu(ecef: tuple[float, float, float],
                 origin_ecef: tuple[float, float, float],
                 origin_latitude_deg: float,
                 origin_longitude_deg: float) -> tuple[float, float, float]:
    """Rotate an ECEF displacement into east-north-up metres.

    The rotation is built from the tangent-plane orientation at the given
    geodetic latitude/longitude, so ``east``/``north`` line up with
    increasing longitude/latitude at the origin.
    """
    lat0 = math.radians(origin_latitude_deg)
    lon0 = math.radians(origin_longitude_deg)
    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)
    dx = ecef[0] - origin_ecef[0]
    dy = ecef[1] - origin_ecef[1]
    dz = ecef[2] - origin_ecef[2]
    east = -sin_lon * dx + cos_lon * dy
    north = (-sin_lat * cos_lon * dx
             - sin_lat * sin_lon * dy
             + cos_lat * dz)
    up = (cos_lat * cos_lon * dx
          + cos_lat * sin_lon * dy
          + sin_lat * dz)
    return east, north, up


def geodetic_to_enu(fixes: list[GPSFix],
                    origin: Optional[GPSFix] = None) -> list[MetricFix]:
    """Project the fixes into an east-north-up frame anchored at ``origin``.

    The origin defaults to the first fix.  Heights (``up_m``) are measured
    relative to the origin's ellipsoid altitude, above the WGS84 ellipsoid.
    """
    if not fixes:
        raise ValueError("cannot project an empty fix list")
    tangent = origin if origin is not None else fixes[0]
    origin_ecef = _geodetic_to_ecef(
        tangent.latitude, tangent.longitude, tangent.altitude_m)
    metric: list[MetricFix] = []
    for fix in fixes:
        x, y, z = _geodetic_to_ecef(
            fix.latitude, fix.longitude, fix.altitude_m)
        east, north, up = _ecef_to_enu(
            (x, y, z), origin_ecef, tangent.latitude, tangent.longitude)
        metric.append(MetricFix(
            timestamp_s=fix.timestamp_s,
            latitude=fix.latitude,
            longitude=fix.longitude,
            altitude_m=fix.altitude_m,
            easting_m=east,
            northing_m=north,
            up_m=up,
        ))
    return metric


def utm_zone(longitude_deg: float, latitude_deg: float) -> int:
    """UTM zone number 1..60 for a WGS84 longitude/latitude pair."""
    del latitude_deg
    zone = int((longitude_deg + 180.0) // 6.0) + 1
    return min(60, max(1, zone))


def utm_hemisphere(latitude_deg: float) -> str:
    """Return the UTM hemisphere letter, ``'N'`` or ``'S'``."""
    return "N" if latitude_deg >= 0.0 else "S"


def utm_epsg_code(zone: int, latitude_deg: float) -> int:
    """EPSG code of a UTM zone: 326xx northern hemisphere, 327xx southern."""
    if not 1 <= zone <= 60:
        raise ValueError(f"UTM zone must be in 1..60, got {zone}")
    return 32600 + zone if latitude_deg >= 0.0 else 32700 + zone


def geodetic_to_utm(fixes: list[GPSFix],
                    zone: Optional[int] = None) -> tuple[list[MetricFix], int]:
    """Project the fixes into a UTM zone via pyproj (always_xy order).

    The zone defaults to the one containing the first fix; heights are kept
    relative to the first fix's ellipsoid altitude so ``up_m`` stays
    consistent with the ENU output.  Returns the metric fixes plus the zone
    actually used.
    """
    if not fixes:
        raise ValueError("cannot project an empty fix list")
    from pyproj import Transformer

    zone_used = zone if zone is not None else utm_zone(
        fixes[0].longitude, fixes[0].latitude)
    code = utm_epsg_code(zone_used, fixes[0].latitude)
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{code}", always_xy=True)
    base_alt = fixes[0].altitude_m
    metric: list[MetricFix] = []
    for fix in fixes:
        easting, northing = transformer.transform(fix.longitude, fix.latitude)
        metric.append(MetricFix(
            timestamp_s=fix.timestamp_s,
            latitude=fix.latitude,
            longitude=fix.longitude,
            altitude_m=fix.altitude_m,
            easting_m=float(easting),
            northing_m=float(northing),
            up_m=fix.altitude_m - base_alt,
        ))
    return metric, zone_used


def track_extent_m(metric: list[MetricFix]) -> float:
    """Diagonal of the metric bounding box, in metres (ground + height).

    Used to decide whether a flat ENU tangent plane is accurate enough for
    the flight: beyond a few kilometres the plane-to-ellipsoid error starts
    to matter and a projected CRS such as UTM should be preferred.
    """
    if not metric:
        return 0.0
    easts = [m.easting_m for m in metric]
    norths = [m.northing_m for m in metric]
    ups = [m.up_m for m in metric]
    de = max(easts) - min(easts)
    dn = max(norths) - min(norths)
    du = max(ups) - min(ups)
    return math.sqrt(de * de + dn * dn + du * du)


def resolve_crs(fixes: list[GPSFix], local_crs: str = "auto",
                auto_crs_max_extent_m: float = 1500.0,
                utm_zone_override: Optional[int] = None) -> tuple[str, Optional[int]]:
    """Decide the local CRS to use for ``fixes``.

    ``local_crs`` is one of ``"auto" | "enu" | "utm"``.  ``"auto"`` picks
    ENU when the ground-track extent of the fixes stays below
    ``auto_crs_max_extent_m`` (flat tangent plane is accurate there) and UTM
    otherwise.  Returns ``(crs_name, utm_zone_or_None)`` where the zone is
    given when the chosen CRS is UTM.
    """
    mode = (local_crs or "auto").strip().lower()
    if mode not in ("auto", "enu", "utm"):
        raise ValueError(
            f"unknown local_crs {local_crs!r} — expected 'auto', 'enu' or 'utm'")
    if mode == "enu":
        return "enu", None
    if mode == "utm":
        zone = utm_zone_override if utm_zone_override is not None else utm_zone(
            fixes[0].longitude, fixes[0].latitude)
        return "utm", zone
    zone = utm_zone_override if utm_zone_override is not None else utm_zone(
        fixes[0].longitude, fixes[0].latitude)
    extent = track_extent_m(geodetic_to_enu(fixes))
    return ("enu", None) if extent <= auto_crs_max_extent_m else ("utm", zone)


_METRIC_CSV_COLUMNS = (
    "timestamp_s", "latitude", "longitude", "altitude_m",
    "easting_m", "northing_m", "up_m", "crs", "zone",
)


def write_metric_csv(metric: list[MetricFix], path: str | Path,
                     crs: str, zone: Optional[str]) -> str:
    """Write the metric fixes to ``path``; returns the absolute path string.

    ``easting_m/northing_m/up_m`` are in ``crs`` (``'enu'`` or ``'utm'``),
    with ``zone`` recording the UTM zone used (empty for ENU).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_METRIC_CSV_COLUMNS)
        zone_cell = zone if zone is not None else ""
        for m in metric:
            writer.writerow([
                f"{m.timestamp_s:.6f}", m.latitude, m.longitude,
                m.altitude_m, f"{m.easting_m:.6f}", f"{m.northing_m:.6f}",
                f"{m.up_m:.6f}", crs, zone_cell,
            ])
    return str(out.resolve())


def write_report_json(result: GpsResult, path: str | Path) -> str:
    """Persist the conversion report as JSON; returns the absolute path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source": result.source,
        "n_fixes": result.n_fixes,
        "crs": result.crs,
        "zone": result.zone if result.zone is not None else "",
        "datum": "WGS84",
        "origin": {
            "timestamp_s": result.origin.timestamp_s,
            "latitude": result.origin.latitude,
            "longitude": result.origin.longitude,
            "altitude_m": result.origin.altitude_m,
        },
        "extent_m": round(result.extent_m, 3),
        "metric_csv": result.metric_csv,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(out.resolve())