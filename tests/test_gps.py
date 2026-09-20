"""STEP 7 — GPS parsing and metric conversion tests.

Ground truths:
* Eiffel Tower  (48.8583701, 2.2944813) to Louvre (48.8606111, 2.3376440)
  is 3350-3360 m on the WGS84 ellipsoid.
* 9e-5 deg of latitude ~ 10 m; at 47.6N, 9e-5 deg of longitude ~ 6.8 m.
* Synthetic flight: origin (47.6062, -122.3321, 100.0), steps north+east,
  +0.4 m altitude per fix -> ENU extent ~1.2 km (stays ENU under auto).
"""

from __future__ import annotations

import json

import pytest

from src.georef.gps import (
    GPSFix,
    GpsResult,
    MetricFix,
    enu_origin,
    geodetic_to_enu,
    geodetic_to_utm,
    read_gps_csv,
    resolve_crs,
    run_gps_conversion,
    track_extent_m,
    utm_epsg_code,
    utm_hemisphere,
    utm_zone,
    validate_fixes,
    wgs84_distance_m,
)

#: Eiffel Tower (left) and Louvre glass pyramid (right), WGS84.
REF_A = GPSFix(0.0, 48.8583701, 2.2944813, 33.0)
REF_B = GPSFix(1.0, 48.8606111, 2.3376440, 35.0)


def synthetic_flight(n=100, lat0=47.6062, lon0=-122.3321, alt0=100.0):
    fixes = []
    for i in range(n):
        fixes.append(GPSFix(
            timestamp_s=float(i) * 0.1,
            latitude=lat0 + 9e-5 * i,
            longitude=lon0 + 9e-5 * i,
            altitude_m=alt0 + 0.4 * i,
        ))
    return fixes


def write_gps_csv(tmp_path, rows, headers=("timestamp", "latitude", "longitude", "altitude")):
    path = tmp_path / "gps.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(headers) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    return path


def test_gps_fix_contract_is_frozen():
    fix = GPSFix(0.0, 48.85, 2.29, 33.0)
    assert fix.timestamp_s == 0.0
    assert fix.latitude == 48.85
    assert fix.longitude == 2.29
    assert fix.altitude_m == 33.0
    with pytest.raises(Exception):
        fix.latitude = 1.0  # frozen dataclass


def test_read_csv_with_canonical_columns(tmp_path):
    path = write_gps_csv(tmp_path, [(0.0, 48.85, 2.29, 33.0),
                                    (0.1, 48.86, 2.30, 34.0)])
    fixes = read_gps_csv(path)
    assert len(fixes) == 2
    assert fixes[0] == GPSFix(0.0, 48.85, 2.29, 33.0)
    assert fixes[1] == GPSFix(0.1, 48.86, 2.30, 34.0)


def test_read_csv_accepts_aliased_columns(tmp_path):
    path = write_gps_csv(tmp_path,
                         [(0.0, 48.85, 2.29, 33.0)],
                         headers=("t", "lat", "lng", "alt"))
    fixes = read_gps_csv(path)
    assert fixes == [GPSFix(0.0, 48.85, 2.29, 33.0)]


def test_read_csv_missing_column_raises(tmp_path):
    path = write_gps_csv(tmp_path, [(0.0, 48.85, 2.29)],
                         headers=("t", "lat", "lng"))
    with pytest.raises(ValueError, match="altitude"):
        read_gps_csv(path)


def test_validate_fixes_accepts_valid_bounds():
    ok = [REF_A, GPSFix(0.0, 90.0, -180.0, -500.0),
          GPSFix(0.0, -90.0, 180.0, 60000.0)]
    assert validate_fixes(ok) == ok


def test_validate_fixes_rejects_out_of_range():
    bad = [GPSFix(-1.0, 0.0, 0.0, 0.0),          # negative timestamp
           GPSFix(0.0, 91.0, 0.0, 0.0),          # latitude too far north
           GPSFix(0.0, 0.0, 181.0, 0.0),         # longitude too far east
           GPSFix(0.0, 0.0, 0.0, -600.0)]        # altitude below cut-off
    with pytest.raises(ValueError, match="invalid GPS fixes"):
        validate_fixes(bad)