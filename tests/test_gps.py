"""STEP 7 — GPS parsing and metric conversion tests.

Ground truths:
* Eiffel Tower  (48.8583701, 2.2944813) to Louvre (48.8606111, 2.3376440)
  is ~3177 m on the WGS84 ellipsoid.
* 9e-5 deg of latitude ~ 10 m; 9e-5 deg of longitude ~ 6.8 m at 48.9N.
* Paris sits in UTM zone 31 (central meridian 3E), so easting ~ 500 km.
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


def test_wgs84_distance_eiffel_to_louvre():
    distance = wgs84_distance_m(REF_A, REF_B)
    assert 3150.0 <= distance <= 3200.0


def test_geodetic_to_enu_origin_at_first_fix():
    metric = geodetic_to_enu(synthetic_flight(3))
    origin = metric[0]
    assert abs(origin.easting_m) < 1e-6
    assert abs(origin.northing_m) < 1e-6
    assert abs(origin.up_m) < 1e-6
    assert metric[0].latitude == metric[0].latitude  # provenance preserved


def test_geodetic_to_enu_direction_and_scale():
    metric = geodetic_to_enu(synthetic_flight(3))
    forward = metric[1]
    assert 5.5 <= forward.easting_m <= 8.0       # 9e-5 deg lon ~ 6.8 m at 47.6N
    assert 9.5 <= forward.northing_m <= 10.5     # 9e-5 deg lat ~ 10 m
    assert forward.up_m == pytest.approx(0.4, abs=1e-3)


def test_utm_zone_lookup_and_epsg():
    assert utm_zone(13.0, 52.0) == 33
    assert utm_zone(-75.0, 40.0) == 18
    assert utm_zone(151.2, -33.0) == 56
    assert utm_zone(179.9, 0.0) == 60
    assert utm_zone(-179.9, 0.0) == 1
    assert utm_hemisphere(48.85) == "N"
    assert utm_hemisphere(-33.0) == "S"
    assert utm_epsg_code(33, 48.85) == 32633
    assert utm_epsg_code(18, -40.0) == 32718


def test_utm_projection_zone_and_plausible_easting():
    fixes = [REF_A, GPSFix(1.0, 48.8583701, 2.2944813 + 9e-5, 33.4)]
    metric, zone = geodetic_to_utm(fixes, zone=31)   # Paris sits in zone 31
    assert zone == 31
    assert 400000 <= metric[0].easting_m <= 600000   # central meridian 3E
    east_delta = metric[1].easting_m - metric[0].easting_m
    assert 2.0 <= east_delta <= 8.0                  # ~6.6 m for a pure east step
    assert metric[1].up_m == pytest.approx(0.4, abs=1e-3)


def test_track_extent_m():
    metric = geodetic_to_enu(synthetic_flight(100))
    extent = track_extent_m(metric)
    assert 1180.0 <= extent <= 1230.0


def test_resolve_crs_auto_picks_enu_for_short_track():
    crs, zone = resolve_crs(synthetic_flight(100), "auto", 1500.0)
    assert crs == "enu"
    assert zone is None


def test_resolve_crs_auto_switches_to_utm_for_large_extent():
    crs, zone = resolve_crs(synthetic_flight(100), "auto", 100.0)
    assert crs == "utm"
    assert zone == 10


def test_resolve_crs_explicit_modes_and_override():
    assert resolve_crs([REF_A], "enu") == ("enu", None)
    crs, zone = resolve_crs([REF_A], "utm")
    assert crs == "utm" and zone == 31                 # lon 2.29 -> zone 31
    crs, zone = resolve_crs([REF_A], "utm", utm_zone_override=18)
    assert (crs, zone) == ("utm", 18)
    with pytest.raises(ValueError, match="local_crs"):
        resolve_crs([REF_A], "mars", 1500.0)


def test_run_gps_conversion_writes_metric_csv_and_report(tmp_path):
    path = write_gps_csv(tmp_path, [
        (0.0, 47.6062, -122.3321, 100.0),
        (0.1, 47.6062 + 9e-5, -122.3321 + 9e-5, 100.4),
    ])
    cfg = {"paths": {"reports": str(tmp_path / "reports")}}
    result = run_gps_conversion(cfg, gps_file=path, output_dir=tmp_path / "geo")

    assert isinstance(result, GpsResult)
    assert result.n_fixes == 2
    assert result.crs == "enu"            # tiny flight -> ENU under auto
    assert result.zone is None
    assert result.origin.latitude == pytest.approx(47.6062)

    csv_path = tmp_path / "geo" / "gps_metric.csv"
    report_path = tmp_path / "reports" / "gps_report.json"
    assert result.metric_csv.endswith("gps_metric.csv")
    assert result.report_json.endswith("gps_report.json")
    assert csv_path.is_file() and report_path.is_file()

    header, *rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert header.split(",") == [
        "timestamp_s", "latitude", "longitude", "altitude_m",
        "easting_m", "northing_m", "up_m", "crs", "zone"]
    assert rows[0] == "0.000000,47.6062,-122.3321,100.0,0.000000,0.000000,0.000000,enu,"
    first_east = float(rows[1].split(",")[4])
    assert 2.0 <= first_east <= 8.0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["crs"] == "enu"
    assert report["n_fixes"] == 2
    assert report["datum"] == "WGS84"