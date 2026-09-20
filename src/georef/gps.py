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

from dataclasses import dataclass
from typing import Optional


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