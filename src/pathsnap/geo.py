"""Geodesy helpers.

Lon/lat are angles, so we project every trip into its local UTM zone (metres).
Trips span at most a few tens of km, well inside one zone, so UTM distortion is
< 0.1% -- far below GPS noise (~5-30 m).
"""
import numpy as np
from pyproj import Transformer


class LocalProjection:
    def __init__(self, lon0, lat0):
        zone = int((lon0 + 180.0) // 6.0) + 1
        epsg = (32600 if lat0 >= 0 else 32700) + zone
        self.epsg = epsg
        self._fwd = Transformer.from_crs(4326, epsg, always_xy=True)
        self._inv = Transformer.from_crs(epsg, 4326, always_xy=True)

    @classmethod
    def for_points(cls, lonlat):
        lon0, lat0 = np.median(lonlat[:, 0]), np.median(lonlat[:, 1])
        return cls(lon0, lat0)

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(x, float), np.asarray(y, float)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x), np.asarray(y))
        return np.asarray(lon, float), np.asarray(lat, float)


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371008.8
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment distance.
    Returns (distance, t) where t in [0,1] is the projection fraction along a->b."""
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * vx + (py - ay) * vy) / l2
    t = np.where(l2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = ax + t * vx, ay + t * vy
    return np.hypot(px - qx, py - qy), t


def angle_diff_deg(a, b):
    """Smallest absolute difference between two compass bearings (degrees)."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) of a displacement in projected metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
