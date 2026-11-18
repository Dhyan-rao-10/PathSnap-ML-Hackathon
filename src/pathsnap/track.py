"""GPS track cleaning, segmentation and kinematics."""
import numpy as np
import pandas as pd

from . import config as C
from .geo import bearing_deg


def clean_track(raw):
    """raw: [n,3] lon, lat, t. Drops invalid/duplicate fixes, sorts by time.
    Returns (clean array, report dict)."""
    rep = {"n_raw": len(raw)}
    ok = np.isfinite(raw).all(1) & (np.abs(raw[:, 1]) <= 90) & (np.abs(raw[:, 0]) <= 180)
    rep["n_invalid"] = int((~ok).sum())
    tr = raw[ok]
    order = np.argsort(tr[:, 2], kind="stable")
    rep["n_out_of_order"] = int((np.diff(tr[:, 2]) < 0).sum())
    tr = tr[order]
    keep = np.ones(len(tr), bool)
    keep[1:] = np.diff(tr[:, 2]) > 0
    rep["n_dup_time"] = int((~keep).sum())
    tr = tr[keep]
    rep["n_clean"] = len(tr)
    return tr, rep


def build_track_frame(tr, proj):
    """Projected track with segment ids, speed and heading."""
    x, y = proj.to_xy(tr[:, 0], tr[:, 1])
    t = tr[:, 2]
    df = pd.DataFrame({"lon": tr[:, 0], "lat": tr[:, 1], "t": t, "x": x, "y": y})
    dt = np.diff(t, prepend=np.nan)
    ds = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = ds / dt
    brk = (dt > C.GAP_BREAK_S) | (v > C.MAX_SPEED_MS)
    brk[0] = True
    df["dt"] = np.nan_to_num(dt, nan=0.0)
    df["step_m"] = np.nan_to_num(ds, nan=0.0)
    df["seg"] = np.cumsum(brk) - 1

    # kinematics from neighbours i-k .. i+k clipped to the same segment
    n = len(df)
    seg = df["seg"].to_numpy()
    seg_start = np.zeros(n, int); seg_end = np.zeros(n, int)
    for s, idx in df.groupby("seg").indices.items():
        seg_start[idx] = idx[0]; seg_end[idx] = idx[-1]
    k = C.HEADING_HALF_WINDOW
    i = np.arange(n)
    lo = np.maximum(i - k, seg_start)
    hi = np.minimum(i + k, seg_end)
    dx, dy = x[hi] - x[lo], y[hi] - y[lo]
    span_t = np.maximum(t[hi] - t[lo], 1e-6)
    df["speed"] = np.where(hi > lo, np.hypot(dx, dy) / span_t, 0.0)
    df["heading"] = bearing_deg(dx, dy)
    df["vx"] = np.where(hi > lo, dx / span_t, 0.0)
    df["vy"] = np.where(hi > lo, dy / span_t, 0.0)
    df["seg_start"] = seg_start
    df["seg_end"] = seg_end
    return df
