"""Candidate generation + per-(fix, candidate arc) feature engineering.

All features are relative/trip-agnostic (no absolute coordinates), because each
trip is in a different city and the evaluator trips are unseen cities.
"""
import numpy as np
import pandas as pd

from . import config as C
from .geo import point_segment, angle_diff_deg

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "ang_diff", "cos_diff", "speed", "along_speed", "cross_speed",
    "arc_len", "proj_t", "d_endpoint", "out_deg_end", "in_deg_start",
    "is_twoway", "ctx_dist_mean", "ctx_excess_mean", "ctx_ang_mean", "dt_prev",
]


def candidate_table(df, net, radius=C.CAND_RADIUS_M, k=C.CAND_K):
    """Long table: one row per (fix, candidate arc) with engineered features."""
    x = df["x"].to_numpy(); y = df["y"].to_numpy()
    arcs, dist, tproj, fallback = net.candidates(x, y, radius=radius, k=k)
    n = len(df)
    pid = np.repeat(np.arange(n), k)
    rank = np.tile(np.arange(k), n)
    a = arcs.ravel(); d = dist.ravel(); tp = tproj.ravel()
    keep = a >= 0
    pid, rank, a, d, tp = pid[keep], rank[keep], a[keep], d[keep], tp[keep]
    n_cand = (arcs >= 0).sum(1)
    dmin = np.where(np.isfinite(dist[:, 0]), dist[:, 0], np.nan)

    out = pd.DataFrame({"pid": pid, "arc": a})
    out["dist"] = d
    out["dist_rank"] = rank
    out["dist_minus_min"] = d - dmin[pid]
    out["dist_ratio"] = d / (dmin[pid] + 1.0)
    out["n_cand"] = n_cand[pid]
    out["fallback"] = fallback[pid]

    heading = df["heading"].to_numpy()[pid]
    speed = df["speed"].to_numpy()[pid]
    vx = df["vx"].to_numpy()[pid]; vy = df["vy"].to_numpy()[pid]
    ang = angle_diff_deg(heading, net.bearing[a])
    out["ang_diff"] = ang
    out["cos_diff"] = np.cos(np.radians(ang))
    out["speed"] = speed
    ux = (net.bx - net.ax)[a] / net.length[a]; uy = (net.by - net.ay)[a] / net.length[a]
    out["along_speed"] = vx * ux + vy * uy
    out["cross_speed"] = np.abs(-vx * uy + vy * ux)
    out["arc_len"] = net.length[a]
    out["proj_t"] = tp
    out["d_endpoint"] = np.minimum(tp, 1 - tp) * net.length[a]
    out["out_deg_end"] = net.out_deg[net.a_to[a]]
    out["in_deg_start"] = net.in_deg[net.a_from[a]]
    out["is_twoway"] = (net.twin[a] >= 0).astype(np.int8)

    # temporal context: how well does this arc explain neighbouring fixes?
    seg_start = df["seg_start"].to_numpy(); seg_end = df["seg_end"].to_numpy()
    hd_all = df["heading"].to_numpy(); sp_all = df["speed"].to_numpy()
    W = C.CTX_HALF_WINDOW
    s_d = np.zeros(len(out)); s_ex = np.zeros(len(out)); s_ang = np.zeros(len(out))
    cnt = np.zeros(len(out)); cnt_ang = np.zeros(len(out))
    for o in [o for o in range(-W, W + 1) if o != 0]:
        j = pid + o
        ok = (j >= seg_start[pid]) & (j <= seg_end[pid])
        jj = np.where(ok, j, pid)
        dj, _ = point_segment(x[jj], y[jj], net.ax[a], net.ay[a], net.bx[a], net.by[a])
        ex = dj - np.nan_to_num(dmin[jj], nan=dj)
        s_d += np.where(ok, dj, 0); s_ex += np.where(ok, ex, 0); cnt += ok
        moving = ok & (sp_all[jj] > 2.0)
        s_ang += np.where(moving, angle_diff_deg(hd_all[jj], net.bearing[a]), 0); cnt_ang += moving
    out["ctx_dist_mean"] = np.where(cnt > 0, s_d / np.maximum(cnt, 1), d)
    out["ctx_excess_mean"] = np.where(cnt > 0, s_ex / np.maximum(cnt, 1), d - dmin[pid])
    out["ctx_ang_mean"] = np.where(cnt_ang > 0, s_ang / np.maximum(cnt_ang, 1), 90.0)
    out["dt_prev"] = df["dt"].to_numpy()[pid]
    return out
