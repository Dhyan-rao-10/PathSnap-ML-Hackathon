"""Derive a reference arc for every GPS fix from the (arc-level) reference route.

The .route file lists arcs in driving order but does not say which fix belongs to
which arc. We solve a monotone alignment by dynamic programming: assign each fix
i a route position j_i with j_0 <= j_1 <= ... minimising sum_i dist(fix_i, arc_{j_i}).
Monotonicity respects driving order and handles routes that revisit an arc.

Used ONLY for training labels and evaluation -- never inside prediction.
"""
import numpy as np

from . import config as C
from .geo import point_segment


def align_track_to_route(df, net, route):
    m = len(route)
    ax, ay = net.ax[route], net.ay[route]
    bx, by = net.bx[route], net.by[route]
    n = len(df)
    x = df["x"].to_numpy(); y = df["y"].to_numpy()
    bp_dtype = np.int16 if m < 32000 else np.int32
    bp = np.empty((n, m), dtype=bp_dtype)
    prev = None
    for i in range(n):
        d, _ = point_segment(x[i], y[i], ax, ay, bx, by)
        if prev is None:
            cost = d
            bp[i] = np.arange(m)
        else:
            # prefix minimum of previous costs (j' <= j) and its argmin
            pm = np.minimum.accumulate(prev)
            idx = np.arange(m)
            is_new = np.concatenate([[True], prev[1:] < pm[:-1]])
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            cost = d + pm
            bp[i] = arg
        prev = cost
    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        j[i - 1] = bp[i, j[i]]
    ref = route[j]
    d_ref, t_ref = point_segment(x, y, net.ax[ref], net.ay[ref], net.bx[ref], net.by[ref])
    return {
        "route_pos": j,
        "ref_arc": ref,
        "ref_dist": d_ref,
        "ref_t": t_ref,
        "label_valid": d_ref <= C.LABEL_MAX_DIST_M,
    }


def route_continuity(route, net):
    """Fraction of consecutive reference arcs that are physically connected."""
    if len(route) < 2:
        return 1.0
    return float(np.mean(net.a_to[route[:-1]] == net.a_from[route[1:]]))
