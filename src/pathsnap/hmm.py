"""HMM map matching decoded with Viterbi.

Hidden state at fix i   : one of the fix's candidate arcs (top-M by classifier prob),
                          together with the projected position along that arc.
Emission  log P(z_i|s)  : w_e * log( p_clf(s) / sum_{s'} p_clf(s') )
                          -- the trained classifier's probability, normalised over
                          the fix's candidate set.
Transition log P(s'|s)  : -| d_route(s -> s') - d_gps(i-1 -> i) | / beta
                          (Newson & Krumm 2009): a true move along the road network
                          should be about as long as the straight-line GPS move.
                          d_route uses directed shortest paths on the arc graph,
                          so wrong-way moves and jumps to unconnected roads are expensive.
Unreachable transitions : constant penalty (HMM_UNREACHABLE_LOGP) instead of -inf so
                          one bad fix cannot make the whole sequence impossible.
GPS gaps / jumps        : the track is split into segments (config.GAP_BREAK_S,
                          MAX_SPEED_MS); Viterbi restarts in every segment.
Fixes with no candidates: skipped (predicted arc = -1); the chain links the previous
                          and next fixes directly.
"""
import numpy as np

from . import config as C


def _route_dist(net, a, pa, b, pb, limit, back_tol):
    """Network distance from position pa on arc a to position pb on arc b (metres)."""
    if a == b and pb >= pa - back_tol:
        return max(pb - pa, 0.0)
    rest = net.length[a] - pa
    nd = net.node_dist(int(net.a_to[a]), int(net.a_from[b]), limit)
    return rest + nd + pb


def viterbi(df, cand_by_pid, net, beta=C.HMM_BETA, w_emit=C.HMM_EMISSION_WEIGHT,
            unreachable=C.HMM_UNREACHABLE_LOGP, back_tol=C.HMM_SAME_ARC_BACK_TOL_M):
    """cand_by_pid: dict pid -> (arcs[int], log_emission[float], pos_along_arc[float]).
    Returns (pred_arc per fix, info dict)."""
    n = len(df)
    pred = np.full(n, -1, dtype=np.int64)
    x = df["x"].to_numpy(); y = df["y"].to_numpy(); t = df["t"].to_numpy()
    seg = df["seg"].to_numpy()
    n_unreach_chosen = 0
    n_chains = 0

    chain_pids, chain_bp, chain_states = [], [], []
    delta = None

    def finish():
        nonlocal n_chains
        if not chain_pids:
            return
        n_chains += 1
        j = int(np.argmax(delta))
        for k in range(len(chain_pids) - 1, -1, -1):
            pred[chain_pids[k]] = chain_states[k][j]
            if k > 0:
                j = int(chain_bp[k][j])

    prev = None
    for i in range(n):
        c = cand_by_pid.get(i)
        if c is None or len(c[0]) == 0:
            continue
        arcs, le, pos = c
        emit = w_emit * le
        if prev is None or seg[i] != seg[prev]:
            finish()
            chain_pids, chain_bp, chain_states = [i], [None], [arcs]
            delta = emit.copy()
            prev = i
            continue
        parcs, _, ppos = cand_by_pid[prev]
        gc = float(np.hypot(x[i] - x[prev], y[i] - y[prev]))
        dt = float(t[i] - t[prev])
        limit = min(5000.0, 2.0 * gc + 200.0 + 20.0 * max(dt - 1.0, 0.0))
        logT = np.empty((len(parcs), len(arcs)))
        for a_i in range(len(parcs)):
            for b_i in range(len(arcs)):
                rd = _route_dist(net, parcs[a_i], ppos[a_i], arcs[b_i], pos[b_i], limit, back_tol)
                logT[a_i, b_i] = -abs(rd - gc) / beta if np.isfinite(rd) else unreachable
        scores = delta[:, None] + logT
        bp = np.argmax(scores, axis=0)
        delta = scores[bp, np.arange(len(arcs))] + emit
        n_unreach_chosen += int(np.sum(logT[bp, np.arange(len(arcs))] <= unreachable))
        chain_pids.append(i); chain_bp.append(bp); chain_states.append(arcs)
        prev = i
    finish()
    return pred, {"n_chains": n_chains, "n_unreachable_transitions": n_unreach_chosen}


def stitch_route(pred, df, net, max_gap_m=5000.0):
    """Turn per-fix arcs into a continuous arc sequence by inserting shortest
    paths between consecutive distinct arcs. Returns (route_arcs, n_breaks)."""
    seg = df["seg"].to_numpy()
    route, breaks = [], 0
    last, last_seg = None, None
    for i, a in enumerate(pred):
        if a < 0 or a == last:
            continue
        a = int(a)
        if last is None:
            route.append(a)
        elif net.a_to[last] == net.a_from[a]:
            route.append(a)
        else:
            path = None
            if seg[i] == last_seg:
                path = net.node_path_arcs(int(net.a_to[last]), int(net.a_from[a]), max_gap_m)
            if path is None:
                breaks += 1
                route.append(a)
            else:
                route.extend(path)
                route.append(a)
        last, last_seg = a, seg[i]
    return route, breaks


def build_cand_dict(cands, max_states=C.HMM_MAX_STATES, prob_col="prob"):
    """cands: DataFrame(pid, arc, prob, proj_t, arc_len) -> dict used by viterbi()."""
    c = cands.sort_values(["pid", prob_col], ascending=[True, False])
    c = c.groupby("pid", sort=False).head(max_states)
    out = {}
    pid = c["pid"].to_numpy(); arc = c["arc"].to_numpy()
    p = c[prob_col].to_numpy(); pos = (c["proj_t"] * c["arc_len"]).to_numpy()
    bounds = np.flatnonzero(np.diff(pid)) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(pid)]):
        pp = np.maximum(p[s:e], 1e-9)
        out[int(pid[s])] = (arc[s:e].astype(np.int64), np.log(pp / pp.sum()), pos[s:e])
    return out
