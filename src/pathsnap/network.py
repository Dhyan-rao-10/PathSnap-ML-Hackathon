"""Directed road network: geometry, spatial index, candidate search, bounded Dijkstra."""
import heapq
import numpy as np
from scipy.spatial import cKDTree

from . import config as C
from .geo import point_segment, bearing_deg


class RoadNetwork:
    def __init__(self, nodes_lonlat, arcs, proj):
        self.proj = proj
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        self.nodes_lonlat = nodes_lonlat
        self.nx, self.ny = proj.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.a_from = arcs[:, 0].astype(np.int64)
        self.a_to = arcs[:, 1].astype(np.int64)
        self.ax, self.ay = self.nx[self.a_from], self.ny[self.a_from]
        self.bx, self.by = self.nx[self.a_to], self.ny[self.a_to]
        self.length = np.maximum(np.hypot(self.bx - self.ax, self.by - self.ay), 0.01)
        self.bearing = bearing_deg(self.bx - self.ax, self.by - self.ay)
        self.out_deg = np.bincount(self.a_from, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.a_to, minlength=self.n_nodes)
        # reverse twin arc (two-way street) lookup
        key = self.a_from * self.n_nodes + self.a_to
        rkey = self.a_to * self.n_nodes + self.a_from
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        hit = key[order][pos] == rkey
        self.twin = np.where(hit, order[pos], -1)
        # CSR adjacency (outgoing arcs per node)
        o = np.argsort(self.a_from, kind="stable")
        self._csr_arc = o
        self._csr_to = self.a_to[o]
        self._csr_w = self.length[o]
        self._indptr = np.concatenate([[0], np.cumsum(self.out_deg)])
        self._dij_cache = {}
        self._kd = None

    # ------------------------------------------------------------------ index
    def build_index(self, step=C.INDEX_STEP_M):
        n = np.maximum(np.ceil(self.length / step).astype(int), 1) + 1
        arc_id = np.repeat(np.arange(self.n_arcs), n)
        start = np.repeat(np.cumsum(n) - n, n)
        k = np.arange(len(arc_id)) - start
        frac = k / np.repeat(n - 1, n)
        sx = self.ax[arc_id] + frac * (self.bx - self.ax)[arc_id]
        sy = self.ay[arc_id] + frac * (self.by - self.ay)[arc_id]
        self._kd = cKDTree(np.column_stack([sx, sy]))
        self._kd_arc = np.append(arc_id, -1)  # sentinel for "no neighbour"
        self._step = step

    def candidates(self, px, py, radius=C.CAND_RADIUS_M, k=C.CAND_K,
                   radius_max=C.CAND_RADIUS_MAX_M, n_query=96):
        """Top-k nearest arcs (by exact perpendicular distance) within `radius`
        for each point. Points with nothing in range fall back to the nearest
        arcs within `radius_max`; if still none, the point has no candidates.

        Returns (arc_ids [n,k] with -1 padding, dist [n,k], t [n,k], fallback[n])."""
        if self._kd is None:
            self.build_index()
        pts = np.column_stack([px, py])
        arcs, d, t = self._query(pts, radius, k, n_query)
        fallback = np.zeros(len(px), bool)
        empty = np.where(arcs[:, 0] < 0)[0]
        if len(empty):
            a2, d2, t2 = self._query(pts[empty], radius_max, k, n_query)
            arcs[empty], d[empty], t[empty] = a2, d2, t2
            fallback[empty] = a2[:, 0] >= 0
        return arcs, d, t, fallback

    def _query(self, pts, radius, k, n_query):
        n = len(pts)
        _, idx = self._kd.query(pts, k=n_query, distance_upper_bound=radius + self._step)
        cand = self._kd_arc[np.minimum(idx, len(self._kd_arc) - 1)]
        valid = cand >= 0
        safe = np.where(valid, cand, 0)
        px = pts[:, 0:1]; py = pts[:, 1:2]
        dist, t = point_segment(px, py, self.ax[safe], self.ay[safe], self.bx[safe], self.by[safe])
        dist = np.where(valid, dist, np.inf)
        # de-duplicate arcs within a row
        o = np.argsort(np.where(valid, cand, np.iinfo(np.int64).max), axis=1, kind="stable")
        cs = np.take_along_axis(cand, o, 1)
        dup = np.zeros_like(cs, bool)
        dup[:, 1:] = (cs[:, 1:] == cs[:, :-1]) & (cs[:, 1:] >= 0)
        dup_orig = np.zeros_like(dup)
        np.put_along_axis(dup_orig, o, dup, 1)
        dist = np.where(dup_orig | (dist > radius), np.inf, dist)
        top = np.argsort(dist, axis=1, kind="stable")[:, :k]
        d_top = np.take_along_axis(dist, top, 1)
        a_top = np.where(np.isfinite(d_top), np.take_along_axis(cand, top, 1), -1)
        t_top = np.take_along_axis(t, top, 1)
        if a_top.shape[1] < k:  # pad
            padn = k - a_top.shape[1]
            a_top = np.pad(a_top, ((0, 0), (0, padn)), constant_values=-1)
            d_top = np.pad(d_top, ((0, 0), (0, padn)), constant_values=np.inf)
            t_top = np.pad(t_top, ((0, 0), (0, padn)), constant_values=0)
        return a_top, d_top, t_top

    # ------------------------------------------------------------- routing
    def _dijkstra(self, src, cutoff):
        dist = {src: 0.0}
        pred = {}
        heap = [(0.0, src)]
        indptr, to, w, arc = self._indptr, self._csr_to, self._csr_w, self._csr_arc
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for e in range(indptr[u], indptr[u + 1]):
                v = to[e]
                nd = d + w[e]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = arc[e]
                    heapq.heappush(heap, (nd, v))
        return dist, pred

    def node_dist(self, u, v, limit):
        """Shortest directed network distance u->v (metres), inf if > limit."""
        if u == v:
            return 0.0
        c = self._dij_cache.get(u)
        if c is None or c[0] < limit:
            cutoff = limit if c is None else max(limit, 2 * c[0])
            dist, pred = self._dijkstra(u, cutoff)
            c = (cutoff, dist, pred)
            self._dij_cache[u] = c
        d = c[1].get(v, np.inf)
        return d if d <= limit else np.inf

    def node_path_arcs(self, u, v, limit=5000.0):
        """Arc ids of the shortest path u->v (empty if u==v, None if unreachable)."""
        if u == v:
            return []
        found = False
        for lim in (300.0, 1500.0, limit):  # grow the search only when needed
            if np.isfinite(self.node_dist(u, v, min(lim, limit))):
                found = True
                break
        if not found:
            return None
        pred = self._dij_cache[u][2]
        path, cur = [], v
        while cur != u:
            a = pred[cur]
            path.append(int(a))
            cur = self.a_from[a]
        return path[::-1]

    def connected(self, a, b):
        return a == b or self.a_to[a] == self.a_from[b]
