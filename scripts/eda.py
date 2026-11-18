"""Exploratory data analysis on the raw TRAIN and STUDENT-TEST partitions.

Outputs -> results/eda/: inventory CSV, summary stats, plots and Folium maps.
Usage: python scripts/eda.py
"""
import os

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import config as C, io
from pathsnap.geo import haversine_m
from pathsnap.trip import Trip
from pathsnap.visualize import trip_map

OUT = C.RESULTS / "eda"
MAP_TRIPS = {"train": ["00000003", "00000021", "00000090"], "student_test": ["00000009"]}


def raw_inventory():
    rows, gaps = [], []
    for part, pdir in [("train", C.TRAIN_DIR), ("student_test", C.STUDENT_TEST_DIR)]:
        for tid in io.list_trips(pdir):
            d = pdir / tid
            files = {ext: (d / f"{tid}.{ext}") for ext in ["track", "nodes", "arcs", "route"]}
            raw = io.load_track(d)
            nodes, arcs = io.load_network(d)
            route = io.load_route(d)
            lon, lat, t = raw[:, 0], raw[:, 1], raw[:, 2]
            dt = np.diff(t)
            step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
            n = len(nodes)
            g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
            n_weak, _ = connected_components(g, directed=True, connection="weak")
            n_strong, lab_s = connected_components(g, directed=True, connection="strong")
            deg = np.bincount(arcs.ravel(), minlength=n)
            largest_scc = np.bincount(lab_s).max() / n
            route_ok = route is not None and len(route) > 1
            rows.append({
                "partition": part, "trip": tid,
                **{f"has_{k}": v.exists() for k, v in files.items()},
                **{f"size_kb_{k}": round(v.stat().st_size / 1024, 1) for k, v in files.items() if v.exists()},
                "n_gps": len(raw), "n_nan": int(np.isnan(raw).sum()),
                "n_nonmonotonic_t": int((dt <= 0).sum()),
                "duration_min": (t.max() - t.min()) / 60,
                "dt_median": float(np.median(dt)), "dt_max": float(dt.max()),
                "frac_dt_eq_1": float(np.mean(dt == 1)), "n_gaps_gt_10s": int((dt > 10).sum()),
                "n_jumps_gt_70ms": int((step / np.maximum(dt, 1e-9) > 70).sum()),
                "length_km": float(step.sum() / 1000),
                "extent_km_x": float(haversine_m(lon.min(), lat.mean(), lon.max(), lat.mean()) / 1000),
                "extent_km_y": float(haversine_m(lon.mean(), lat.min(), lon.mean(), lat.max()) / 1000),
                "centroid_lon": float(np.median(lon)), "centroid_lat": float(np.median(lat)),
                "n_nodes": n, "n_arcs": len(arcs), "n_isolated_nodes": int((deg == 0).sum()),
                "n_weak_components": int(n_weak), "n_strong_components": int(n_strong),
                "largest_scc_frac": float(largest_scc),
                "n_route_arcs": len(route) if route is not None else 0,
                "route_continuity": float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0])) if route_ok else np.nan,
                "route_repeated_arcs": int(len(route) - len(np.unique(route))) if route_ok else 0,
            })
            gaps.append(pd.DataFrame({"partition": part, "trip": tid, "dt": dt,
                                      "speed": step / np.maximum(dt, 1e-9)}))
    return pd.DataFrame(rows), pd.concat(gaps, ignore_index=True)


def plots(inv, gaps):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for part, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        s = inv[inv.partition == part]
        ax[0, 0].hist(s.n_gps, bins=30, alpha=.6, color=c, label=part)
        ax[0, 1].hist(np.log10(s.n_arcs), bins=30, alpha=.6, color=c, label=part)
        ax[0, 2].hist(s.length_km, bins=30, alpha=.6, color=c, label=part)
        g = gaps[gaps.partition == part]
        ax[1, 0].hist(np.clip(g.dt, 0, 20), bins=40, alpha=.6, color=c, label=part, log=True)
        ax[1, 1].hist(np.clip(g.speed, 0, 60), bins=60, alpha=.6, color=c, label=part, density=True)
    ax[0, 0].set(title="GPS fixes per trip", xlabel="fixes")
    ax[0, 1].set(title="Road-network size", xlabel="log10(#arcs)")
    ax[0, 2].set(title="Trajectory length", xlabel="km")
    ax[1, 0].set(title="Sampling interval (clipped at 20s)", xlabel="dt (s)", ylabel="count (log)")
    ax[1, 1].set(title="Implied speed between fixes", xlabel="m/s (clipped 60)")
    ax[1, 2].scatter(inv.centroid_lon, inv.centroid_lat,
                     c=inv.partition.map({"train": "tab:blue", "student_test": "tab:orange"}), s=20)
    ax[1, 2].set(title="Trip locations (centroids)", xlabel="lon", ylabel="lat")
    for a in ax.ravel()[:5]:
        a.legend()
    fig.tight_layout(); fig.savefig(OUT / "eda_overview.png", dpi=120); plt.close(fig)

    # trajectory + network plots for a few trips (projected metres)
    tids = MAP_TRIPS["train"] + MAP_TRIPS["student_test"]
    fig, axes = plt.subplots(1, len(tids), figsize=(5 * len(tids), 5))
    for a, tid in zip(axes, tids):
        part = "student_test" if tid in MAP_TRIPS["student_test"] else "train"
        trip = Trip((C.STUDENT_TEST_DIR if part == "student_test" else C.TRAIN_DIR) / tid)
        net, df = trip.net, trip.df
        m = 300
        sel = ((net.ax > df.x.min() - m) & (net.ax < df.x.max() + m) &
               (net.ay > df.y.min() - m) & (net.ay < df.y.max() + m))
        for i in np.where(sel)[0][:30000]:
            a.plot([net.ax[i], net.bx[i]], [net.ay[i], net.by[i]], color="0.75", lw=.5)
        route = io.load_route(trip.dir)
        for i in route:
            a.plot([net.ax[i], net.bx[i]], [net.ay[i], net.by[i]], color="tab:green", lw=2.5)
        a.scatter(df.x, df.y, s=2, color="tab:blue")
        a.set_title(f"{part} {tid}"); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "trajectories_with_network.png", dpi=120); plt.close(fig)


def label_and_candidate_plots():
    pts = C.PROCESSED / "train_points.parquet"
    if not pts.exists():
        return
    p = pd.read_parquet(pts)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(np.clip(p.ref_dist, 0, 100), bins=100)
    ax[0].axvline(C.LABEL_MAX_DIST_M, color="r", ls="--", label="label validity cut-off")
    ax[0].set(title="Distance GPS fix -> aligned reference arc (train)", xlabel="m (clipped 100)")
    ax[0].legend()
    grid = pd.read_csv(OUT / "candidate_miss_grid_train.csv")
    for r, g in grid.groupby("radius"):
        ax[1].plot(g.k, g.miss_rate * 100, marker="o", label=f"radius {r} m")
    ax[1].set(title="Candidate miss rate (train)", xlabel="k candidates", ylabel="miss %", yscale="log")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "labels_and_candidates.png", dpi=120); plt.close(fig)


def maps():
    for part, tids in MAP_TRIPS.items():
        pdir = C.STUDENT_TEST_DIR if part == "student_test" else C.TRAIN_DIR
        for tid in tids:
            trip = Trip(pdir / tid)
            m = trip_map(trip, reference_arcs=io.load_route(trip.dir),
                         title=f"EDA - {part} trip {tid}: GPS vs reference route")
            m.save(str(OUT / f"map_{part}_{tid}.html"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv, gaps = raw_inventory()
    inv.to_csv(OUT / "inventory.csv", index=False)
    summary = inv.groupby("partition").agg(
        trips=("trip", "size"), fixes_total=("n_gps", "sum"), fixes_median=("n_gps", "median"),
        nan_total=("n_nan", "sum"), nonmonotonic_t=("n_nonmonotonic_t", "sum"),
        dt_eq_1_mean=("frac_dt_eq_1", "mean"), dt_max=("dt_max", "max"),
        gaps_gt_10s=("n_gaps_gt_10s", "sum"), jumps=("n_jumps_gt_70ms", "sum"),
        length_km_median=("length_km", "median"), arcs_median=("n_arcs", "median"),
        isolated_nodes=("n_isolated_nodes", "sum"), weak_components_median=("n_weak_components", "median"),
        largest_scc_frac_min=("largest_scc_frac", "min"),
        route_cont_min=("route_continuity", "min"), trips_with_repeats=("route_repeated_arcs", lambda s: int((s > 0).sum())))
    summary.T.to_csv(OUT / "summary_by_partition.csv")
    print(summary.T.to_string())
    plots(inv, gaps)
    label_and_candidate_plots()
    maps()
    print("EDA written to", OUT)


if __name__ == "__main__":
    main()
