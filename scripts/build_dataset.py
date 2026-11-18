"""Build candidate/feature tables for the train and student_test partitions.

Generates a *superset* of candidates (radius 100 m, k 16), measures candidate
miss rate for a grid of (radius, k) on TRAIN ONLY, picks the operating point, and
writes the filtered feature tables to data/processed/.

Usage: python scripts/build_dataset.py
"""
import json
import time

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pathsnap import config as C, io
from pathsnap.trip import Trip
from pathsnap.labels import align_track_to_route, route_continuity
from pathsnap.features import candidate_table

SUPER_R, SUPER_K = 100.0, 16


def process_partition(part, part_dir):
    rows, points, trips = [], [], []
    for tid in io.list_trips(part_dir):
        t0 = time.time()
        trip = Trip(part_dir / tid)
        route = io.load_route(trip.dir)
        lab = align_track_to_route(trip.df, trip.net, route)
        ct = candidate_table(trip.df, trip.net, radius=SUPER_R, k=SUPER_K)
        ct["y"] = (ct["arc"].to_numpy() == lab["ref_arc"][ct["pid"].to_numpy()]).astype(np.int8)
        ct["label_valid"] = lab["label_valid"][ct["pid"].to_numpy()]
        ct.insert(0, "trip", tid)
        rows.append(ct)
        pdf = trip.df[["t", "x", "y", "seg", "speed", "dt"]].copy()
        pdf.insert(0, "pid", np.arange(len(pdf)))
        pdf.insert(0, "trip", tid)
        for k in ["ref_arc", "ref_dist", "route_pos", "label_valid"]:
            pdf[k] = lab[k]
        points.append(pdf)
        trips.append({"trip": tid, "partition": part, **trip.clean_report,
                      "n_segments": int(trip.df["seg"].nunique()),
                      "n_nodes": trip.net.n_nodes, "n_arcs": trip.net.n_arcs,
                      "n_route_arcs": len(route),
                      "route_continuity": route_continuity(route, trip.net),
                      "label_valid_frac": float(lab["label_valid"].mean()),
                      "ref_dist_median": float(np.median(lab["ref_dist"])),
                      "secs": time.time() - t0})
        print(f"[{part}] {tid}: {len(trip.df)} fixes, {trip.net.n_arcs} arcs, {time.time()-t0:.1f}s", flush=True)
    return pd.concat(rows, ignore_index=True), pd.concat(points, ignore_index=True), pd.DataFrame(trips)


def miss_grid(cands, points):
    """Candidate miss rate over labelled fixes for each (radius, k)."""
    valid = points[points["label_valid"]][["trip", "pid"]]
    out = []
    for r in [20, 30, 50, 75, 100]:
        for k in [2, 4, 6, 8, 10, 12, 16]:
            s = cands[(cands["dist"] <= r) & (cands["dist_rank"] < k) & (cands["y"] == 1)]
            hit = valid.merge(s[["trip", "pid"]].drop_duplicates(), how="left", indicator=True)
            miss = float((hit["_merge"] == "left_only").mean())
            n_rows = int(((cands["dist"] <= r) & (cands["dist_rank"] < k)).sum())
            out.append({"radius": r, "k": k, "miss_rate": miss,
                        "rows_per_fix": n_rows / len(points)})
    return pd.DataFrame(out)


def filter_cands(c, r, k):
    # mirror inference: fixes with nothing inside r fall back to their nearest arcs
    dmin = c.groupby(["trip", "pid"])["dist"].transform("min")
    fb = dmin > r
    c = c[((c["dist"] <= r) | fb) & (c["dist_rank"] < k)].copy()
    c["fallback"] = fb[c.index]
    c["n_cand"] = c.groupby(["trip", "pid"])["arc"].transform("size")
    return c


def main():
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "eda").mkdir(parents=True, exist_ok=True)
    tr_c, tr_p, tr_t = process_partition("train", C.TRAIN_DIR)
    st_c, st_p, st_t = process_partition("student_test", C.STUDENT_TEST_DIR)
    pd.concat([tr_t, st_t]).to_csv(C.RESULTS / "eda" / "trip_processing_summary.csv", index=False)

    grid = miss_grid(tr_c, tr_p)
    grid.to_csv(C.RESULTS / "eda" / "candidate_miss_grid_train.csv", index=False)
    print(grid.pivot(index="radius", columns="k", values="miss_rate").round(4))
    # operating point: smallest rows_per_fix whose miss rate is within 0.2 pp of the best
    best = grid["miss_rate"].min()
    ok = grid[grid["miss_rate"] <= best + 0.002].sort_values("rows_per_fix").iloc[0]
    r, k = float(ok["radius"]), int(ok["k"])
    print(f"Chosen candidate radius={r} k={k} (train miss={ok['miss_rate']:.4f})")
    json.dump({"cand_radius_m": r, "cand_k": k, "train_miss_rate": float(ok["miss_rate"])},
              open(C.PROCESSED / "candidate_params.json", "w"), indent=2)

    filter_cands(tr_c, r, k).to_parquet(C.PROCESSED / "train_cands.parquet")
    filter_cands(st_c, r, k).to_parquet(C.PROCESSED / "student_test_cands.parquet")
    tr_p.to_parquet(C.PROCESSED / "train_points.parquet")
    st_p.to_parquet(C.PROCESSED / "student_test_points.parquet")


if __name__ == "__main__":
    main()
