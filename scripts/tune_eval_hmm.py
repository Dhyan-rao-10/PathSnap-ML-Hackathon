"""Tune the HMM decoder and compare classifier-only vs classifier+HMM.

1. Tuning (TRAIN only): out-of-fold probabilities of the tuned classifier on a set of
   train trips -> grid search over beta (transition scale) and the emission weight.
   Objective: per-fix accuracy (micro-average over labelled fixes).
2. Evaluation (STUDENT-TEST): tuned classifier + chosen decoder parameters.
3. Writes artifacts/config.json (everything the API needs) and Folium maps.

Usage: python scripts/tune_eval_hmm.py
"""
import json
import time

import _common  # noqa: F401
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config as C, io
from pathsnap.features import FEATURES, candidate_table
from pathsnap.hmm import viterbi, stitch_route, build_cand_dict
from pathsnap.labels import align_track_to_route
from pathsnap.metrics import continuity, route_prf
from pathsnap.trip import Trip
from pathsnap.visualize import trip_map

OUT = C.RESULTS / "hmm"
BETAS = [2, 5, 10, 20, 40]
WEIGHTS = [0.5, 1.0, 2.0, 3.0, 4.0]
N_TUNE_TRIPS = 25


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cand_params = json.load(open(C.PROCESSED / "candidate_params.json"))
    oof = pd.read_parquet(C.PROCESSED / "train_oof_tuned.parquet")
    pts = pd.read_parquet(C.PROCESSED / "train_points.parquet")
    summ = pd.read_csv(C.RESULTS / "eda" / "trip_processing_summary.csv", dtype={"trip": str})
    summ["trip"] = summ["trip"].str.zfill(8)
    sizes = pts.groupby("trip").size()
    eligible = summ[(summ.partition == "train") & (summ.label_valid_frac > 0.95)]["trip"]
    eligible = [t for t in eligible if sizes[t] <= 6000]
    rng = np.random.default_rng(C.SEED)
    tune_trips = sorted(rng.choice(eligible, size=min(N_TUNE_TRIPS, len(eligible)), replace=False))
    print("tuning trips:", tune_trips)

    # ---------------- grid search on train (OOF probabilities) ----------------
    trips = {}
    for tid in tune_trips:
        tr = Trip(C.TRAIN_DIR / tid)
        p = pts[pts.trip == tid].sort_values("pid")
        trips[tid] = (tr, p["ref_arc"].to_numpy(), p["label_valid"].to_numpy(),
                      build_cand_dict(oof[oof.trip == tid], max_states=C.HMM_MAX_STATES))
    rows = []
    clf_correct = clf_total = 0
    for tid, (tr, ref, valid, cd) in trips.items():
        c = oof[oof.trip == tid]
        top = c.loc[c.groupby("pid").prob.idxmax()]
        pred = np.full(len(tr.df), -1); pred[top.pid] = top.arc
        clf_correct += int((pred[valid] == ref[valid]).sum()); clf_total += int(valid.sum())
    print(f"classifier-only (OOF) accuracy on tuning trips: {clf_correct / clf_total:.4f}")
    for beta in BETAS:
        for w in WEIGHTS:
            correct = total = 0; secs = 0
            for tid, (tr, ref, valid, cd) in trips.items():
                t0 = time.time()
                pred, _ = viterbi(tr.df, cd, tr.net, beta=beta, w_emit=w)
                secs += time.time() - t0
                correct += int((pred[valid] == ref[valid]).sum()); total += int(valid.sum())
            rows.append({"beta": beta, "emission_weight": w, "accuracy": correct / total, "seconds": secs})
            print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "hmm_grid_train_oof.csv", index=False)
    best = grid.sort_values("accuracy", ascending=False).iloc[0]
    beta, w = float(best.beta), float(best.emission_weight)
    print(f"chosen beta={beta}, emission_weight={w}, train-OOF accuracy={best.accuracy:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for wv, g in grid.groupby("emission_weight"):
        ax.plot(g.beta, g.accuracy, marker="o", label=f"emission weight {wv}")
    ax.axhline(clf_correct / clf_total, color="k", ls="--", label="classifier only")
    ax.set(xscale="log", xlabel="beta (m)", ylabel="per-fix accuracy",
           title="HMM decoder tuning (train trips, out-of-fold probabilities)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "hmm_tuning.png", dpi=120); plt.close(fig)

    # ---------------- write deployment config ----------------
    cfg = {
        "model_file": "classifier_lgbm_tuned.joblib",
        "features": FEATURES,
        "cand_radius_m": cand_params["cand_radius_m"], "cand_k": cand_params["cand_k"],
        "hmm_beta": beta, "hmm_emission_weight": w, "hmm_max_states": C.HMM_MAX_STATES,
        "hmm_unreachable_logp": C.HMM_UNREACHABLE_LOGP,
        "gap_break_s": C.GAP_BREAK_S, "max_speed_ms": C.MAX_SPEED_MS,
        "label_max_dist_m": C.LABEL_MAX_DIST_M,
    }
    json.dump(cfg, open(C.ARTIFACTS / "config.json", "w"), indent=2)

    # ---------------- student-test evaluation ----------------
    model = joblib.load(C.ARTIFACTS / cfg["model_file"])
    rows = []
    for tid in io.list_trips(C.STUDENT_TEST_DIR):
        tr = Trip(C.STUDENT_TEST_DIR / tid)
        route = io.load_route(tr.dir)
        lab = align_track_to_route(tr.df, tr.net, route)
        v = lab["label_valid"]; ref = lab["ref_arc"]
        t0 = time.time()
        c = candidate_table(tr.df, tr.net, cfg["cand_radius_m"], cfg["cand_k"])
        c["prob"] = model.predict_proba(c[FEATURES])[:, 1]
        t_clf = time.time() - t0
        top = c.loc[c.groupby("pid").prob.idxmax()]
        clf = np.full(len(tr.df), -1); clf[top.pid] = top.arc
        t0 = time.time()
        hmm, info = viterbi(tr.df, build_cand_dict(c, C.HMM_MAX_STATES), tr.net, beta=beta, w_emit=w)
        stitched, breaks = stitch_route(hmm, tr.df, tr.net)
        t_hmm = time.time() - t0
        clf_r = route_prf(clf, route, tr.net); hmm_r = route_prf(stitched, route, tr.net)
        stitched_arr = np.array(stitched)
        rows.append({
            "trip": tid, "n_fixes": len(tr.df), "n_labelled": int(v.sum()),
            "clf_accuracy": float((clf[v] == ref[v]).mean()), "hmm_accuracy": float((hmm[v] == ref[v]).mean()),
            "clf_fix_continuity": continuity(clf, tr.net), "hmm_fix_continuity": continuity(hmm, tr.net),
            "hmm_stitched_route_continuity": continuity(stitched_arr, tr.net),
            "route_breaks": breaks,
            "clf_route_precision": clf_r["route_precision"], "clf_route_recall": clf_r["route_recall"], "clf_route_f1": clf_r["route_f1"],
            "hmm_route_precision": hmm_r["route_precision"], "hmm_route_recall": hmm_r["route_recall"], "hmm_route_f1": hmm_r["route_f1"],
            "clf_seconds": t_clf, "hmm_seconds": t_hmm,
        })
        print(rows[-1], flush=True)
        if tid in ("00000009", "00000076", "00000056"):
            m = trip_map(tr, reference_arcs=route, predicted_arcs=stitched, classifier_arcs=[a for a in clf if a >= 0],
                         title=f"Student-test trip {tid}: classifier {rows[-1]['clf_accuracy']:.1%} -> HMM {rows[-1]['hmm_accuracy']:.1%}")
            m.save(str(OUT / f"map_student_test_{tid}.html"))
    res = pd.DataFrame(rows)
    tot = {"trip": "ALL (fix-weighted)"}
    for col in ["clf_accuracy", "hmm_accuracy"]:
        tot[col] = float((res[col] * res.n_labelled).sum() / res.n_labelled.sum())
    for col in res.columns:
        if col not in tot and col != "trip":
            tot[col] = float(res[col].mean()) if col not in ("n_fixes", "n_labelled", "route_breaks") else float(res[col].sum())
    res = pd.concat([res, pd.DataFrame([tot])], ignore_index=True)
    res.to_csv(OUT / "student_test_clf_vs_hmm.csv", index=False)
    print(res.round(4).to_string())

    r = res[res.trip != "ALL (fix-weighted)"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    x = np.arange(len(r)); wd = .38
    ax[0].bar(x - wd / 2, r.clf_accuracy, wd, label="classifier only"); ax[0].bar(x + wd / 2, r.hmm_accuracy, wd, label="classifier + HMM")
    ax[0].set_xticks(x, r.trip, rotation=45); ax[0].set(ylim=(0.6, 1), title="Per-fix accuracy (student test)"); ax[0].legend()
    ax[1].bar(x - wd / 2, r.clf_route_f1, wd, label="classifier only"); ax[1].bar(x + wd / 2, r.hmm_route_f1, wd, label="classifier + HMM")
    ax[1].set_xticks(x, r.trip, rotation=45); ax[1].set(ylim=(0.6, 1), title="Route F1 (length-weighted, student test)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "clf_vs_hmm_student_test.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
