"""Collect experiment outputs into Markdown tables for the technical report.
Usage: python scripts/make_report_tables.py  -> report/results_tables.md"""
import json

import _common  # noqa: F401
import pandas as pd

from pathsnap import config as C

OUT = C.ROOT / "report" / "results_tables.md"


def md(df, floatfmt=4):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: f"{v:.{floatfmt}f}")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    parts = ["# PathSnap: results tables (auto-generated from results/)\n"]
    eda = C.RESULTS / "eda"
    s = pd.read_csv(eda / "summary_by_partition.csv", index_col=0)
    parts += ["## Dataset summary (raw files, train + student_test)\n", md(s.reset_index().rename(columns={"index": "statistic"}), 3), ""]
    g = pd.read_csv(eda / "candidate_miss_grid_train.csv")
    parts += ["## Candidate miss rate on TRAIN (rows = radius m, cols = k)\n",
              md(g.pivot(index="radius", columns="k", values="miss_rate").reset_index()), ""]
    cp = json.load(open(C.PROCESSED / "candidate_params.json"))
    parts += [f"Chosen: radius = {cp['cand_radius_m']} m, k = {cp['cand_k']} (train miss rate {cp['train_miss_rate']:.4f}).\n"]

    mc = pd.read_csv(C.RESULTS / "models" / "model_comparison.csv")
    t = pd.DataFrame({
        "Model": mc.model, "Algorithm family": mc.family, "Key configuration": mc.config,
        "CV macro F1 (mean±std)": [f"{a:.4f} ± {b:.4f}" for a, b in zip(mc.cv_macro_f1_mean, mc.cv_macro_f1_std)],
        "CV top-1": mc.cv_top1_mean,
        "Student-test top-1": mc.test_top1_acc,
        "Route continuity (clf-only)": mc.test_route_continuity,
        "Student-test macro F1": mc.test_macro_f1,
        "Student-test log loss": mc.test_log_loss,
        "Inference ms / 1000 fixes": mc.inference_ms_per_1000_fixes,
    })
    parts += ["## 3.1 Overall model comparison\n", md(t), ""]
    t2 = mc[["model", "train_rows_used", "train_macro_f1", "cv_macro_f1_mean", "test_macro_f1", "train_log_loss",
             "cv_log_loss_mean", "test_log_loss", "test_precision_pos", "test_recall_pos", "test_route_f1",
             "fit_time_s_full"]]
    parts += ["## Over/under-fitting evidence and extra metrics\n", md(t2), ""]

    op = C.RESULTS / "optuna"
    if (op / "tuned_vs_untuned.csv").exists():
        tv = pd.read_csv(op / "tuned_vs_untuned.csv", index_col=0)
        bp = json.load(open(op / "best_params.json"))
        parts += ["## 4. Bayesian optimisation (LightGBM, Optuna TPE)\n",
                  f"Objective: {bp['objective']}. Trials: {bp['n_trials']}. Search time: {bp['search_seconds']:.0f} s "
                  f"on a {bp['search_fixes']:,}-fix sample. Best search score: {bp['best_cv_top1_search']:.4f}.\n",
                  "Best parameters: `" + json.dumps(bp["best_params"]) + "`\n",
                  md(tv.drop(columns=["params"]).reset_index().rename(columns={"index": "model"})), ""]
    hm = C.RESULTS / "hmm"
    if (hm / "student_test_clf_vs_hmm.csv").exists():
        hg = pd.read_csv(hm / "hmm_grid_train_oof.csv")
        parts += ["## 5. HMM decoder tuning grid (train trips, out-of-fold probabilities)\n", md(hg), ""]
        h = pd.read_csv(hm / "student_test_clf_vs_hmm.csv", dtype={"trip": str})
        cols = ["trip", "n_fixes", "clf_accuracy", "hmm_accuracy", "clf_fix_continuity", "hmm_fix_continuity",
                "route_breaks", "clf_route_f1", "hmm_route_f1", "hmm_route_precision", "hmm_route_recall", "hmm_seconds"]
        parts += ["## 5. Classifier-only vs classifier + HMM (student test)\n", md(h[cols]), ""]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
