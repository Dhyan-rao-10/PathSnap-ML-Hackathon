"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model (LightGBM).

- Objective: mean observation-level top-1 accuracy over 3-fold GroupKFold
  (groups = trips) on the 80 TRAIN trips. Top-1 is our primary metric because the
  per-fix arc choice is what the route reconstruction consumes.
- Search uses a 40k-fix sample per run to keep each trial ~20 s.
- Afterwards: untuned vs tuned compared with identical 5-fold CV + student-test
  evaluation; the tuned model is refit on all train fixes and exported to artifacts/;
  out-of-fold probabilities for the train trips are saved for HMM tuning.

Usage: python scripts/tune_optuna.py [--trials 30] [--timeout 1200]
"""
import argparse
import json
import time

import _common  # noqa: F401
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config as C
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics, top1_accuracy

OUT = C.RESULTS / "optuna"
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63)
SEARCH_FIXES = 40_000


def sample_fixes(df, cap, seed=C.SEED):
    keys = df[["trip", "pid"]].drop_duplicates()
    if len(keys) > cap:
        keys = keys.sample(cap, random_state=seed)
    return df.merge(keys, on=["trip", "pid"])


def make(params):
    return lgb.LGBMClassifier(random_state=C.SEED, verbose=-1, n_jobs=-1, **params)


def cv_eval(params, df, points, n_folds, cap=None, return_oof=False):
    # score only the fixes present in df (df may be a sample of fixes)
    points = points.merge(df[["trip", "pid"]].drop_duplicates(), on=["trip", "pid"])
    gkf = GroupKFold(n_splits=n_folds)
    f1s, lls, t1s = [], [], []
    oof = np.zeros(len(df))
    for tri, vai in gkf.split(df, groups=df["trip"]):
        tr = df.iloc[tri]
        if cap:
            tr = sample_fixes(tr, cap)
        va = df.iloc[vai]
        m = make(params).fit(tr[FEATURES], tr["y"])
        p = m.predict_proba(va[FEATURES])[:, 1]
        oof[vai] = p
        cm = candidate_metrics(va["y"].to_numpy(), p)
        t1, _ = top1_accuracy(va.assign(prob=p), points[points["trip"].isin(va["trip"].unique())])
        f1s.append(cm["macro_f1"]); lls.append(cm["log_loss"]); t1s.append(t1)
    res = {"cv_macro_f1": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s)),
           "cv_log_loss": float(np.mean(lls)), "cv_top1": float(np.mean(t1s)), "cv_top1_std": float(np.std(t1s))}
    return (res, oof) if return_oof else res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train_all = pd.read_parquet(C.PROCESSED / "train_cands.parquet")
    test_all = pd.read_parquet(C.PROCESSED / "student_test_cands.parquet")
    train_pts = pd.read_parquet(C.PROCESSED / "train_points.parquet")
    test_pts = pd.read_parquet(C.PROCESSED / "student_test_points.parquet")
    train = train_all[train_all["label_valid"]].reset_index(drop=True)
    search = sample_fixes(train, SEARCH_FIXES)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        t0 = time.time()
        r = cv_eval(params, search, train_pts, n_folds=3)
        trial.set_user_attr("cv_macro_f1", r["cv_macro_f1"])
        trial.set_user_attr("cv_log_loss", r["cv_log_loss"])
        trial.set_user_attr("seconds", time.time() - t0)
        return r["cv_top1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED),
                                study_name="pathsnap_lgbm")
    study.enqueue_trial({**UNTUNED, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                         "reg_lambda": 1e-3, "reg_alpha": 1e-3})  # baseline as first trial
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)
    search_s = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = {**study.best_params, "subsample_freq": 1}
    print("best", study.best_value, best)

    # ---- untuned vs tuned, identical protocol ----
    rows = {}
    oof_tuned = None
    for name, params in [("untuned", UNTUNED), ("tuned", best)]:
        cvr, oof = cv_eval(params, train, train_pts, n_folds=5, return_oof=True)
        t0 = time.time(); m = make(params).fit(train[FEATURES], train["y"]); fit_s = time.time() - t0
        t0 = time.time(); p = m.predict_proba(test_all[FEATURES])[:, 1]; inf_s = time.time() - t0
        lab = test_all[test_all["label_valid"]]
        cm = candidate_metrics(lab["y"].to_numpy(), p[test_all["label_valid"].to_numpy()])
        t1, miss = top1_accuracy(test_all.assign(prob=p), test_pts)
        n_fixes = test_all[["trip", "pid"]].drop_duplicates().shape[0]
        rows[name] = {**cvr, "test_top1": t1, "test_macro_f1": cm["macro_f1"], "test_log_loss": cm["log_loss"],
                      "test_precision_pos": cm["precision_pos"], "test_recall_pos": cm["recall_pos"],
                      "fit_time_full_s": fit_s, "inference_ms_per_1000_fixes": 1e6 * inf_s / n_fixes,
                      "params": json.dumps(params)}
        print(name, {k: v for k, v in rows[name].items() if k != "params"}, flush=True)
        if name == "tuned":
            oof_tuned = oof
            joblib.dump(m, C.ARTIFACTS / "classifier_lgbm_tuned.joblib")
        else:
            joblib.dump(m, OUT / "classifier_lgbm_untuned.joblib")
    comp = pd.DataFrame(rows).T
    comp.to_csv(OUT / "tuned_vs_untuned.csv")
    json.dump({"best_params": best, "best_cv_top1_search": study.best_value, "n_trials": len(study.trials),
               "search_seconds": search_s, "search_fixes": SEARCH_FIXES, "objective": "3-fold GroupKFold mean top-1 accuracy"},
              open(OUT / "best_params.json", "w"), indent=2)

    oof_df = train[["trip", "pid", "arc", "y", "proj_t", "arc_len"]].assign(prob=oof_tuned)
    oof_df.to_parquet(C.PROCESSED / "train_oof_tuned.parquet")

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    vals = trials["value"].to_numpy()
    ax[0].plot(vals, "o", alpha=.6, label="trial"); ax[0].plot(np.maximum.accumulate(vals), "r-", label="best so far")
    ax[0].set(title="Optuna TPE optimisation history", xlabel="trial", ylabel="CV top-1 accuracy"); ax[0].legend()
    try:
        imp = optuna.importance.get_param_importances(study)
        ax[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1])
        ax[1].set(title="Hyperparameter importance (fANOVA)")
    except Exception as e:  # pragma: no cover
        ax[1].text(0.1, 0.5, str(e))
    fig.tight_layout(); fig.savefig(OUT / "optuna_history_importance.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1"]
    x = np.arange(len(metrics)); w = .38
    ax.bar(x - w / 2, comp.loc["untuned", metrics].astype(float), w, label="untuned")
    ax.bar(x + w / 2, comp.loc["tuned", metrics].astype(float), w, label="tuned")
    ax.set_xticks(x, metrics); ax.set_ylim(0.8, 1); ax.legend(); ax.set_title("LightGBM: untuned vs tuned")
    fig.tight_layout(); fig.savefig(OUT / "tuned_vs_untuned.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
