"""Train and compare candidate classifiers from several algorithm families.

Protocol
- 5-fold GroupKFold on the 80 TRAIN trips (group = trip, so no trip leaks across folds).
- Each model is refit on all train trips and evaluated on the 10 STUDENT-TEST trips.
- Rows = (GPS fix, candidate arc); target = 1 if the arc is the fix's reference arc.
- Fixes with no reliable label (far from the reference route) are excluded.
- Slow learners (KNN, SVM) use smaller training samples; all others use the same
  sample size. Samples are drawn by fix so candidate sets stay intact.

Outputs -> results/models/
Usage: python scripts/train_models.py [--only lr,lgbm,...]
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
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss

from pathsnap import config as C, io
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics, top1_accuracy, top1_table, continuity, route_prf
from pathsnap.trip import Trip

OUT = C.RESULTS / "models"
N_FOLDS = 5
DEFAULT_CAP = 80_000   # fixes per training set


class MLPWithHistory:
    """sklearn MLP trained epoch by epoch so we can record train/val log loss."""

    def __init__(self, epochs=25, **kw):
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(**kw)
        self.history = {"epoch": [], "train_log_loss": [], "val_log_loss": []}

    def fit(self, X, y, X_val=None, y_val=None):
        Xs = self.scaler.fit_transform(X)
        Xv = self.scaler.transform(X_val) if X_val is not None else None
        rng = np.random.default_rng(C.SEED)
        sub = rng.choice(len(Xs), min(len(Xs), 100_000), replace=False)
        for e in range(self.epochs):
            perm = rng.permutation(len(Xs))
            for s in range(0, len(perm), 20_000):
                b = perm[s:s + 20_000]
                self.mlp.partial_fit(Xs[b], y[b], classes=[0, 1])
            self.history["epoch"].append(e + 1)
            self.history["train_log_loss"].append(log_loss(y[sub], self.mlp.predict_proba(Xs[sub])[:, 1], labels=[0, 1]))
            self.history["val_log_loss"].append(
                log_loss(y_val, self.mlp.predict_proba(Xv)[:, 1], labels=[0, 1]) if Xv is not None else np.nan)
        return self

    def predict_proba(self, X):
        return self.mlp.predict_proba(self.scaler.transform(X))


class LGBMWithHistory:
    def __init__(self, **kw):
        self.model = lgb.LGBMClassifier(**kw)
        self.history = None

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None:
            self.model.fit(X, y, eval_set=[(X, y), (X_val, y_val)], eval_names=["train", "valid"],
                           eval_metric="binary_logloss")
            r = self.model.evals_result_
            self.history = {"iteration": list(range(1, len(r["train"]["binary_logloss"]) + 1)),
                            "train_log_loss": r["train"]["binary_logloss"],
                            "val_log_loss": r["valid"]["binary_logloss"]}
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def model_specs():
    return {
        "logreg": dict(family="Linear", cap=DEFAULT_CAP,
                       config="StandardScaler + L2 LogisticRegression C=1.0, lbfgs",
                       build=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
        "gnb": dict(family="Probabilistic", cap=DEFAULT_CAP, config="GaussianNB var_smoothing=1e-9",
                    build=lambda: GaussianNB()),
        "knn": dict(family="Instance-based", cap=20_000,
                    config="StandardScaler + KNN k=50, distance-weighted, euclidean",
                    build=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1))),
        "linsvm": dict(family="Kernel (linear SVM)", cap=40_000,
                       config="StandardScaler + LinearSVC C=1.0, sigmoid-calibrated (3-fold)",
                       build=lambda: make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(C=1.0), cv=3, method="sigmoid"))),
        "rf": dict(family="Ensemble (bagging)", cap=DEFAULT_CAP,
                   config="RandomForest 200 trees, max_depth=20, min_samples_leaf=5",
                   build=lambda: RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                                        n_jobs=-1, random_state=C.SEED)),
        "lgbm": dict(family="Boosting", cap=DEFAULT_CAP,
                     config="LightGBM 400 trees, lr=0.05, num_leaves=63 (defaults otherwise)",
                     build=lambda: LGBMWithHistory(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                                   random_state=C.SEED, verbose=-1, n_jobs=-1)),
        "mlp": dict(family="Neural network", cap=DEFAULT_CAP,
                    config="StandardScaler + MLP (64,32) ReLU, Adam lr=1e-3, alpha=1e-4, 25 epochs",
                    build=lambda: MLPWithHistory(epochs=25, hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                 learning_rate_init=1e-3, random_state=C.SEED)),
    }


def sample_fixes(df, cap, seed=C.SEED):
    keys = df[["trip", "pid"]].drop_duplicates()
    if len(keys) > cap:
        keys = keys.sample(cap, random_state=seed)
    return df.merge(keys, on=["trip", "pid"])


def fit_model(spec, train_df, val_df=None):
    tr = sample_fixes(train_df, spec["cap"])
    model = spec["build"]()
    t0 = time.time()
    if isinstance(model, (MLPWithHistory, LGBMWithHistory)):
        Xv = val_df[FEATURES].to_numpy() if val_df is not None else None
        yv = val_df["y"].to_numpy() if val_df is not None else None
        model.fit(tr[FEATURES].to_numpy(), tr["y"].to_numpy(), Xv, yv)
    else:
        model.fit(tr[FEATURES].to_numpy(), tr["y"].to_numpy())
    return model, time.time() - t0, len(tr)


def predict(model, df):
    t0 = time.time()
    p = model.predict_proba(df[FEATURES].to_numpy())[:, 1]
    return p, time.time() - t0


def route_level(cands_with_prob, nets, routes):
    """Classifier-only route: per-fix top-1 arc in time order."""
    top = top1_table(cands_with_prob).sort_values(["trip", "pid"])
    conts, f1s = [], []
    for tid, g in top.groupby("trip"):
        seq = g["pred_arc"].to_numpy()
        conts.append(continuity(seq, nets[tid]))
        f1s.append(route_prf(seq, routes[tid], nets[tid])["route_f1"])
    return float(np.mean(conts)), float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = model_specs()
    if args.only:
        specs = {k: v for k, v in specs.items() if k in args.only.split(",")}

    train_all = pd.read_parquet(C.PROCESSED / "train_cands.parquet")
    test_all = pd.read_parquet(C.PROCESSED / "student_test_cands.parquet")
    train_pts = pd.read_parquet(C.PROCESSED / "train_points.parquet")
    test_pts = pd.read_parquet(C.PROCESSED / "student_test_points.parquet")
    train = train_all[train_all["label_valid"]].reset_index(drop=True)
    test = test_all[test_all["label_valid"]].reset_index(drop=True)
    print(f"train rows {len(train):,} ({train[['trip','pid']].drop_duplicates().shape[0]:,} fixes), "
          f"test rows {len(test):,}, positive rate {train['y'].mean():.3f}")

    test_nets, test_routes = {}, {}
    for tid in io.list_trips(C.STUDENT_TEST_DIR):
        t = Trip(C.STUDENT_TEST_DIR / tid)
        test_nets[tid] = t.net
        test_routes[tid] = io.load_route(t.dir)

    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(train, groups=train["trip"]))
    results, fold_rows, histories, confusions = [], [], {}, {}
    for name, spec in specs.items():
        print(f"\n=== {name} ===", flush=True)
        fold_f1, fold_ll, fold_top1, fit_times = [], [], [], []
        for k, (tri, vai) in enumerate(folds):
            tr, va = train.iloc[tri], train.iloc[vai]
            model, ft, n_fit = fit_model(spec, tr, va if k == 0 else None)
            if k == 0 and getattr(model, "history", None):
                histories[name] = model.history
            p, _ = predict(model, va)
            cm = candidate_metrics(va["y"].to_numpy(), p)
            vv = va.assign(prob=p)
            t1, _ = top1_accuracy(vv, train_pts[train_pts["trip"].isin(va["trip"].unique())])
            fold_f1.append(cm["macro_f1"]); fold_ll.append(cm["log_loss"]); fold_top1.append(t1)
            fit_times.append(ft)
            fold_rows.append({"model": name, "fold": k, "macro_f1": cm["macro_f1"],
                              "log_loss": cm["log_loss"], "top1_acc": t1, "fit_s": ft})
            print(f"  fold {k}: macroF1={cm['macro_f1']:.4f} logloss={cm['log_loss']:.4f} top1={t1:.4f} fit={ft:.1f}s", flush=True)
        # refit on all train, evaluate on student test
        model, ft, n_fit = fit_model(spec, train)
        p, inf_s = predict(model, test_all)  # all fixes (incl. unlabelled) for route metrics
        test_all_p = test_all.assign(prob=p)
        lab = test_all_p[test_all_p["label_valid"]]
        cm = candidate_metrics(lab["y"].to_numpy(), lab["prob"].to_numpy())
        t1, miss = top1_accuracy(test_all_p, test_pts)
        cont, rf1 = route_level(test_all_p, test_nets, test_routes)
        n_fixes = test_all[["trip", "pid"]].drop_duplicates().shape[0]
        # train-set fit (overfitting check)
        tr_s = sample_fixes(train, 20_000, seed=1)
        ptr, _ = predict(model, tr_s)
        train_cm = candidate_metrics(tr_s["y"].to_numpy(), ptr)
        confusions[name] = cm["confusion"]
        row = {"model": name, "family": spec["family"], "config": spec["config"],
               "train_rows_used": n_fit,
               "cv_macro_f1_mean": np.mean(fold_f1), "cv_macro_f1_std": np.std(fold_f1),
               "cv_log_loss_mean": np.mean(fold_ll), "cv_top1_mean": np.mean(fold_top1),
               "cv_top1_std": np.std(fold_top1),
               "train_macro_f1": train_cm["macro_f1"], "train_log_loss": train_cm["log_loss"],
               "test_macro_f1": cm["macro_f1"], "test_precision_pos": cm["precision_pos"],
               "test_recall_pos": cm["recall_pos"], "test_f1_pos": cm["f1_pos"],
               "test_accuracy": cm["accuracy"], "test_log_loss": cm["log_loss"],
               "test_top1_acc": t1, "test_candidate_miss_rate": miss,
               "test_route_continuity": cont, "test_route_f1": rf1,
               "fit_time_s_full": ft, "inference_ms_per_1000_fixes": 1000 * inf_s / n_fixes * 1000}
        results.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        if name not in ("rf", "knn"):  # keep repo small; RF/KNN are large
            joblib.dump(model, OUT / f"model_{name}.joblib")
        pd.DataFrame(results).to_csv(OUT / f"model_comparison{'_partial' if args.only else ''}.csv", index=False)

    res = pd.DataFrame(results)
    suffix = "_partial" if args.only else ""
    res.to_csv(OUT / f"model_comparison{suffix}.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / f"cv_folds{suffix}.csv", index=False)
    json.dump(histories, open(OUT / f"training_histories{suffix}.json", "w"))
    json.dump(confusions, open(OUT / f"confusion_matrices{suffix}.json", "w"))
    if not args.only:
        make_plots(res, pd.DataFrame(fold_rows), histories, confusions)
    print(res[["model", "cv_macro_f1_mean", "cv_top1_mean", "test_macro_f1", "test_top1_acc",
               "test_log_loss", "test_route_continuity", "inference_ms_per_1000_fixes"]].round(4).to_string())


def make_plots(res, folds, histories, confusions):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].bar(res.model, res.cv_macro_f1_mean, yerr=res.cv_macro_f1_std, capsize=4, color="tab:blue")
    ax[0].set(title="5-fold GroupKFold CV macro F1 (train)", ylim=(0.5, 1))
    x = np.arange(len(res)); w = .38
    ax[1].bar(x - w / 2, res.test_top1_acc, w, label="Top-1 accuracy")
    ax[1].bar(x + w / 2, res.test_macro_f1, w, label="Macro F1")
    ax[1].set_xticks(x, res.model); ax[1].set(title="Student-test performance", ylim=(0.5, 1)); ax[1].legend()
    ax[2].bar(res.model, res.test_log_loss, color="tab:red")
    ax[2].set(title="Student-test log loss (lower is better)")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, g in folds.groupby("model"):
        ax.plot(g.fold, g.top1_acc, marker="o", label=m)
    ax.set(title="Per-fold top-1 accuracy (stability across folds)", xlabel="fold", ylabel="top-1")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "cv_fold_stability.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(res.model, res.inference_ms_per_1000_fixes); ax[0].set(title="Inference ms / 1000 fixes", yscale="log")
    ax[1].bar(res.model, res.test_route_continuity); ax[1].set(title="Classifier-only route continuity", ylim=(0.5, 1))
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(OUT / "efficiency_continuity.png", dpi=120); plt.close(fig)

    n = len(confusions)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    for a, (m, cm) in zip(np.atleast_1d(axes), confusions.items()):
        cm = np.array(cm); a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8)
        a.set(title=m, xticks=[0, 1], yticks=[0, 1], xlabel="pred", ylabel="true")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=120); plt.close(fig)

    for m, h in histories.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        xk = "epoch" if "epoch" in h else "iteration"
        ax.plot(h[xk], h["train_log_loss"], label="train"); ax.plot(h[xk], h["val_log_loss"], label="validation (CV fold 0)")
        ax.set(title=f"{m}: training history", xlabel=xk, ylabel="log loss"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / f"history_{m}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
