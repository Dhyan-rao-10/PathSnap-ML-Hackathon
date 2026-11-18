"""Evaluation metrics: candidate classification, per-fix top-1, route-level."""
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, log_loss, precision_score, recall_score,
                             accuracy_score, confusion_matrix)


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision_pos": precision_score(y, yhat, zero_division=0),
        "recall_pos": recall_score(y, yhat, zero_division=0),
        "f1_pos": f1_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def top1_table(cands, prob_col="prob"):
    """Per fix: arc with highest score. cands needs trip, pid, arc, y, prob."""
    idx = cands.groupby(["trip", "pid"])[prob_col].idxmax()
    top = cands.loc[idx, ["trip", "pid", "arc", "y", prob_col]].rename(columns={"arc": "pred_arc"})
    return top.reset_index(drop=True)


def top1_accuracy(cands, points, prob_col="prob"):
    """Accuracy over labelled fixes (fixes without candidates count as wrong).
    Also returns candidate miss rate."""
    lab = points[points["label_valid"]][["trip", "pid"]]
    top = top1_table(cands, prob_col)
    m = lab.merge(top[["trip", "pid", "y"]], how="left", on=["trip", "pid"])
    hit = cands[cands["y"] == 1][["trip", "pid"]].drop_duplicates()
    miss = lab.merge(hit, how="left", indicator=True)["_merge"].eq("left_only").mean()
    return float(m["y"].fillna(0).mean()), float(miss)


def continuity(arc_seq, net):
    """Fraction of consecutive fix->fix transitions that stay on the same arc or
    move to a directly connected arc."""
    a = np.asarray([x for x in arc_seq if x >= 0])
    if len(a) < 2:
        return 1.0
    ok = (a[1:] == a[:-1]) | (net.a_to[a[:-1]] == net.a_from[a[1:]])
    return float(ok.mean())


def route_prf(pred_arcs, ref_arcs, net):
    """Length-weighted precision / recall / F1 between predicted and reference
    arc sets (an arc counts once even if traversed twice)."""
    p = set(int(a) for a in pred_arcs if a >= 0)
    r = set(int(a) for a in ref_arcs)
    L = net.length
    inter = sum(L[a] for a in p & r)
    lp = sum(L[a] for a in p); lr = sum(L[a] for a in r)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def per_fix_accuracy(pred_arc, ref_arc, valid):
    pred_arc = np.asarray(pred_arc); ref_arc = np.asarray(ref_arc); valid = np.asarray(valid, bool)
    if valid.sum() == 0:
        return float("nan")
    return float((pred_arc[valid] == ref_arc[valid]).mean())


def summarise(rows):
    return pd.DataFrame(rows)
