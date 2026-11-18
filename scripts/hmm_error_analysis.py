"""Error analysis of the deployed pipeline on the STUDENT-TEST trips.

Every wrong labelled fix is put in exactly one category (tested in this order):
  reverse_twin   predicted arc is the opposite-direction twin of the reference arc
  adjacent_arc   shares a node with the reference arc (and is not the twin)
  other_on_route another arc of the reference route, not adjacent
  unrelated_arc  an arc that is not on the reference route
  no_candidate   no predicted arc (-1)
Also reports accuracy by GPS speed (speed from the +-2-fix window in track.py).

Outputs -> results/hmm_analysis/
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathsnap import config as C, io
from pathsnap.inference import PathSnapModel
from pathsnap.labels import align_track_to_route
from pathsnap.trip import Trip

OUT = C.RESULTS / "hmm_analysis"
OUT.mkdir(parents=True, exist_ok=True)
CATS = ["reverse_twin", "adjacent_arc", "other_on_route", "unrelated_arc", "no_candidate"]
SPEED_BINS = [0, 2, 5, 10, 20, np.inf]
SPEED_LABELS = ["< 2", "2-5", "5-10", "10-20", "> 20"]


def categorise(pred, ref, route_set, net):
    if pred < 0:
        return "no_candidate"
    if net.twin[ref] == pred:
        return "reverse_twin"
    if {net.a_from[pred], net.a_to[pred]} & {net.a_from[ref], net.a_to[ref]}:
        return "adjacent_arc"
    if pred in route_set:
        return "other_on_route"
    return "unrelated_arc"


model = PathSnapModel(C.ARTIFACTS)
rows = []
for tid in io.list_trips(C.STUDENT_TEST_DIR):
    trip = Trip(C.STUDENT_TEST_DIR / tid)
    pred = model.predict_trip(trip)
    route = io.load_route(trip.dir)
    lab = align_track_to_route(trip.df, trip.net, route)
    route_set = set(route.tolist())
    valid = lab["label_valid"]
    for i in np.flatnonzero(valid):
        ref = int(lab["ref_arc"][i])
        rec = {"trip": tid, "fix": int(i), "speed": float(trip.df["speed"].iat[i]), "ref_arc": ref}
        for stage, key in (("clf", "classifier_pred"), ("hmm", "hmm_pred")):
            p = int(pred[key][i])
            rec[f"{stage}_correct"] = p == ref
            rec[f"{stage}_cat"] = "correct" if p == ref else categorise(p, ref, route_set, trip.net)
        rows.append(rec)
    print(tid, "done")

df = pd.DataFrame(rows)
df.to_csv(OUT / "per_fix_errors.csv", index=False)

# error categories (pooled)
n = len(df)
cat = pd.DataFrame({
    "classifier_only": df["clf_cat"].value_counts().reindex(CATS, fill_value=0),
    "classifier_hmm": df["hmm_cat"].value_counts().reindex(CATS, fill_value=0),
})
cat["classifier_only_share"] = cat["classifier_only"] / cat["classifier_only"].sum()
cat["classifier_hmm_share"] = cat["classifier_hmm"] / cat["classifier_hmm"].sum()
cat.loc["all_errors"] = [cat["classifier_only"].sum(), cat["classifier_hmm"].sum(), 1.0, 1.0]
cat.to_csv(OUT / "error_categories.csv")

# per trip
by_trip = (df[df["hmm_cat"] != "correct"].groupby(["trip", "hmm_cat"]).size()
           .unstack(fill_value=0).reindex(columns=CATS, fill_value=0))
by_trip["hmm_accuracy"] = df.groupby("trip")["hmm_correct"].mean()
by_trip.to_csv(OUT / "error_categories_by_trip.csv")

# accuracy by speed
df["speed_bin"] = pd.cut(df["speed"], SPEED_BINS, labels=SPEED_LABELS, right=False)
g = df.groupby("speed_bin", observed=False)
spd = pd.DataFrame({
    "labelled_fixes": g.size(),
    "classifier_accuracy": g["clf_correct"].mean(),
    "hmm_accuracy": g["hmm_correct"].mean(),
    "hmm_errors": g["hmm_correct"].apply(lambda s: int((~s).sum())),
    "hmm_reverse_twin_errors": g["hmm_cat"].apply(lambda s: int((s == "reverse_twin").sum())),
})
spd.to_csv(OUT / "accuracy_by_speed.csv")

print(f"\nlabelled fixes: {n}")
print(cat.round(4).to_string())
print(spd.round(4).to_string())
print(by_trip.to_string())

# plots
fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(CATS))
ax.bar(x - 0.2, cat.loc[CATS, "classifier_only"], 0.4, label="classifier only")
ax.bar(x + 0.2, cat.loc[CATS, "classifier_hmm"], 0.4, label="classifier + HMM")
ax.set_xticks(x, [c.replace("_", " ") for c in CATS])
ax.set(ylabel="wrong labelled fixes", title="Error categories, student-test trips")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "error_categories.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(SPEED_LABELS))
ax.bar(x - 0.2, spd["classifier_accuracy"], 0.4, label="classifier only")
ax.bar(x + 0.2, spd["hmm_accuracy"], 0.4, label="classifier + HMM")
ax.set_xticks(x, SPEED_LABELS); ax.set_ylim(0.5, 1.0)
ax.set(xlabel="GPS speed (m/s)", ylabel="per-fix accuracy", title="Accuracy by speed, student-test trips")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "accuracy_by_speed.png", dpi=120); plt.close(fig)
