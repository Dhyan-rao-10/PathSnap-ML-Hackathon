"""Server-side scoring of a prediction against the reference route.

Kept separate from inference: the prediction never sees the .route file; only
this module reads it, after prediction has finished.
"""
import numpy as np

from . import io
from .labels import align_track_to_route
from .metrics import per_fix_accuracy, route_prf


def evaluate_trip(trip, prediction):
    route = io.load_route(trip.dir)
    if route is None:
        return {"has_reference": False}
    lab = align_track_to_route(trip.df, trip.net, route)
    valid = lab["label_valid"]
    out = {
        "has_reference": True,
        "n_labelled_fixes": int(valid.sum()),
        "hmm_accuracy": per_fix_accuracy(prediction["hmm_pred"], lab["ref_arc"], valid),
        "classifier_accuracy": per_fix_accuracy(prediction["classifier_pred"], lab["ref_arc"], valid),
    }
    r = route_prf(prediction["route_arcs"], route, trip.net)
    out.update(r)
    clf_r = route_prf(prediction["classifier_pred"], route, trip.net)
    out["classifier_route_f1"] = clf_r["route_f1"]
    # only aggregate scores leave this module; reference arcs/geometry never do
    return out
