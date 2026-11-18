"""End-to-end prediction for one trip (GPS track + road network only)."""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import config as C
from .features import FEATURES, candidate_table
from .hmm import viterbi, stitch_route, build_cand_dict
from .metrics import continuity
from .trip import Trip


class PathSnapModel:
    def __init__(self, artifacts_dir=C.ARTIFACTS):
        artifacts_dir = Path(artifacts_dir)
        self.cfg = json.load(open(artifacts_dir / "config.json"))
        self.model = joblib.load(artifacts_dir / self.cfg["model_file"])
        assert self.cfg["features"] == FEATURES, "feature list mismatch between artifact and code"

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict_trip(self, trip: Trip):
        cfg = self.cfg
        timings = {}
        t0 = time.perf_counter()
        cands = candidate_table(trip.df, trip.net, radius=cfg["cand_radius_m"], k=cfg["cand_k"])
        timings["candidates_features_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cands["prob"] = self.predict_proba(cands[FEATURES]) if len(cands) else []
        timings["classifier_s"] = time.perf_counter() - t0

        # classifier-only: top-1 per fix
        clf_pred = np.full(len(trip.df), -1, dtype=np.int64)
        if len(cands):
            top = cands.loc[cands.groupby("pid")["prob"].idxmax()]
            clf_pred[top["pid"].to_numpy()] = top["arc"].to_numpy()

        t0 = time.perf_counter()
        cd = build_cand_dict(cands, max_states=cfg["hmm_max_states"])
        hmm_pred, info = viterbi(trip.df, cd, trip.net, beta=cfg["hmm_beta"],
                                 w_emit=cfg["hmm_emission_weight"])
        timings["viterbi_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        route, breaks = stitch_route(hmm_pred, trip.df, trip.net)
        timings["stitch_s"] = time.perf_counter() - t0
        return {
            "classifier_pred": clf_pred,
            "hmm_pred": hmm_pred,
            "route_arcs": route,
            "route_breaks": breaks,
            "n_chains": info["n_chains"],
            "n_fixes": len(trip.df),
            "n_fixes_without_candidates": int((hmm_pred < 0).sum()),
            "hmm_continuity": continuity(hmm_pred, trip.net),
            "classifier_continuity": continuity(clf_pred, trip.net),
            "timings": timings,
            "cands": cands,
        }
