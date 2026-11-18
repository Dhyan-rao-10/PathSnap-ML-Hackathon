"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR  folder with one sub-folder per evaluation trip
                          (<id>/<id>.track, .nodes, .arcs and, server-side only, .route)

Endpoints
  GET  /api/health          model + config summary
  GET  /api/trips           evaluation trips available on this server
  POST /api/predict         {"trip_id": "..."} -> predicted route + server-side scores

The reference .route is read only by pathsnap.evaluation AFTER prediction, and only
aggregate scores are returned -- reference arcs/geometry never leave the server.
"""
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from pathsnap import config as C, io  # noqa: E402
from pathsnap.evaluation import evaluate_trip  # noqa: E402
from pathsnap.inference import PathSnapModel  # noqa: E402
from pathsnap.trip import Trip  # noqa: E402
from pathsnap.visualize import _route_polyline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = PathSnapModel(C.ARTIFACTS)
EVAL_DIR = C.EVAL_DIR
log.info("Loaded model %s | config %s", MODEL.cfg["model_file"],
         {k: v for k, v in MODEL.cfg.items() if k not in ("features", "model_file")})
log.info("Evaluation trips folder: %s (%d trips)", EVAL_DIR, len(io.list_trips(EVAL_DIR)))


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code,
             1000 * (time.perf_counter() - t0))
    return response


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL.cfg["model_file"], "eval_dir": str(EVAL_DIR),
            "n_eval_trips": len(io.list_trips(EVAL_DIR)),
            "config": {k: v for k, v in MODEL.cfg.items() if k not in ("features",)}}


@app.get("/api/trips")
def trips():
    out = []
    for tid in io.list_trips(EVAL_DIR):
        with open(EVAL_DIR / tid / f"{tid}.track") as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": tid, "n_fixes": n})
    return {"trips": out}


@app.post("/api/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in io.list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip '{tid}'")
    log.info("[predict] trip %s: loading track + road network", tid)
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    trip = Trip(EVAL_DIR / tid)
    load_s = time.perf_counter() - t0
    log.info("[predict] trip %s: %d fixes, %d segments, %d nodes, %d arcs", tid, len(trip.df),
             trip.df["seg"].nunique(), trip.net.n_nodes, trip.net.n_arcs)

    pred = MODEL.predict_trip(trip)  # sees GPS + network only
    tm = pred["timings"]
    log.info("[predict] trip %s: candidates %.2fs | classifier %.2fs | viterbi %.2fs | route %d arcs, %d breaks",
             tid, tm["candidates_features_s"], tm["classifier_s"], tm["viterbi_s"],
             len(pred["route_arcs"]), pred["route_breaks"])
    total_pred_s = time.perf_counter() - t_all

    scores = evaluate_trip(trip, pred)  # server-side only
    if scores.get("has_reference"):
        log.info("[evaluate] trip %s: HMM accuracy %.2f%% | classifier-only %.2f%% | route F1 %.3f",
                 tid, 100 * scores["hmm_accuracy"], 100 * scores["classifier_accuracy"], scores["route_f1"])

    df, net = trip.df, trip.net
    seq = pred["hmm_pred"]
    return {
        "trip_id": tid,
        "n_fixes": int(len(df)),
        "n_segments": int(df["seg"].nunique()),
        "bounds": [[float(df.lat.min()), float(df.lon.min())], [float(df.lat.max()), float(df.lon.max())]],
        "gps": [[round(a, 7), round(b, 7)] for a, b in zip(df["lat"], df["lon"])],
        "predicted_route": [[[round(p[0], 7), round(p[1], 7)] for p in pl]
                            for pl in _route_polyline(net, pred["route_arcs"])],
        "stats": {
            "route_arcs": len(pred["route_arcs"]),
            "route_length_km": float(net.length[pred["route_arcs"]].sum() / 1000) if pred["route_arcs"] else 0.0,
            "route_breaks": pred["route_breaks"],
            "hmm_chains": pred["n_chains"],
            "fixes_without_candidates": pred["n_fixes_without_candidates"],
            "matched_fixes": int((seq >= 0).sum()),
            "hmm_fix_continuity": pred["hmm_continuity"],
            "classifier_fix_continuity": pred["classifier_continuity"],
        },
        "timings": {"load_s": load_s, **tm, "total_prediction_s": total_pred_s},
        "evaluation": {k: v for k, v in scores.items()},
    }
