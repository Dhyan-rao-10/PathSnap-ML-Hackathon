# PathSnap: map-matching and route reconstruction

PathSnap matches noisy 1 Hz GPS tracks to a directed road network and rebuilds the route that was driven:

```
raw trip ─► clean + segment track ─► project to UTM metres
        ─► candidate arcs per GPS fix (KD-tree, radius R, top-k)
        ─► 20 relative features per (fix, candidate)
        ─► LightGBM classifier  P(candidate is the true arc)
        ─► HMM + Viterbi on the road graph (emission = classifier, transition = route vs GPS distance)
        ─► stitched continuous route  ─► FastAPI (Laptop 1)  ─► React UI (Laptop 2)
```

UE24CS352A Machine Learning hackathon, PES University.
Team: Dhyan, Dhruva, Dilipan, Dhyeya.

## Repository layout

| Folder | Contents |
|---|---|
| `src/pathsnap/` | Core library: `io` (raw readers), `geo` (UTM projection, geometry), `track` (cleaning, gaps, kinematics), `network` (graph, KD-tree candidates, bounded Dijkstra), `labels` (fix→route-arc alignment), `features`, `hmm` (Viterbi + route stitching), `metrics`, `inference` (prediction only), `evaluation` (server-side scoring), `visualize` (Folium) |
| `scripts/` | Reproducible pipeline steps (see below) |
| `api/` | FastAPI backend (`api/main.py`) |
| `ui/` | React + Vite + react-leaflet evaluator UI |
| `artifacts/` | Deployed model (`classifier_lgbm_tuned.joblib`) and `config.json` (candidate, HMM and track parameters) |
| `results/` | EDA, model comparison, Optuna and HMM outputs: CSVs, plots and Folium HTML maps |
| `data/` | Raw / processed / eval data. Not committed; see `data/README.md` |
| `report/` | Technical report material |

## Setup

Python 3.12 and Node 18+ (Node is only needed to *build* the UI).

```bash
pip install -r requirements.txt
```

Put the provided data in `data/raw/train` and `data/raw/student_test` (see `data/README.md`).

## Reproducing the experiments

Run each step in order. Every step writes into `results/` and `data/processed/`.

```bash
python scripts/build_dataset.py      # labels, candidates, features, candidate radius/k selection
python scripts/eda.py                # EDA tables, plots, Folium maps   -> results/eda/
python scripts/train_models.py       # 7 models, GroupKFold CV + student test -> results/models/
python scripts/tune_optuna.py        # Optuna TPE for LightGBM, exports artifacts/ -> results/optuna/
python scripts/tune_eval_hmm.py      # HMM tuning (train OOF), classifier vs HMM, config.json -> results/hmm/
```

Random seed: 42 everywhere (`src/pathsnap/config.py`).

### Data-leakage rules we follow
- No trip is split across partitions. CV uses `GroupKFold` grouped by trip.
- The candidate radius/k and the HMM parameters are chosen on **train** only, and the HMM uses out-of-fold classifier probabilities. The student-test trips are used only for development comparison.
- Evaluator trips are never touched during development. At demo time the API reads their `.route` only inside `pathsnap/evaluation.py`, *after* prediction. Only aggregate scores are returned; reference arcs and geometry never leave the server.
- Nothing is hard-coded per trip. Every request runs the full pipeline, and the API console logs each step.

## Deployment (two laptops)

Both laptops must be on the same network (a phone hotspot works). Find Laptop 1's IP with `ipconfig`.

**Laptop 1 (model + API)**
```bash
pip install -r requirements.txt
# put the 10 evaluator trips in data/eval/<trip_id>/  (or set PATHSNAP_EVAL_DIR)
# rehearsal: python scripts/use_student_test_as_eval.py
run_api.bat          # = python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
If Windows Firewall prompts, allow Python on private networks. Health check: `http://<laptop1-ip>:8000/api/health`. API docs: `http://<laptop1-ip>:8000/docs`.

**Laptop 2 (UI)**
```bash
cd ui
npm install
npm run build        # once
cd ..
run_ui.bat           # serves ui/dist on http://localhost:5173 (Python only)
# or for development: cd ui && npm run dev
```
Open `http://localhost:5173`, enter `http://<laptop1-ip>:8000` in the API box, and click **Connect**. Then pick a trip and click **Run PathSnap**. The UI shows the GPS fixes, the predicted route and the server-side scores. The API console on Laptop 1 logs every request with per-step timings.

### API
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | none | model / config summary |
| GET | `/api/trips` | none | `[{trip_id, n_fixes}]` from the eval folder |
| POST | `/api/predict` | `{"trip_id": "00000009"}` | GPS fixes, predicted route polylines, stats, timings, evaluation scores |

## Results (student-test partition, 10 trips / 19,160 GPS fixes)

| Stage | Per-fix accuracy | Route F1 (length-weighted) | Fix-to-fix continuity |
|---|---|---|---|
| Best untuned classifier (LightGBM), top-1 | 0.898 | 0.951 | 0.947 |
| Tuned LightGBM (Optuna), top-1 | 0.904 | 0.947 | 0.946 |
| **Tuned LightGBM + HMM/Viterbi** | **0.925** | **0.980** | **0.977** |

Seven models were compared with 5-fold GroupKFold. LightGBM had the best CV top-1 (0.8995) and the lowest log loss (0.067). The full tables are in `report/results_tables.md`. Every trip runs in about 0.2–2.3 s through the API.

## Method summary
- **Labels.** `.route` lists the arcs driven in order but does not say which GPS fix belongs to which arc. A monotone dynamic-programming alignment assigns each fix to a route arc. Fixes more than 60 m from their aligned arc are treated as unlabelled; this mostly affects trips 33 and 54, whose reference route is offset from the GPS.
- **Candidates.** Arcs are densified every 10 m into a KD-tree. Radius 75 m and k=10 give a 0.97% candidate miss rate on train. If no arc is in range, the search widens to 300 m.
- **Features.** Distance plus its rank and margin within the candidate set, heading agreement, along-arc and cross-arc speed, arc geometry and junction degree, two-way flag, and context features (how well the arc explains the ±5 neighbouring fixes). None of them use absolute coordinates, so they generalise to unseen cities.
- **HMM.** Hidden states are each fix's top-8 candidate arcs. Emission = classifier probability normalised over the candidate set, raised to a weight. Transition = exp(−|route distance − GPS distance| / β), with directed shortest paths from a bounded Dijkstra. Unreachable transitions get a fixed penalty. Viterbi restarts at gaps > 120 s or jumps > 70 m/s. Consecutive arcs are stitched with shortest paths.
