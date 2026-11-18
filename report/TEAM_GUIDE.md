# Team guide: ownership, viva prep, demo checklist

## Who explains what in the viva
Everyone should be able to explain the whole pipeline at a high level. Each person owns one area in depth.

| Person | Owns | Files to know well |
|---|---|---|
| Dhyan | Data processing, labels, candidates, features | `src/pathsnap/{io,geo,track,network,labels,features}.py`, `scripts/build_dataset.py` |
| Dhruva | EDA, Folium maps, report | `scripts/eda.py`, `src/pathsnap/visualize.py`, `results/eda/` |
| Dilipan | Model comparison, Optuna | `scripts/train_models.py`, `scripts/tune_optuna.py`, `results/models/`, `results/optuna/` |
| Dhyeya | HMM/Viterbi, API, UI, deployment | `src/pathsnap/{hmm,inference,evaluation}.py`, `api/main.py`, `ui/src/App.jsx` |

## Likely viva questions (with where the answer lives)
- **Why no per-fix labels, and how did you make them?** `.route` is an ordered arc list. `labels.py` uses monotone DP alignment: each fix gets the route arc minimising total distance while respecting driving order. Fixes more than 60 m from their arc are unlabelled (trips 33 and 54 are mostly offset).
- **Why UTM?** Lon/lat are angles, so Euclidean degrees are wrong (1° of lon shrinks with latitude). Per-trip UTM zone gives metres with < 0.1% distortion at city scale.
- **Why is it a binary classification?** Arc ids are local to each trip, so we can't train a multi-class model over arcs. Instead we score every (fix, candidate) pair: "is this the true arc?" Top-1 per fix = argmax.
- **Why radius 75 m / k 10?** `results/eda/candidate_miss_grid_train.csv`: the smallest candidate set whose miss rate is within 0.2 pp of the best (0.97% on train).
- **Why GroupKFold?** Fixes from one trip are highly correlated. Random K-fold would leak a trip into both train and validation and inflate scores.
- **Primary metric?** Observation-level top-1 accuracy. It is exactly what the evaluator scores and what the HMM consumes. Log loss is secondary because probability quality matters for the HMM emissions.
- **Emission and transition?** See the docstring at the top of `src/pathsnap/hmm.py`.
- **Large gaps / disconnected roads?** Chain restarts at gaps > 120 s or jumps > 70 m/s. Unreachable transitions get a fixed log penalty (−40) instead of −∞. Stitching inserts shortest paths, and the route is left broken only if no path is within 5 km.
- **Leakage?** README section "Data-leakage rules we follow".

## Demo-day checklist
1. Laptop 1: `pip install -r requirements.txt` (exact versions, because the joblib model needs the same sklearn/LightGBM).
2. Put the 10 evaluator trips in `data/eval/<trip_id>/` and delete the rehearsal copies of the student-test trips first.
3. `run_api.bat`, then allow the firewall prompt. Note the IP from `ipconfig`.
4. Laptop 2: `ui/dist` built (`npm install && npm run build`), then `run_ui.bat` and open `http://localhost:5173`.
5. Enter `http://<laptop1-ip>:8000`, click Connect, and check that 10 trips appear.
6. Run all 10 trips once before the evaluator arrives.
7. Keep the Laptop 1 console visible: it shows each request and its step timings (required).
8. Screenshots for the report: the UI with results, and the Laptop 1 console with the `POST /api/predict ... 200` lines.

## Still to do (team)
- [ ] Write the per-model analysis paragraphs (report section 3.2) in your own words, from `results/models/`.
- [ ] Export the Folium maps (`results/eda/*.html`, `results/hmm/*.html`) as screenshots for the PDF.
- [ ] UI and API-console screenshots from the real two-laptop setup.
- [ ] Create the public GitHub repo and push. Check that `artifacts/` is included.
- [ ] Convert the report to PDF.
