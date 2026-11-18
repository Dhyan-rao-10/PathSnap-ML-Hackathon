# PathSnap: Distributed Map-Matching & Route Reconstruction
UE24CS352A Machine Learning, PES University (RR Campus)
Team: Dhyan, Dhruva, Dilipan, Dhyeya · GitHub: `<repo link>`

> Draft built from the repository outputs. Every number comes from a file in `results/`
> (exact tables: `report/results_tables.md`, regenerate with `python scripts/make_report_tables.py`).
> Sections marked **[TEAM]** need your own analysis in your own words before submission.

---

## 1. Problem and approach overview
Each trip has a noisy 1 Hz GPS track and a local directed road network. The aim is to name, for every GPS fix, the road arc the vehicle was actually on, and to rebuild a continuous route.

Our pipeline:
1. **Preprocessing.** Clean the track, split it at gaps and jumps, and project it to UTM metres.
2. **Candidates.** Generate candidate arcs near each fix.
3. **Classifier.** Score each candidate with a trained classifier that estimates P(arc is correct).
4. **Decoding.** Decode with an HMM and Viterbi, using road connectivity.
5. **Stitching.** Join the chosen arcs into a continuous route with shortest paths.

Steps 1–5 run inside a FastAPI service on Laptop 1. A React UI on Laptop 2 sends trip ids to it and shows the results.

## 2.2 Dataset understanding and EDA
*Partitions used: TRAIN (80 trips) and STUDENT-TEST (10 trips). Evaluator trips were never opened.*
Script: `scripts/eda.py`. Outputs are in `results/eda/`.

### Raw data structure (verified, not assumed)
| File | Format (verified) | Notes |
|---|---|---|
| `.track` | lon, lat, t (s), whitespace, no header | 262–19,313 fixes per trip; 0 NaNs; 0 non-monotonic timestamps |
| `.nodes` | lon, lat; row index = node id | 299–155k nodes; 0 isolated nodes |
| `.arcs` | from_node, to_node; row index = arc id | **directed**: a two-way street appears as two arcs, so #arcs ≈ 2 × #nodes |
| `.route` | ordered arc ids | **arc-level, not fix-level**: 59–2,266 arcs vs 262–19,313 fixes. Consecutive arcs connect (to = next from) in ≥ 98.1% of steps; 10 trips revisit an arc |

### Observations
- **Global spread.** Trip centroids range from −100° to 137° longitude, i.e. many cities on several continents (`eda_overview.png`). So **no absolute-coordinate features**: the evaluator trips come from unseen places.
- **Sampling.** 93.8% (train) and 98.2% (student test) of steps are exactly 1 s. There are 590 gaps > 10 s in train. The largest gaps are 42 h (trip 90) and 14.7 h (trip 78), which are really separate drives in one file. There are 147 jumps above 70 m/s.
- **Speeds** (implied, `eda_overview.png`) mostly fall between 0 and 35 m/s, a normal urban/rural driving profile.
- **Road networks** have a median of 19 weakly connected components, and the largest strongly connected component covers 64–100% of nodes. So a shortest path between two candidate arcs can be unreachable, and the HMM must survive that (Section 5). **[TEAM]** Check where the small components lie (map-clip boundary?) on one map.
- **Reference-route quality.** After alignment the median GPS-to-reference-arc distance is ~4–5 m (`labels_and_candidates.png`). Two training trips (33 and 54) have reference routes offset 120–140 m from the GPS. We treat those fixes as unlabelled and do not "correct" them. The reference route is used as reference information, not as error-free ground truth.

### Preprocessing (`src/pathsnap/track.py`, `geo.py`)
1. Drop non-finite and out-of-range coordinates, sort by time, and drop duplicate timestamps. None were present, but the checks stay for unseen trips.
2. **Projection.** Use the per-trip UTM zone (pyproj). Euclidean distance in degrees is wrong because a degree of longitude shrinks with latitude. Within one city, UTM distortion is < 0.1%, far below GPS noise.
3. **Segmentation.** Start a new segment when dt > 120 s or implied speed > 70 m/s. The HMM restarts at each segment.
4. **Kinematics.** Compute heading and speed from fixes i−2 … i+2 inside the same segment. Single-fix differences are too noisy at 1 Hz.

### Turning the arc-level route into per-fix labels (`labels.py`)
Fixes are not linked to arcs in the raw data. We solve a **monotone alignment** with dynamic programming: each fix i gets a route position jᵢ, with j non-decreasing, minimising Σ dist(fixᵢ, route_arc_jᵢ). Monotonicity respects driving order and handles repeated arcs. Fixes more than 60 m from their aligned arc are marked *unlabelled*: they are excluded from training and scoring, and still predicted.

### Candidate generation (`network.py`)
Arcs are densified every 10 m into a KD-tree. For each fix we keep the k nearest arcs (exact perpendicular distance) within radius R. If nothing is within R, the search widens to 300 m. R and k were picked on **train only** from the miss-rate grid (`candidate_miss_grid_train.csv`): the smallest candidate set within 0.2 percentage points of the best. That gave **R = 75 m and k = 10**, with a train candidate miss rate of 0.97% and a student-test miss rate of 0.13%.

### Features (`features.py`), one row per (fix, candidate arc)
| Feature group | Features | Why |
|---|---|---|
| Distance | `dist`, `dist_rank`, `dist_minus_min`, `dist_ratio`, `n_cand` | Closeness is the main signal. *Relative* closeness matters in dense areas with parallel streets |
| Direction | `ang_diff`, `cos_diff`, `along_speed`, `cross_speed`, `speed` | Arcs are directed: the reverse twin of the right street has the same geometry and differs only in direction |
| Arc geometry / topology | `arc_len`, `proj_t`, `d_endpoint`, `out_deg_end`, `in_deg_start`, `is_twoway` | Junction ambiguity: near a node, several arcs are equally close |
| Temporal context | `ctx_dist_mean`, `ctx_excess_mean`, `ctx_ang_mean` (±5 fixes) | One noisy fix can be closer to a side street. The right arc explains the *neighbouring* fixes too |
| Sampling | `dt_prev` | Gaps reduce reliability |

Positive rate: 13.2% of candidate rows. The training table has 1.40M rows (186k labelled fixes).

**Folium maps:** `results/eda/map_*.html` show GPS fixes, nearby road network and reference route, with a legend and layer control. **[TEAM]** Add screenshots.

---

## 3. Model experimentation and comparison
Script: `scripts/train_models.py`. Outputs: `results/models/`.

**Task framing.** Arc ids are local to each trip, so a multi-class model over arcs can't generalise. We score every (fix, candidate) pair as a binary problem, "is this the fix's reference arc?". The predicted arc for a fix is the highest-scoring candidate.

**Protocol (same for all 7 models).**
- **Validation.** 5-fold `GroupKFold` grouped by trip on the 80 training trips, then a refit on all training trips and evaluation on the 10 student-test trips.
- **Training samples.** Samples are drawn by fix, so each fix keeps its whole candidate set.
  - Most models: 80k fixes (601k rows) per training set.
  - KNN: 20k fixes, because prediction cost grows with the training set.
  - Linear SVM: 40k fixes, because of the calibration folds.
- **Metrics:**
  - Macro F1 on candidate rows at a 0.5 threshold.
  - Log loss.
  - Observation-level top-1 accuracy.
  - Candidate miss rate.
  - Classifier-only route continuity: the share of consecutive fixes whose top-1 arcs are equal or directly connected.
  - Length-weighted route F1.
  - Inference time.

**Primary selection metric: observation-level top-1 accuracy.** It is exactly what is scored per GPS fix, and it is what the decoder consumes. Macro F1 depends on a 0.5 threshold that we never use for routing. Log loss is the secondary metric because the HMM uses the probabilities themselves as emissions.

### 3.1 Overall comparison (actual results)
| Model | Family | Key configuration | CV macro F1 (±std) | CV top-1 | Student-test top-1 | Route continuity (clf-only, test) | Student-test macro F1 | Student-test log loss | Inference ms / 1000 fixes |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | StandardScaler + L2, C=1 | 0.9117 ± 0.0132 | 0.8831 | 0.8792 | 0.9240 | 0.9125 | 0.1001 | 1.1 |
| Gaussian NB | Probabilistic | var_smoothing=1e-9 | 0.8450 ± 0.0044 | 0.8555 | 0.8487 | 0.9206 | 0.8402 | 0.6036 | 3.3 |
| KNN | Instance-based | k=50, distance-weighted, scaled | 0.9218 ± 0.0086 | 0.8745 | 0.8848 | 0.9279 | 0.9241 | 0.1041 | 1201.6 |
| Linear SVM | Kernel | LinearSVC C=1 + sigmoid calibration | 0.9116 ± 0.0135 | 0.8834 | 0.8793 | 0.9235 | 0.9118 | 0.1007 | 2.6 |
| Random Forest | Ensemble (bagging) | 200 trees, depth 20, min_leaf 5 | 0.9410 ± 0.0094 | 0.8982 | 0.8944 | 0.9426 | 0.9373 | 0.0678 | 24.9 |
| **LightGBM** | Boosting | 400 trees, lr 0.05, 63 leaves | **0.9410 ± 0.0106** | **0.8995** | **0.8981** | **0.9473** | **0.9378** | **0.0673** | 22.6 |
| MLP | Neural network | (64,32) ReLU, Adam 1e-3, α=1e-4, 25 epochs | 0.9339 ± 0.0093 | 0.8899 | 0.8866 | 0.9459 | 0.9308 | 0.0808 | 5.0 |

Plots: `model_comparison.png` (CV macro F1 with fold std, student-test top-1 and macro F1, log loss), `cv_fold_stability.png`, `efficiency_continuity.png`, `confusion_matrices.png`, `history_lgbm.png` and `history_mlp.png` (the only two models with real iteration-wise histories).

**Selection: LightGBM.**
- Highest CV top-1, highest student-test top-1, lowest log loss (best probabilities for the HMM), and highest classifier-only continuity.
- Random Forest is statistically tied on CV macro F1 and close on top-1. LightGBM fits about 9× faster (6.5 s vs 58 s on 601k rows) and its model file is far smaller (1.3 MB vs hundreds of MB for RF). That matters for the API laptop and for the repo size limit.
- Fold-to-fold variation (std of top-1 ≈ 0.022–0.025) is mostly *between trips* and is shared by all models. The fold ranking of LightGBM vs RF is consistent.

### 3.2 Per-model notes (facts from our runs; **[TEAM]** expand each into your own analysis)
For every model, the loss/objective is the standard one for its family (listed below). Hyperparameters are the initial values in the table above.

- **Logistic Regression.**
  - Objective: L2-regularised binary cross-entropy (lbfgs).
  - Train F1 0.9146 vs CV 0.9117: no overfitting. It is underfitting, since it is linear in features where the true boundary is not (for example, distance only matters *relative to* other candidates).
  - Fastest inference.
- **Gaussian NB.**
  - Objective: maximum likelihood of per-class Gaussians.
  - Its feature-independence assumption is badly violated: `dist`, `ctx_dist_mean` and `dist_minus_min` are strongly correlated. The result is over-confident probabilities: log loss 0.60, about 9× worse than LightGBM, and precision 0.61 against recall 0.91.
  - Train and CV F1 are equal (0.848 vs 0.845), so this is underfitting / model bias.
- **KNN.**
  - No training objective (lazy learner). Distance-weighted vote of k=50 scaled neighbours.
  - Train F1 0.940 vs CV 0.922 is mild overfitting.
  - Inference is about 1000× slower than the linear models (1.2 s per 1000 fixes), which is impractical for the API.
- **Linear SVM.**
  - Objective: hinge loss with L2 penalty; probabilities from Platt (sigmoid) calibration.
  - Nearly identical to logistic regression (top-1 0.8793 vs 0.8792), as expected for two linear boundaries on the same features.
- **Random Forest.**
  - Objective: Gini impurity per split, bagged.
  - Train F1 0.963 vs CV 0.941 is moderate overfitting, kept in check by depth 20 and min_leaf 5.
  - Second-best overall.
- **LightGBM.**
  - Objective: binary log loss, gradient boosting.
  - Train F1 0.967 vs CV 0.941. The fold-0 history (`history_lgbm.png`) shows validation log loss bottoming out at iteration 98 (0.0824) and creeping up to 0.0862 by iteration 400. The untuned configuration overfits slightly, which motivates the Bayesian tuning in Section 4.
- **MLP.**
  - Objective: binary cross-entropy with Adam.
  - `history_mlp.png`: train log loss falls from 0.073 to 0.054 over 25 epochs while validation stays flat around 0.093. It overfits from the first few epochs, so early stopping or stronger α would help.
  - Train F1 0.950 vs CV 0.934.

## 4. Bayesian hyperparameter optimisation (LightGBM)
Script: `scripts/tune_optuna.py`. Outputs: `results/optuna/`.

1. **Hyperparameters searched:**
   - `n_estimators` 100–600 (step 50)
   - `learning_rate` 0.01–0.2 (log)
   - `num_leaves` 15–255 (log)
   - `min_child_samples` 5–200 (log)
   - `subsample` 0.5–1 (with `subsample_freq=1`)
   - `colsample_bytree` 0.4–1
   - `reg_lambda` 1e-3–10 (log)
   - `reg_alpha` 1e-3–10 (log)

   These ranges cover capacity (leaves, trees), step size, and the three regularisers. Section 3 showed mild overfitting, so regularisation was the focus.
2. **Sampler:** Optuna TPE (seed 42). The untuned configuration was enqueued as trial 0 so the search starts from the baseline.
3. **Objective:** mean observation-level top-1 accuracy over 3-fold GroupKFold on a 40k-fix sample of the TRAIN trips.
4. **Trials:** 30, in 284 s.
5. **Best:** `n_estimators=450, learning_rate=0.0441, num_leaves=25, min_child_samples=157, subsample=0.766, colsample_bytree=0.592, reg_lambda=0.0017, reg_alpha=0.0015`. The search preferred much smaller trees with large leaves, i.e. more regularised.
6. **Tuned vs untuned:** both refit on all 186k labelled train fixes, with the same 5-fold CV and the same student-test set.

| | CV macro F1 | CV top-1 | Student-test top-1 | Student-test macro F1 | Student-test log loss | Fit time (s) | Inference ms / 1000 fixes |
|---|---|---|---|---|---|---|---|
| Untuned | 0.9408 | 0.8974 | 0.8901 | 0.9338 | 0.0773 | 14.2 | 24.3 |
| Tuned | 0.9413 | 0.8997 | **0.9037** | **0.9395** | **0.0665** | 13.6 | 19.8 |

The tuned model is better on every metric, by +1.4 points top-1 and −14% log loss on the student test. The CV gain is small (+0.2 points). The improvement is real but modest, and the student-test gain partly reflects which 10 trips happen to be in that partition. Plots: `optuna_history_importance.png`, `tuned_vs_untuned.png`.

## 4. Bayesian hyperparameter optimisation
<!-- FILLED BELOW -->

## 5. HMM and Viterbi route reconstruction
Code: `src/pathsnap/hmm.py`. Tuning and evaluation: `scripts/tune_eval_hmm.py`. Outputs: `results/hmm/`.

1. **Hidden states.**
   - At fix *i*, the states are its candidate arcs: the top 8 by classifier probability, out of the k = 10 generated.
   - Each state keeps the fix's projected position along that arc, so the candidate-to-arc correspondence is never lost: a state *is* an arc id.
2. **Emission.**
   - log P(zᵢ | s) = w · log( p̂(s) / Σₛ′ p̂(s′) ).
   - p̂ is the tuned LightGBM probability, normalised over the fix's candidate set.
   - w is the emission weight; it sets how much the decoder trusts the classifier against road continuity.
3. **Transition** (Newson & Krumm-style).
   - log P(s′ | s) = −|d_route(s→s′) − d_gps(i−1→i)| / β.
   - d_route is the directed network distance: the rest of arc s, plus the shortest path from s's end node to s′'s start node (bounded Dijkstra, cached per node), plus the position on s′.
   - On the same arc, forward movement is just the difference in position; backward movement over 10 m must loop through the network.
   - Wrong-way travel and jumps to parallel streets therefore get large route distances and low probability.
4. **Missing candidates and gaps.**
   - Fixes with no arc within 75 m use a 300 m fallback. If there is still nothing, the fix is skipped (arc = −1) and the chain links its neighbours directly.
   - At gaps over 120 s or jumps over 70 m/s, the chain is cut and Viterbi restarts with emissions only.
   - The Dijkstra search radius grows with the GPS distance and dt (up to 5 km).
5. **Disconnected consecutive segments.**
   - An unreachable transition gets a constant log-penalty of −40 instead of −∞, so one bad fix or a missing network link cannot make the whole sequence impossible.
   - When the decoded arcs are stitched into a route, shortest paths are inserted between consecutive distinct arcs. If none exists within 5 km, a *route break* is recorded and reported (3 breaks across all 10 student-test trips).
6. **Viterbi recurrence.**
   - δᵢ(s′) = max_s [δᵢ₋₁(s) + log P(s′|s)] + log P(zᵢ|s′), with back-pointers, then backtracking from argmax δ_N.
   - All in log space. O(N·M²) per trip with M ≤ 8.

**Decoder tuning (train only).**
- β ∈ {2, 5, 10, 20, 40} × w ∈ {0.5, 1, 2, 3, 4}, on 25 train trips, using **out-of-fold** probabilities from the 5-fold CV of the tuned model. This means the decoder sees probabilities as noisy as on unseen trips.
- Best: **β = 10 m, w = 2** (interior optimum), with per-fix accuracy 0.9141 against 0.8917 for classifier-only on the same trips (`hmm_grid_train_oof.csv`, `hmm_tuning.png`).

**Classifier-only vs classifier + HMM (student test, 19,160 fixes)**

| | Per-fix accuracy | Fix-to-fix continuity | Route F1 (length-weighted) | Route precision | Route recall |
|---|---|---|---|---|---|
| Classifier only (top-1) | 0.9038 | 0.9458 | 0.9471 | n/a | n/a |
| Classifier + HMM/Viterbi | **0.9247** | **0.9771** | **0.9795** | 0.9632 | 0.9970 |

Per-trip results: `student_test_clf_vs_hmm.csv` and `clf_vs_hmm_student_test.png`.
- **Wins.** The HMM improved accuracy on all 10 trips. The largest gains came where the classifier jumped between parallel or adjacent arcs: trip 76 went from 93.4% to 97.6%, trip 56 from 92.7% to 96.9%, and trip 9 from 88.2% to 91.9%.
- **Where it did not help: trip 77**, which went from 80.1% to only 80.4%.
  - Of its 765 remaining errors, **575 choose the reverse-direction twin of the correct street**.
  - The vehicle's median speed on that trip is 3.7 m/s, so GPS heading is unreliable.
  - Both directions of a two-way street are geometrically identical, and for slow movement their route-distance costs are almost equal. Neither the features nor the transition model can tell them apart.
  - Route F1 still rises from 0.899 to 0.936, because the stitched route covers the right street.
- **Fix-level continuity is not 100% even after the HMM** (0.977). At 1 Hz a car often passes a whole short arc between two fixes. The *stitched* route fills those arcs back in, and its continuity is 0.999 (route breaks only).

**[TEAM]** Add the screenshot of `results/hmm/map_student_test_00000076.html`, with the classifier-only layer switched on, to show a corrected junction.

## 6. Geographical visualisation (Folium)
Map code: `src/pathsnap/visualize.py::trip_map`, called from `scripts/eda.py` and `scripts/tune_eval_hmm.py`.
Every map has:
- **Layers:** GPS fixes (blue dots, with a time tooltip), nearby road network (grey; two-way streets drawn once), reference route (green, thick), classifier-only top-1 route (orange, off by default), and HMM predicted route (red).
- **Controls:** start and end markers, a legend, a LayerControl, and fit-to-bounds zoom.

| Map | Purpose |
|---|---|
| `results/eda/map_train_00000003.html`, `..._00000021.html`, `..._00000090.html`, `map_student_test_00000009.html` | EDA: raw GPS vs network vs reference |
| `results/hmm/map_student_test_00000009.html`, `..._00000056.html`, `..._00000076.html` | Prediction: classifier-only vs HMM vs reference |

**[TEAM]** Paste one screenshot of each kind. Zoom in on a junction or parallel-road area where the HMM fixes a classifier mistake.

## 7. API, UI and deployment
- **Laptop 1** runs `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
  - At start-up it loads `artifacts/classifier_lgbm_tuned.joblib` and `artifacts/config.json`.
  - `GET /api/trips` lists the trips in the eval folder.
  - `POST /api/predict {trip_id}` runs load → candidates → features → classifier → Viterbi → stitching.
  - After prediction finishes, `pathsnap/evaluation.py` reads the trip's `.route`. It returns only aggregate scores (per-fix accuracy for HMM and classifier-only, route precision/recall/F1).
  - Reference arcs and geometry are never sent to the UI. The prediction code never imports the route reader.
  - Every request is logged with client IP, status, per-step timings and accuracy.
- **Laptop 2** serves the built React app (`ui/dist`, any static server).
  - The UI lists the 10 trips and sends the chosen id.
  - It draws the GPS fixes and the predicted route on a Leaflet map, with metric cards, a timing breakdown and a run log.
- **Typical latency** (student-test trips, Laptop 1): about 1–5 s per trip. Viterbi plus route stitching dominate.

**[TEAM]** Add a screenshot of the UI after a run, and one of the Laptop 1 console showing the `POST /api/predict … 200` lines with step timings.
