# Viva prep: progress review (EDA, features, algorithm, accuracy, tuning)

**Modules completed:**
1. Data processing + EDA
2. Feature engineering
3. Model comparison (7 models)
4. Bayesian tuning
5. HMM/Viterbi
6. FastAPI
7. React UI

Everything is reproducible from `scripts/` (see README).

---

## 1. EDA (`scripts/eda.py` → `results/eda/`)
**Show:** `eda_overview.png`, `trajectories_with_network.png`, `labels_and_candidates.png`, any `map_*.html` (Folium).

- **Data:** 80 train + 10 student-test trips, with 208,417 + 19,160 GPS fixes.
- **Each trip folder:**
  - `.track`: lon, lat, time
  - `.nodes`: lon, lat
  - `.arcs`: from_node, to_node; **directed**, so two-way roads appear twice
  - `.route`: the ordered arc ids that were driven
- **Quality:**
  - **0 NaNs**, 0 out-of-order timestamps.
  - ~94% of steps are exactly 1 s.
  - 590 gaps > 10 s. The largest is **42 hours** (trip 90): two drives in one file.
  - 147 GPS jumps > 70 m/s.
- **Spread:** trips come from cities all over the world (longitude −100° to 137°). So we use **no absolute lat/lon features**, otherwise the model wouldn't generalise to new cities.
- **Key discovery: no per-point labels.** `.route` gives arcs, not which GPS point is on which arc. There are 5–10× fewer route arcs than GPS points. We built the labels ourselves with a **monotone dynamic-programming alignment** (points must follow route order).
- **Label quality:** the median GPS-to-road distance is ~4–5 m. Trips 33 and 54 have routes 120–140 m off their GPS. Points more than 60 m off are marked unlabelled rather than trusted.
- **Preprocessing:**
  - Projected to **UTM metres**, because degrees aren't metres: a degree of longitude shrinks with latitude.
  - Split the track at gaps > 120 s or jumps > 70 m/s.
  - Heading and speed computed from ±2 neighbouring points.

## 2. Feature selection / engineering (`src/pathsnap/features.py`)
**Show:** `results/models/feature_importance_tuned_lgbm.png`

**Framing:**
- For each GPS point, find the **10 nearest road arcs within 75 m** (the candidates).
- Each (point, candidate) pair becomes one row. The label is 1 if it's the true arc.
- That gives 1.4M training rows; 13% are positive.

**Why 75 m / 10:** a grid search on train only (`candidate_miss_grid_train.csv`). The true arc is missing from the candidates only 0.97% of the time. Bigger values gave almost no gain and more rows.

**20 features in 5 groups:**

| Group | Features | Why |
|---|---|---|
| Distance | dist, dist_rank, dist_minus_min, dist_ratio, n_cand | Closest road is the main signal |
| Direction | ang_diff, cos_diff, along_speed, cross_speed, speed | Arcs are directed: the right road in the wrong direction looks identical |
| Arc geometry | arc_len, proj_t, d_endpoint, in/out degree, is_twoway | Near junctions, many arcs are equally close |
| Temporal context (±5 points) | ctx_dist_mean, ctx_excess_mean, ctx_ang_mean | One noisy point can jump to a side street; the right arc also fits the neighbours |
| Sampling | dt_prev | Gaps reduce reliability |

**What the model actually uses** (tuned LightGBM, % of total gain):
- `dist_minus_min` 30.4%, `ctx_ang_mean` 20.4%, `dist_rank` 9.2%, `dist_ratio` 7.7%, `d_endpoint` 6.2%. The top 5 carry about 74% of the gain.
- **Raw `dist` is only 1.6%.** What matters is *relative* distance, i.e. how much closer this arc is than the other candidates. This justifies the engineered features.
- The temporal-context direction feature is #2, so smoothing over neighbouring points works.
- `dt_prev` contributes 0.03%: a candidate to drop. **Next step:** retrain without the bottom features and confirm accuracy doesn't fall.

## 3. ML algorithms (`scripts/train_models.py` → `results/models/`)
**Show:** `model_comparison.png`, `cv_fold_stability.png`, `history_mlp.png`, `history_lgbm.png`.

**Protocol:** 7 models from 7 families, all evaluated with **5-fold GroupKFold grouped by trip**.
- **Why grouped by trip:** points from the same trip are near-duplicates. Random K-fold would leak the trip into both sides and inflate scores.
- Then each model is refit on all 80 trips and tested on the 10 student-test trips.

| Model | Family | CV macro F1 | CV top-1 | Test top-1 | Test log loss |
|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.912 | 0.883 | 0.879 | 0.100 |
| Gaussian NB | Probabilistic | 0.845 | 0.856 | 0.849 | 0.604 |
| KNN (k=50) | Instance | 0.922 | 0.875 | 0.885 | 0.104 |
| Linear SVM | Kernel | 0.912 | 0.883 | 0.879 | 0.101 |
| Random Forest | Bagging | 0.941 | 0.898 | 0.894 | 0.068 |
| **LightGBM** | **Boosting** | **0.941** | **0.900** | **0.898** | **0.067** |
| MLP (64,32) | Neural net | 0.934 | 0.890 | 0.887 | 0.081 |

**Why LightGBM:**
- Best top-1 and best log loss (most reliable probabilities, which the HMM needs).
- About as good as Random Forest, but trains ~9× faster (6.5 s vs 58 s) and the model file is 1.3 MB.
- The relationship is non-linear: distance only matters *relative* to the others. That's why linear models plateau around 88%.

**Behaviours to point out:**
- **Naive Bayes:** log loss is 9× worse. It assumes independent features, but dist, dist_ratio and ctx_dist are strongly correlated, so it is over-confident.
- **MLP overfits:** train loss falls from 0.073 to 0.054 while validation stays flat at ~0.093 (`history_mlp.png`).
- **KNN** is ~1000× slower at inference (1.2 s per 1000 points), which is impractical for the API.

## 4. Accuracy (what we measure and why)
**Primary metric: observation-level top-1 accuracy.** For each GPS point, is the highest-scoring arc the true one? It's exactly what the evaluator scores per point.

Also reported:
- Macro F1 (threshold 0.5)
- Log loss (probability quality)
- Candidate miss rate
- Route continuity
- Length-weighted route F1
- Inference time

**Final results on the 10 student-test trips (19,160 points):**

| Stage | Per-point accuracy | Route F1 | Continuity |
|---|---|---|---|
| Tuned LightGBM alone | 90.4% | 0.947 | 94.6% |
| **+ HMM/Viterbi** | **92.5%** | **0.980** | **97.7%** |

- The HMM improved all 10 trips. Best gains: trip 76 (93.4% → 97.6%) and trip 56 (92.7% → 96.9%).
- **Weak case: trip 77 at 80.4%.** 75% of its errors pick the opposite direction of the right street. The car is crawling (median 3.7 m/s), so its heading is unreliable.

## 5. Hyperparameter tuning (`scripts/tune_optuna.py` → `results/optuna/`)
**Show:** `optuna_history_importance.png`, `tuned_vs_untuned.png`.

- **Method:** Bayesian optimisation with **Optuna TPE**. TPE builds a probability model of good vs bad configurations and samples where improvement is likely, instead of a blind grid. Trial 0 was the untuned baseline.
- **Objective:** mean top-1 accuracy, 3-fold GroupKFold, on training trips only.
- **Search:** 30 trials in 284 s.
- **Search space:**
  - n_estimators 100–600
  - learning_rate 0.01–0.2 (log)
  - num_leaves 15–255 (log)
  - min_child_samples 5–200 (log)
  - subsample 0.5–1
  - colsample_bytree 0.4–1
  - L1/L2 regularisation 1e-3–10 (log)
- **Best:** num_leaves 25 (was 63), min_child_samples 157, lr 0.044, 450 trees, subsample 0.77, colsample 0.59. That means **smaller, more regularised trees**, which fits our observation that the untuned model overfit: its validation loss bottoms at iteration 98 of 400.

| | CV top-1 | Test top-1 | Test macro F1 | Test log loss | Inference ms / 1000 points |
|---|---|---|---|---|---|
| Untuned | 0.897 | 0.890 | 0.934 | 0.077 | 24.3 |
| **Tuned** | **0.900** | **0.904** | **0.940** | **0.066** | **19.8** |

- **HMM decoder parameters were tuned too:** β ∈ {2…40} × emission weight ∈ {0.5…4}, on out-of-fold training predictions. Best: β = 10 m, weight = 2 (`hmm_tuning.png`).

## Likely follow-up questions
- **Why not multi-class over road IDs?** Arc IDs are different in every trip, so there's nothing to learn across trips. Binary "is this the right arc" generalises.
- **Leakage?** No trip is split across folds or partitions. Radius, k and HMM parameters were chosen on train only. Evaluator trips were never opened.
- **What's left?** Feature pruning experiment, report write-up, two-laptop network test.
