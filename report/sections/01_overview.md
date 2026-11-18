## 1. Problem and approach overview
Each trip has a noisy 1 Hz GPS track and a local directed road network. The aim is to name, for every GPS fix, the road arc the vehicle was actually on, and to rebuild a continuous route.

Our pipeline:
1. **Preprocessing.** Clean the track, split it at gaps and jumps, and project it to UTM metres.
2. **Candidates.** Find the 10 nearest arcs within 75 m of each fix.
3. **Classifier.** Score each candidate with a trained classifier that estimates P(arc is correct).
4. **Decoding.** Decode with an HMM and Viterbi, using road connectivity.
5. **Stitching.** Join the chosen arcs into a continuous route with shortest paths.

Steps 1–5 run inside a FastAPI service on Laptop 1. A React UI on Laptop 2 sends trip ids to it and shows the results.

**Data partitions.** 80 TRAIN trips for fitting, cross-validation and all tuning. 10 STUDENT-TEST trips for comparing models and for development checks. The 10 evaluator trips were only used by the API at the final evaluation.

**Main results (student-test partition, 10 trips, 19,160 GPS fixes).**

| Stage | Per-fix accuracy | Route F1 (length-weighted) | Fix-to-fix continuity |
|---|---|---|---|
| Best untuned classifier (LightGBM, Section 3), top-1 | 0.8981 | 0.9507 | 0.9473 |
| Tuned LightGBM (Section 4), top-1 | 0.9038 | 0.9471 | 0.9458 |
| **Tuned LightGBM + HMM/Viterbi (Section 5)** | **0.9247** | **0.9795** | **0.9771** |

Seven models were compared (Section 3). LightGBM was selected and tuned with Optuna (Section 4). The HMM adds about 2 points of per-fix accuracy and 3 points of route F1 on top of the tuned classifier (Section 5).
