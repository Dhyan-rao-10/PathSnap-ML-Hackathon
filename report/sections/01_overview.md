## 1. Problem and approach overview
Each trip has a noisy 1 Hz GPS track and a local directed road network. The aim is to name, for every GPS fix, the road arc the vehicle was actually on, and to rebuild a continuous route.

Our pipeline:
1. **Preprocessing.** Clean the track, split it at gaps and jumps, and project it to UTM metres.
2. **Candidates.** Generate candidate arcs near each fix.
3. **Classifier.** Score each candidate with a trained classifier that estimates P(arc is correct).
4. **Decoding.** Decode with an HMM and Viterbi, using road connectivity.
5. **Stitching.** Join the chosen arcs into a continuous route with shortest paths.

Steps 1–5 run inside a FastAPI service on Laptop 1. A React UI on Laptop 2 sends trip ids to it and shows the results.
