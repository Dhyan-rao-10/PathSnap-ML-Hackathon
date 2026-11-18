# PathSnap: results tables (auto-generated from results/)

## Dataset summary (raw files, train + student_test)

| statistic | student_test | train |
|---|---|---|
| trips | 10.000 | 80.000 |
| fixes_total | 19160.000 | 208417.000 |
| fixes_median | 1420.000 | 1529.000 |
| nan_total | 0.000 | 0.000 |
| nonmonotonic_t | 0.000 | 0.000 |
| dt_eq_1_mean | 0.982 | 0.938 |
| dt_max | 210.000 | 153064.000 |
| gaps_gt_10s | 27.000 | 590.000 |
| jumps | 1.000 | 147.000 |
| length_km_median | 20.936 | 24.066 |
| arcs_median | 13145.000 | 19526.500 |
| isolated_nodes | 0.000 | 0.000 |
| weak_components_median | 11.000 | 20.000 |
| largest_scc_frac_min | 0.817 | 0.636 |
| route_cont_min | 0.997 | 0.981 |
| trips_with_repeats | 1.000 | 9.000 |

## Candidate miss rate on TRAIN (rows = radius m, cols = k)

| radius | 2 | 4 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| 20 | 0.1066 | 0.0601 | 0.0480 | 0.0446 | 0.0436 | 0.0430 | 0.0425 |
| 30 | 0.0922 | 0.0426 | 0.0286 | 0.0242 | 0.0226 | 0.0218 | 0.0210 |
| 50 | 0.0845 | 0.0333 | 0.0185 | 0.0134 | 0.0115 | 0.0106 | 0.0096 |
| 75 | 0.0832 | 0.0319 | 0.0169 | 0.0117 | 0.0097 | 0.0088 | 0.0078 |
| 100 | 0.0832 | 0.0319 | 0.0169 | 0.0117 | 0.0097 | 0.0088 | 0.0078 |

Chosen: radius = 75.0 m, k = 10 (train miss rate 0.0097).

## 3.1 Overall model comparison

| Model | Algorithm family | Key configuration | CV macro F1 (mean±std) | CV top-1 | Student-test top-1 | Route continuity (clf-only) | Student-test macro F1 | Student-test log loss | Inference ms / 1000 fixes |
|---|---|---|---|---|---|---|---|---|---|
| logreg | Linear | StandardScaler + L2 LogisticRegression C=1.0, lbfgs | 0.9117 ± 0.0132 | 0.8831 | 0.8792 | 0.9240 | 0.9125 | 0.1001 | 1.0944 |
| gnb | Probabilistic | GaussianNB var_smoothing=1e-9 | 0.8450 ± 0.0044 | 0.8555 | 0.8487 | 0.9206 | 0.8402 | 0.6036 | 3.3242 |
| knn | Instance-based | StandardScaler + KNN k=50, distance-weighted, euclidean | 0.9218 ± 0.0086 | 0.8745 | 0.8848 | 0.9279 | 0.9241 | 0.1041 | 1201.5547 |
| linsvm | Kernel (linear SVM) | StandardScaler + LinearSVC C=1.0, sigmoid-calibrated (3-fold) | 0.9116 ± 0.0135 | 0.8834 | 0.8793 | 0.9235 | 0.9118 | 0.1007 | 2.5765 |
| rf | Ensemble (bagging) | RandomForest 200 trees, max_depth=20, min_samples_leaf=5 | 0.9410 ± 0.0094 | 0.8982 | 0.8944 | 0.9426 | 0.9373 | 0.0678 | 24.9384 |
| lgbm | Boosting | LightGBM 400 trees, lr=0.05, num_leaves=63 (defaults otherwise) | 0.9410 ± 0.0106 | 0.8995 | 0.8981 | 0.9473 | 0.9378 | 0.0673 | 22.6169 |
| mlp | Neural network | StandardScaler + MLP (64,32) ReLU, Adam lr=1e-3, alpha=1e-4, 25 epochs | 0.9339 ± 0.0093 | 0.8899 | 0.8866 | 0.9459 | 0.9308 | 0.0808 | 4.9947 |

## Over/under-fitting evidence and extra metrics

| model | train_rows_used | train_macro_f1 | cv_macro_f1_mean | test_macro_f1 | train_log_loss | cv_log_loss_mean | test_log_loss | test_precision_pos | test_recall_pos | test_route_f1 | fit_time_s_full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 601137 | 0.9146 | 0.9117 | 0.9125 | 0.1044 | 0.1078 | 0.1001 | 0.8477 | 0.8468 | 0.9299 | 1.2867 |
| gnb | 601137 | 0.8483 | 0.8450 | 0.8402 | 0.5597 | 0.5950 | 0.6036 | 0.6118 | 0.9077 | 0.9053 | 0.2057 |
| knn | 150770 | 0.9403 | 0.9218 | 0.9241 | 0.0801 | 0.1174 | 0.1041 | 0.8947 | 0.8410 | 0.9356 | 0.0713 |
| linsvm | 300856 | 0.9138 | 0.9116 | 0.9118 | 0.1048 | 0.1076 | 0.1007 | 0.8504 | 0.8417 | 0.9313 | 5.2166 |
| rf | 601137 | 0.9626 | 0.9410 | 0.9373 | 0.0488 | 0.0682 | 0.0678 | 0.9059 | 0.8754 | 0.9459 | 57.9596 |
| lgbm | 601137 | 0.9668 | 0.9410 | 0.9378 | 0.0412 | 0.0705 | 0.0673 | 0.9036 | 0.8794 | 0.9507 | 6.5397 |
| mlp | 601137 | 0.9504 | 0.9339 | 0.9308 | 0.0592 | 0.0833 | 0.0808 | 0.9103 | 0.8491 | 0.9459 | 89.9783 |

## 4. Bayesian optimisation (LightGBM, Optuna TPE)

Objective: 3-fold GroupKFold mean top-1 accuracy. Trials: 30. Search time: 284 s on a 40,000-fix sample. Best search score: 0.9001.

Best parameters: `{"n_estimators": 450, "learning_rate": 0.044110582978665505, "num_leaves": 25, "min_child_samples": 157, "subsample": 0.7661911021778365, "colsample_bytree": 0.5924886460817442, "reg_lambda": 0.001683852628154045, "reg_alpha": 0.0014516427144130194, "subsample_freq": 1}`

| model | cv_macro_f1 | cv_macro_f1_std | cv_log_loss | cv_top1 | cv_top1_std | test_top1 | test_macro_f1 | test_log_loss | test_precision_pos | test_recall_pos | fit_time_full_s | inference_ms_per_1000_fixes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untuned | 0.9408 | 0.0106 | 0.0747 | 0.8974 | 0.0218 | 0.8901 | 0.9338 | 0.0773 | 0.8904 | 0.8786 | 14.1628 | 24.3122 |
| tuned | 0.9413 | 0.0106 | 0.0677 | 0.8997 | 0.0225 | 0.9037 | 0.9395 | 0.0665 | 0.9060 | 0.8829 | 13.6153 | 19.7567 |

## 5. HMM decoder tuning grid (train trips, out-of-fold probabilities)

| beta | emission_weight | accuracy | seconds |
|---|---|---|---|
| 2 | 0.5000 | 0.8934 | 12.3050 |
| 2 | 1.0000 | 0.9060 | 8.5285 |
| 2 | 2.0000 | 0.9104 | 8.2111 |
| 2 | 3.0000 | 0.9127 | 8.1154 |
| 2 | 4.0000 | 0.9128 | 7.9255 |
| 5 | 0.5000 | 0.9028 | 8.1422 |
| 5 | 1.0000 | 0.9098 | 8.5304 |
| 5 | 2.0000 | 0.9137 | 7.9523 |
| 5 | 3.0000 | 0.9130 | 8.3610 |
| 5 | 4.0000 | 0.9129 | 7.9684 |
| 10 | 0.5000 | 0.9049 | 8.4900 |
| 10 | 1.0000 | 0.9109 | 8.6646 |
| 10 | 2.0000 | 0.9141 | 8.2346 |
| 10 | 3.0000 | 0.9126 | 8.2739 |
| 10 | 4.0000 | 0.9107 | 8.3951 |
| 20 | 0.5000 | 0.9071 | 8.8826 |
| 20 | 1.0000 | 0.9112 | 7.9544 |
| 20 | 2.0000 | 0.9121 | 8.1693 |
| 20 | 3.0000 | 0.9098 | 8.7982 |
| 20 | 4.0000 | 0.9078 | 8.5177 |
| 40 | 0.5000 | 0.9083 | 8.6752 |
| 40 | 1.0000 | 0.9093 | 8.8920 |
| 40 | 2.0000 | 0.9088 | 8.6317 |
| 40 | 3.0000 | 0.9078 | 8.6880 |
| 40 | 4.0000 | 0.9066 | 8.7114 |

## 5. Classifier-only vs classifier + HMM (student test)

| trip | n_fixes | clf_accuracy | hmm_accuracy | clf_fix_continuity | hmm_fix_continuity | route_breaks | clf_route_f1 | hmm_route_f1 | hmm_route_precision | hmm_route_recall | hmm_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 00000000 | 2503.0000 | 0.9525 | 0.9604 | 0.9804 | 0.9952 | 2.0000 | 0.9862 | 0.9933 | 0.9877 | 0.9991 | 0.3490 |
| 00000007 | 1354.0000 | 0.9513 | 0.9690 | 0.9970 | 0.9993 | 0.0000 | 0.9876 | 0.9872 | 0.9815 | 0.9930 | 0.1984 |
| 00000009 | 1049.0000 | 0.8818 | 0.9190 | 0.8979 | 0.9427 | 0.0000 | 0.9069 | 0.9627 | 0.9389 | 0.9878 | 0.5199 |
| 00000040 | 2995.0000 | 0.9359 | 0.9469 | 0.9706 | 0.9953 | 0.0000 | 0.9723 | 0.9973 | 0.9946 | 1.0000 | 0.4580 |
| 00000052 | 262.0000 | 0.9847 | 1.0000 | 0.9464 | 0.9655 | 0.0000 | 0.9595 | 0.9984 | 1.0000 | 0.9969 | 0.1165 |
| 00000056 | 3958.0000 | 0.9270 | 0.9692 | 0.9121 | 0.9644 | 1.0000 | 0.9212 | 0.9775 | 0.9569 | 0.9990 | 1.1858 |
| 00000065 | 768.0000 | 0.8737 | 0.8945 | 0.9544 | 0.9844 | 0.0000 | 0.9431 | 0.9836 | 0.9677 | 1.0000 | 0.2958 |
| 00000076 | 1486.0000 | 0.9341 | 0.9758 | 0.9192 | 0.9650 | 0.0000 | 0.9415 | 0.9699 | 0.9429 | 0.9984 | 1.0899 |
| 00000077 | 3911.0000 | 0.8013 | 0.8044 | 0.9621 | 0.9903 | 0.0000 | 0.8991 | 0.9361 | 0.8798 | 1.0000 | 1.0586 |
| 00000091 | 874.0000 | 0.9119 | 0.9405 | 0.9175 | 0.9691 | 0.0000 | 0.9537 | 0.9888 | 0.9822 | 0.9956 | 0.3593 |
| ALL (fix-weighted) | 19160.0000 | 0.9038 | 0.9247 | 0.9458 | 0.9771 | 3.0000 | 0.9471 | 0.9795 | 0.9632 | 0.9970 | 0.5631 |
