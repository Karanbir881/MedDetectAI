## breastcancer: baseline vs improved models

569 samples. **[app]** = the 5 inputs the web form collects (concave points_mean, area_mean, radius_mean, perimeter_mean, concavity_mean); **[select5]** = the 5 best of all 30 columns, chosen inside each training fold; **[all]** = all 30 columns. 5x repeated stratified 5-fold cross-validation (seed 42), the same folds for every model. Mean ± standard deviation over folds.

| model | roc auc | balanced accuracy | accuracy | sensitivity | specificity | macro f1 | Δ ROC-AUC vs baseline (p) |
|---|---|---|---|---|---|---|---|
| baseline (HealthCure RF) [app] | 0.9729 ± 0.0096 | 0.9118 ± 0.0236 | 0.9188 ± 0.0201 | 0.8841 ± 0.0493 | 0.9395 ± 0.0264 | 0.9128 ± 0.0217 | — |
| tuned random forest [app] | 0.9743 ± 0.0080 | 0.9138 ± 0.0256 | 0.9209 ± 0.0223 | 0.8860 ± 0.0544 | 0.9417 ± 0.0322 | 0.9150 ± 0.0241 | +0.0015 (p=0.607) |
| logistic regression [app] | 0.9770 ± 0.0071 | 0.9038 ± 0.0291 | 0.9153 ± 0.0251 | 0.8586 ± 0.0614 | 0.9490 ± 0.0345 | 0.9081 ± 0.0275 | +0.0042 (p=0.189) |
| SVM (RBF) [app] | 0.9778 ± 0.0099 | 0.9093 ± 0.0258 | 0.9216 ± 0.0213 | 0.8605 ± 0.0576 | 0.9580 ± 0.0299 | 0.9147 ± 0.0235 | +0.0050 (p=0.289) |
| gradient boosting [app] | 0.9757 ± 0.0079 | 0.9117 ± 0.0186 | 0.9184 ± 0.0178 | 0.8850 ± 0.0459 | 0.9384 ± 0.0345 | 0.9125 ± 0.0185 | +0.0029 (p=0.463) |
| **soft-voting ensemble (LR+SVM+RF) [app]** | 0.9783 ± 0.0070 | 0.9030 ± 0.0274 | 0.9150 ± 0.0227 | 0.8558 ± 0.0590 | 0.9501 ± 0.0296 | 0.9076 ± 0.0251 | +0.0054 (p=0.037) |
| logistic regression [select5] | 0.9882 ± 0.0076 | 0.9493 ± 0.0194 | 0.9543 ± 0.0177 | 0.9293 ± 0.0337 | 0.9692 ± 0.0209 | 0.9509 ± 0.0190 | +0.0153 (p=0.001) |
| soft-voting ensemble (LR+SVM+RF) [select5] | 0.9862 ± 0.0082 | 0.9408 ± 0.0220 | 0.9466 ± 0.0193 | 0.9180 ± 0.0473 | 0.9636 ± 0.0283 | 0.9426 ± 0.0207 | +0.0133 (p=0.002) |
| baseline config RF [all] | 0.9893 ± 0.0092 | 0.9520 ± 0.0275 | 0.9568 ± 0.0231 | 0.9332 ± 0.0572 | 0.9708 ± 0.0285 | 0.9535 ± 0.0250 | +0.0165 (p=0.000) |
| logistic regression [all] | 0.9951 ± 0.0043 | 0.9693 ± 0.0165 | 0.9754 ± 0.0127 | 0.9454 ± 0.0335 | 0.9933 ± 0.0082 | 0.9734 ± 0.0139 | +0.0222 (p=0.000) |
| **SVM (RBF) [all]** | 0.9955 ± 0.0047 | 0.9727 ± 0.0149 | 0.9754 ± 0.0139 | 0.9623 ± 0.0236 | 0.9832 ± 0.0156 | 0.9736 ± 0.0149 | +0.0227 (p=0.000) |
| gradient boosting [all] | 0.9923 ± 0.0055 | 0.9618 ± 0.0171 | 0.9670 ± 0.0147 | 0.9416 ± 0.0311 | 0.9821 ± 0.0143 | 0.9644 ± 0.0158 | +0.0194 (p=0.000) |
| soft-voting ensemble (LR+SVM+RF) [all] | 0.9950 ± 0.0045 | 0.9753 ± 0.0149 | 0.9779 ± 0.0125 | 0.9651 ± 0.0281 | 0.9854 ± 0.0118 | 0.9762 ± 0.0134 | +0.0221 (p=0.000) |

For reference, HealthCure reported 91.81% accuracy for this task on a single train/test split.

p-values: corrected resampled t-test (Nadeau & Bengio, 2003) on the per-fold paired differences.
Best model on the app's inputs (servable today): **soft-voting ensemble (LR+SVM+RF) [app]**.
Best model overall: **SVM (RBF) [all]**.
