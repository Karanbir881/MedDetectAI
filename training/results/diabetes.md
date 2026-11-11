## diabetes: baseline vs improved models

768 samples. **[app]** = the 8 inputs the web form collects (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age). 5x repeated stratified 5-fold cross-validation (seed 42), the same folds for every model. Mean ± standard deviation over folds.

| model | roc auc | balanced accuracy | accuracy | sensitivity | specificity | macro f1 | Δ ROC-AUC vs baseline (p) |
|---|---|---|---|---|---|---|---|
| baseline (HealthCure RF) [app] | 0.8262 ± 0.0323 | 0.7284 ± 0.0330 | 0.7648 ± 0.0303 | 0.6081 ± 0.0569 | 0.8488 ± 0.0379 | 0.7337 ± 0.0341 | — |
| baseline RF + missing-value imputation [app] | 0.8283 ± 0.0307 | 0.7283 ± 0.0254 | 0.7628 ± 0.0235 | 0.6142 ± 0.0514 | 0.8424 ± 0.0357 | 0.7326 ± 0.0259 | +0.0021 (p=0.771) |
| tuned random forest [app] | 0.8301 ± 0.0291 | 0.7192 ± 0.0285 | 0.7630 ± 0.0277 | 0.5739 ± 0.0581 | 0.8644 ± 0.0446 | 0.7267 ± 0.0305 | +0.0038 (p=0.594) |
| logistic regression [app] | 0.8365 ± 0.0303 | 0.7191 ± 0.0355 | 0.7693 ± 0.0307 | 0.5530 ± 0.0679 | 0.8852 ± 0.0385 | 0.7288 ± 0.0373 | +0.0103 (p=0.305) |
| SVM (RBF) [app] | 0.8358 ± 0.0283 | 0.7137 ± 0.0316 | 0.7620 ± 0.0277 | 0.5537 ± 0.0577 | 0.8736 ± 0.0334 | 0.7224 ± 0.0329 | +0.0095 (p=0.291) |
| gradient boosting [app] | 0.8360 ± 0.0305 | 0.7253 ± 0.0358 | 0.7622 ± 0.0314 | 0.6030 ± 0.0684 | 0.8476 ± 0.0402 | 0.7303 ± 0.0361 | +0.0098 (p=0.234) |
| **soft-voting ensemble (LR+SVM+RF) [app]** | 0.8416 ± 0.0297 | 0.7198 ± 0.0346 | 0.7661 ± 0.0297 | 0.5663 ± 0.0630 | 0.8732 ± 0.0340 | 0.7283 ± 0.0362 | +0.0154 (p=0.053) |

For reference, HealthCure reported 66.80% accuracy for this task on a single train/test split.

p-values: corrected resampled t-test (Nadeau & Bengio, 2003) on the per-fold paired differences.
Best model on the app's inputs (servable today): **soft-voting ensemble (LR+SVM+RF) [app]**.
Best model overall: **soft-voting ensemble (LR+SVM+RF) [app]**.
