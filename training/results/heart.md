## heart: baseline vs improved models

300 samples. **[app]** = the 7 inputs the web form collects (ca, cp, exang, thal, oldpeak, thalach, age); **[select5]** = the 5 best of all 13 columns, chosen inside each training fold; **[all]** = all 13 columns. 5x repeated stratified 5-fold cross-validation (seed 42), the same folds for every model. Mean ± standard deviation over folds.

| model | roc auc | balanced accuracy | accuracy | sensitivity | specificity | macro f1 | Δ ROC-AUC vs baseline (p) |
|---|---|---|---|---|---|---|---|
| baseline (HealthCure RF) [app] | 0.8946 ± 0.0333 | 0.8013 ± 0.0382 | 0.8047 ± 0.0365 | 0.8433 ± 0.0600 | 0.7593 ± 0.0807 | 0.8015 ± 0.0377 | — |
| tuned random forest [app] | 0.9045 ± 0.0311 | 0.8229 ± 0.0406 | 0.8280 ± 0.0396 | 0.8824 ± 0.0568 | 0.7634 ± 0.0698 | 0.8245 ± 0.0406 | +0.0099 (p=0.259) |
| logistic regression [app] | 0.8883 ± 0.0358 | 0.8209 ± 0.0421 | 0.8280 ± 0.0396 | 0.9056 ± 0.0541 | 0.7361 ± 0.0935 | 0.8227 ± 0.0421 | -0.0063 (p=0.475) |
| SVM (RBF) [app] | 0.8960 ± 0.0372 | 0.8253 ± 0.0393 | 0.8300 ± 0.0382 | 0.8798 ± 0.0528 | 0.7709 ± 0.0718 | 0.8267 ± 0.0394 | +0.0015 (p=0.876) |
| gradient boosting [app] | 0.8985 ± 0.0340 | 0.8109 ± 0.0437 | 0.8160 ± 0.0423 | 0.8726 ± 0.0584 | 0.7492 ± 0.0821 | 0.8120 ± 0.0440 | +0.0040 (p=0.614) |
| **soft-voting ensemble (LR+SVM+RF) [app]** | 0.9048 ± 0.0343 | 0.8211 ± 0.0387 | 0.8260 ± 0.0363 | 0.8800 ± 0.0485 | 0.7622 ± 0.0809 | 0.8223 ± 0.0379 | +0.0102 (p=0.154) |
| baseline config RF [all] | 0.9027 ± 0.0355 | 0.8236 ± 0.0365 | 0.8280 ± 0.0353 | 0.8749 ± 0.0561 | 0.7722 ± 0.0756 | 0.8247 ± 0.0361 | +0.0082 (p=0.339) |
| logistic regression [all] | 0.9028 ± 0.0360 | 0.8225 ± 0.0407 | 0.8307 ± 0.0390 | 0.9179 ± 0.0508 | 0.7271 ± 0.0787 | 0.8251 ± 0.0412 | +0.0082 (p=0.362) |
| SVM (RBF) [all] | 0.8960 ± 0.0359 | 0.8133 ± 0.0424 | 0.8153 ± 0.0438 | 0.8394 ± 0.0908 | 0.7872 ± 0.0793 | 0.8127 ± 0.0437 | +0.0014 (p=0.904) |
| gradient boosting [all] | 0.8981 ± 0.0361 | 0.8174 ± 0.0359 | 0.8220 ± 0.0343 | 0.8726 ± 0.0591 | 0.7622 ± 0.0827 | 0.8183 ± 0.0360 | +0.0035 (p=0.770) |
| **soft-voting ensemble (LR+SVM+RF) [all]** | 0.9079 ± 0.0409 | 0.8336 ± 0.0391 | 0.8387 ± 0.0387 | 0.8933 ± 0.0592 | 0.7739 ± 0.0688 | 0.8354 ± 0.0394 | +0.0134 (p=0.148) |

For reference, HealthCure reported 86.96% accuracy for this task on a single train/test split.

p-values: corrected resampled t-test (Nadeau & Bengio, 2003) on the per-fold paired differences.
Best model on the app's inputs (servable today): **soft-voting ensemble (LR+SVM+RF) [app]**.
Best model overall: **soft-voting ensemble (LR+SVM+RF) [all]**.
