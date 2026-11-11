# MedDetect AI

MedDetect AI is a Flask web app that runs seven disease detectors: four on medical
images (COVID-19 and pneumonia from chest X-rays, brain tumour and Alzheimer's
disease from MRI) and three on clinical measurements (diabetes, heart disease and
breast cancer).

> **Research prototype, not a medical device.** Results are not a diagnosis and
> must not be used for clinical decisions.

## Credit

This project is built on **HealthCure** by **Abhishek Sharma**
([project page](https://machinelearningprojects.net/healthcure-medical-project/),
2021). HealthCure provided the original Flask app, the HTML templates and the
seven pretrained models in `models/`.

## What we changed

**Bug fixes**
- Alzheimer's results were never shown: the original always rendered "NonDemented". The app now shows the model's predicted class, in the model's real (alphabetical) class order: Keras numbers class folders alphabetically, and the label list in the original code did not match it.
- Images were passed to the models in OpenCV's BGR channel order and resized differently from training. They are now converted to RGB and resized exactly as Keras did during training.
- Clinical form values were passed to the models as text; they are now validated and converted to numbers, with an error message when a field is invalid.
- The brain-tumour crop crashed on images where no outline is found; it now falls back to the full image.
- Invalid uploads redirected to a POST-only URL (HTTP 405); they now return to the form with an explanation.
- `requirements` pinned scikit-learn 0.24.2, but the saved models were pickled with 1.0.2; the version is now correct.

**New interface**
- Redesigned from scratch: our own layout, stylesheet and copy, with no Bootstrap template or original images. One shared layout, form and result template replace the 15 copied pages.
- Every result shows the model's probability for each outcome (all four stages for Alzheimer's), the input or scan it was based on, and the detector's known limits.
- Clinical forms explain each measurement and its unit; image upload has drag-and-drop with a preview.
- Patient name, age and sex are optional and only shown on the result page; where age is a model input (diabetes, heart disease) it is asked for with the measurements.

**Privacy and safety**
- Removed the phone and email fields, which were collected but never used.
- Uploads are saved under random names, so users can't overwrite each other's images or guess file names.
- Debug mode and a hard-coded secret key are no longer the defaults; uploads are size-limited.
- Every result page now carries a not-a-diagnosis notice.

**Code**
- Shared preprocessing (`preprocessing.py`) so the app and the training code feed models identical inputs.
- Removed seven near-identical copies of the request handling.

**Training and evaluation (new)**
- `training/` contains our own training and evaluation pipeline for all seven tasks, with stratified splits, class weighting, cross-validation for the tabular models, and full metrics (accuracy, balanced accuracy, macro F1, ROC-AUC, confusion matrix).
- Every pretrained HealthCure model can be evaluated on the same test split as a baseline (`--evaluate`), and its architecture or settings retrained under the same split as a fair baseline.
- Experiments comparing the baseline with improved models; see **Experiments** below.

## Running the app

Requires Python 3.9 (TensorFlow 2.12 supports Python 3.8–3.11 only).

```bash
conda create -n meddetect python=3.9 -y
conda activate meddetect
python --version                      # should print Python 3.9.x
python -m pip install -r requirements.txt
python app.py                         # then open http://127.0.0.1:5000
```

Use `python -m pip` rather than `pip`, so packages go into the active environment; if
`conda create` is run without `python=3.9`, pip falls back to another Python and
TensorFlow 2.12 can't be found. Set `MEDDETECT_SECRET_KEY` in the environment for any
shared deployment.

## Experiments: baseline vs improved models

**Tabular tasks** (`training/experiment_tabular.py`) compare the HealthCure model's
configuration, retrained fold by fold, with tuned random forest, logistic regression,
SVM, gradient boosting and a soft-voting ensemble. Every model is scored on the same
5x repeated stratified 5-fold CV, tuning happens inside each training fold, and
differences are tested with the corrected resampled t-test.

```bash
python -m training.experiment_tabular --task breastcancer
python -m training.experiment_tabular --task diabetes --csv data/diabetes.csv
python -m training.experiment_tabular --task heart --csv data/heart.csv
```

**Imaging tasks** (`training/run_image_experiments.py`, or double-click `run_experiments.bat`)
run, on the same images: the saved HealthCure model as is, HealthCure's architecture
retrained (the fair baseline), our CNN with BatchNorm, augmentation and class weights,
and MobileNetV2 / EfficientNetB0 transfer learning with fine-tuning (VGG16 with and
without fine-tuning for brain tumour). Pneumonia and Alzheimer's use their official test
sets; COVID (392 images) and brain tumour (228 unique images) use 5-fold CV over all
images, because their test folders hold only 30 and 10 images. Exact duplicate images
are removed first (the brain tumour folder has 25), and every result has a bootstrap
95% confidence interval.

```bash
python -m training.run_image_experiments --data "C:\Users\Karan\Downloads\meddetect\meddetectAI"
```

### Results: clinical-data models

Full tables with every model, metric and p-value: `training/results/`. All results use
5x repeated stratified 5-fold CV with the same folds for every model; "baseline" is
HealthCure's random forest configuration retrained on each fold.

| Task | Model | Inputs | Accuracy | ROC-AUC | Balanced acc. | Sensitivity | Δ ROC-AUC (p) |
|---|---|---|---|---|---|---|---|
| Breast cancer | HealthCure baseline | form's 5 | 91.9% | 0.973 | 0.912 | 0.884 | — |
| | Logistic regression | best 5 (chosen in-fold) | 95.4% | 0.988 | 0.949 | 0.929 | +0.015 (0.001) |
| | SVM (RBF) | all 30 | 97.5% | **0.996** | 0.973 | 0.962 | +0.023 (<0.001) |
| | Soft-voting ensemble | all 30 | **97.8%** | 0.995 | **0.975** | **0.965** | +0.022 (<0.001) |
| Diabetes | HealthCure baseline | form's 8 | 76.5% | 0.826 | **0.728** | **0.608** | — |
| | Soft-voting ensemble + missing-value handling | form's 8 | **76.6%** | **0.842** | 0.720 | 0.566 | +0.015 (0.053) |
| Heart disease | HealthCure baseline | form's 7 | 80.5% | 0.895 | 0.801 | 0.843 | — |
| | Soft-voting ensemble | form's 7 | 82.6% | 0.905 | 0.821 | 0.880 | +0.010 (0.154) |
| | Soft-voting ensemble | all 13 | **83.9%** | **0.908** | **0.834** | **0.893** | +0.013 (0.148) |

Accuracy is the mean over the 25 folds; standard deviations (about ±1–4 points) are in
the full tables. Balanced accuracy and ROC-AUC are the fairer measures for diabetes,
where only 35% of patients are positive.

What this shows:
- **Breast cancer:** the only clear, statistically significant gain, and it comes from the
  inputs, not the algorithm. On the form's 5 inputs no model beats the baseline; with
  better inputs ROC-AUC rises from 0.973 to 0.988 (same form length) or 0.996 (all 30).
- **Diabetes:** better algorithms and treating unmeasured zeros as missing give a small,
  borderline gain in ROC-AUC (p = 0.053), but at the default 0.5 threshold they catch
  *fewer* diabetics than the baseline (sensitivity 0.57 vs 0.61). Diabetes needs a lower
  decision threshold or class weighting before any model is clinically useful.
- **Heart disease:** about +0.01 ROC-AUC and +3 to +5 points of sensitivity, but with 300
  patients the gain is not statistically significant (p ≈ 0.15).
- HealthCure's reported single-split accuracies (Documents/accuracies.txt: diabetes 66.8%,
  heart 86.96%, breast cancer 91.81%) are not comparable to these numbers: a single
  small test split varies by several points from one random split to the next.

Caveat: these are cross-validated results on one public dataset; an external test set
would be needed before claiming the model generalises.

### Results: imaging models

Not run yet. `run_experiments.bat` writes them to `training/results/images.md`. Until
then, the only figures are HealthCure's own (pneumonia 83.2%, Alzheimer's 73.5%,
COVID-19 93%, brain tumour 100%), which were not checked independently; the brain
tumour figure came from a test set of 10 images.

## Models currently served

| Task | Input | Model (from HealthCure) |
|---|---|---|
| COVID-19 | chest X-ray, 224 px | custom CNN |
| Pneumonia | chest X-ray, 150 px | custom CNN |
| Alzheimer's (4 classes) | MRI, 176 px | custom CNN |
| Brain tumour | MRI, 224 px, auto-cropped | VGG16 transfer learning |
| Diabetes | 8 clinical values | random forest |
| Heart disease | 7 clinical values | random forest |
| Breast cancer | 5 tumour measurements | random forest |

## Authors

MedDetect AI : Karanbir Singh 

