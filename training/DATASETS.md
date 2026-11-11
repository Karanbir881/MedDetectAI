# Datasets

None of the data is included in this repository. Download each dataset yourself,
read and follow its license and citation requirements, and cite it in the paper.
The pretrained models in `models/` came with HealthCure; we do not know exactly
which images or splits they were trained on, so their scores on any test set
should be reported as a baseline with that caveat.

| Task | Suggested dataset | Layout expected by the scripts |
|---|---|---|
| Breast cancer | Wisconsin Diagnostic Breast Cancer (UCI) — bundled with scikit-learn, no download needed | built in |
| Diabetes | Pima Indians Diabetes (UCI / Kaggle `uciml/pima-indians-diabetes-database`) | `data/diabetes.csv` with columns `Pregnancies … Age, Outcome` |
| Heart disease | UCI Heart Disease, Cleveland subset | `data/heart.csv` with columns `age, cp, thalach, exang, oldpeak, ca, thal, target` |
| Pneumonia | Chest X-Ray Images (Pneumonia), Kermany et al. (Kaggle `paultimothymooney/chest-xray-pneumonia`) | `data/pneumonia/{train,test}/{normal,pneumonia}/` |
| COVID-19 | COVID-19 Radiography Database (Kaggle `tawsifurrahman/covid19-radiography-database`) | `data/covid/train/{covid,normal}/` |
| Brain tumour | Brain MRI Images for Brain Tumor Detection (Kaggle `navoneel/...`) or Br35H | `data/braintumor/train/{no,yes}/` |
| Alzheimer's | Alzheimer's Dataset, 4 classes (Kaggle `tourist55/alzheimers-dataset-4-class-of-images`) | `data/alzheimer/{train,test}/{NonDemented,VeryMildDemented,MildDemented,ModerateDemented}/` |

Pitfalls worth checking before you trust a number:

- **Duplicates.** The popular Kaggle `heart.csv` (1,025 rows) is mostly duplicated rows; drop duplicates first or the test set leaks into training.
- **Patient-level splits.** Several imaging datasets contain multiple images per patient. Use the official train/test split when one exists, so the same patient is never in both.
- **Shortcut learning.** COVID datasets mix images from different hospitals and sources; a model can learn the source instead of the disease. Report this as a limitation or test on an external dataset.
- **Class imbalance.** Report balanced accuracy, macro F1 and ROC-AUC, not just accuracy.
