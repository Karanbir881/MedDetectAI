"""Train and evaluate the clinical-data models (diabetes, heart disease, breast cancer).

Features are taken in exactly the order app.py sends them, so a trained model
can replace the one in models/ directly.

Examples
--------
  # breast cancer: uses the Wisconsin Diagnostic dataset bundled with scikit-learn
  python -m training.train_tabular --task breastcancer
  # diabetes: Pima Indians Diabetes CSV
  python -m training.train_tabular --task diabetes --csv data/diabetes.csv
  # heart: UCI Heart Disease (Cleveland) CSV
  python -m training.train_tabular --task heart --csv data/heart.csv
  # evaluate the pretrained HealthCure model on the same test split (baseline)
  python -m training.train_tabular --task breastcancer --evaluate models/cancer_model.pkl
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training import metrics

# For each task: the CSV columns in the order app.py passes them, the label column,
# and class names (0, 1). Column names follow the CSVs in the HealthCure dataset folder.
TASKS = {
    "diabetes": dict(
        features=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                  "DiabetesPedigreeFunction", "Age"],
        label="Outcome", classes=["negative", "positive"]),
    "heart": dict(
        # app.py form fields: nmv=ca, tcp=cp, eia=exang, thal, op=oldpeak, mhra=thalach, age
        features=["ca", "cp", "exang", "thal", "oldpeak", "thalach", "age"],
        label="target", classes=["negative", "positive"]),
    "breastcancer": dict(
        # Kaggle/UCI data.csv names; label from diagnosis (M = malignant = 1)
        features=["concave points_mean", "area_mean", "radius_mean", "perimeter_mean",
                  "concavity_mean"],
        label="malignant", classes=["benign", "malignant"]),
}
NOT_FEATURES = {"id", "diagnosis", "Unnamed: 32"}


def load_table(task, csv):
    """Returns X_app, X_all, y, spec, all_column_names.

    X_app holds the columns the web form collects, X_all every feature column in the CSV.
    Cleaning matches the HealthCure notebooks, plus duplicate removal.
    """
    spec = TASKS[task]
    if csv is None:
        if task != "breastcancer":
            raise SystemExit(f"--csv is required for {task}; see training/DATASETS.md")
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        df = d.frame.drop(columns="target")
        df.columns = [c.replace("mean ", "").replace(" ", "_") + "_mean" if c.startswith("mean ")
                      else c.replace("worst ", "") .replace(" ", "_") + "_worst" if c.startswith("worst ")
                      else c.replace(" error", "").replace(" ", "_") + "_se" for c in df.columns]
        df.columns = [c.replace("concave_points", "concave points") for c in df.columns]
        df["malignant"] = 1 - d.target  # sklearn: 0 = malignant
    else:
        df = pd.read_csv(csv, encoding="utf-8-sig")
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
        if task == "breastcancer":
            df["malignant"] = (df["diagnosis"] == "M").astype(int)
        if task == "heart":
            if "thal" in df:
                df = df[df["thal"] != 0]  # as in the HealthCure notebook: thal=0 is not a valid code
            if df[spec["label"]].nunique() > 2:
                df[spec["label"]] = (df[spec["label"]] > 0).astype(int)  # UCI 0-4 severity -> presence
    all_cols = [c for c in df.columns if c not in NOT_FEATURES and c != spec["label"]]
    missing = [c for c in spec["features"] + [spec["label"]] if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns {missing}; found {list(df.columns)}")
    df = df[all_cols + [spec["label"]]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(df)
    df = df.drop_duplicates()  # duplicated rows would leak between train and test
    if len(df) < n:
        print(f"dropped {n - len(df)} duplicate rows")
    y = df[spec["label"]].to_numpy(int)
    return (df[spec["features"]].to_numpy(float), df[all_cols].to_numpy(float), y, spec, all_cols)


def load(task, csv):
    X_app, _, y, spec = load_table(task, csv)[:4]
    return X_app, y, spec


def candidates(seed):
    return {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }


def load_model(path):
    """joblib.load also reads plain pickles (diabetes.sav, heart_disease.pickle.dat)."""
    try:
        return joblib.load(path)
    except Exception as err:
        import sklearn
        raise SystemExit(f"could not load {path} with scikit-learn {sklearn.__version__}: {err}\n"
                         "The HealthCure models need scikit-learn 1.0.2 (pip install -r requirements.txt).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", default="training/outputs")
    ap.add_argument("--evaluate", metavar="MODEL", help="only evaluate an existing model file")
    args = ap.parse_args()

    X, y, spec = load(args.task, args.csv)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, stratify=y,
                                              random_state=args.seed)
    out_dir = os.path.join(args.out, args.task)
    print(f"{args.task}: {len(y)} rows, train {len(y_tr)}, test {len(y_te)}, "
          f"positives {y.mean():.1%}")

    if args.evaluate:
        model = load_model(args.evaluate)
        m = metrics.evaluate(y_te, model.predict_proba(X_te), spec["classes"])
        m.update(model=args.evaluate, seed=args.seed,
                 note="test split may overlap the data this model was trained on")
        metrics.save(m, out_dir, "metrics_existing_model.json")
        return

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    results, best_name, best_score = {}, None, -1.0
    for name, model in candidates(args.seed).items():
        scores = cross_val_score(clone(model), X_tr, y_tr, cv=cv, scoring="roc_auc")
        results[name] = {"cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()}
        print(f"  {name:20s} 5-fold CV ROC-AUC {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_name, best_score = name, scores.mean()

    model = clone(candidates(args.seed)[best_name]).fit(X_tr, y_tr)
    m = metrics.evaluate(y_te, model.predict_proba(X_te), spec["classes"])
    m.update(model=best_name, features=spec["features"], seed=args.seed, cross_validation=results)
    if best_name == "random_forest":
        m["feature_importance"] = dict(zip(spec["features"], model.feature_importances_.tolist()))
    metrics.save(m, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    print(f"best model: {best_name}; saved {out_dir}/model.pkl")


if __name__ == "__main__":
    main()
