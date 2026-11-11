"""Compare the HealthCure baseline with improved models on a clinical-data task.

Every model is evaluated on the *same* folds of repeated stratified k-fold
cross-validation, so the comparison is paired and fair. Hyperparameters of the
tuned models are chosen inside each training fold (nested CV), so the test
fold is never used for tuning.

The baseline is the configuration of the pretrained HealthCure model,
retrained fold by fold: we can't score the saved model fairly because we don't
know which rows it was trained on.

Examples
--------
  python -m training.experiment_tabular --task breastcancer
  python -m training.experiment_tabular --task diabetes --csv data/diabetes.csv
  python -m training.experiment_tabular --task heart --csv data/heart.csv
  # also fit the winning model on all the data and save it for the app
  python -m training.experiment_tabular --task breastcancer --save
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import zeros_to_nan  # noqa: E402
from training.train_tabular import load_table  # noqa: E402

# Accuracy HealthCure reported (single train/test split; Documents/accuracies.txt).
REPORTED = {"breastcancer": 0.9181, "diabetes": 0.668, "heart": 0.8696}

# Settings of the pretrained HealthCure models, read from the pickles in models/ and the
# notebooks: scikit-learn RandomForestClassifier defaults (100 trees, max_features="sqrt").
BASELINE = RandomForestClassifier(n_estimators=100, max_features="sqrt", n_jobs=-1)
BASELINE_SEED = {"diabetes": 10}  # the diabetes notebook fixed random_state=10


def models(seed, n_app, n_all, task="breastcancer"):
    """(name, feature set, estimator). Feature sets: "app" = the inputs the web form
    collects today; "all" = every column in the dataset; "select5" = the 5 best
    columns, chosen inside each training fold."""
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    tuned_rf = GridSearchCV(
        RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        {"max_features": [1, None], "min_samples_leaf": [1, 5]},
        scoring="roc_auc", cv=inner, n_jobs=-1)
    logreg = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
        scoring="roc_auc", cv=inner, n_jobs=-1)
    svm = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed)),
        {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]},
        scoring="roc_auc", cv=inner, n_jobs=-1)
    hgb = GridSearchCV(
        HistGradientBoostingClassifier(random_state=seed),
        {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 31]},
        scoring="roc_auc", cv=inner, n_jobs=-1)
    ensemble = VotingClassifier(
        [("lr", make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000))),
         ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed))),
         ("rf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))],
        voting="soft")
    select5 = lambda est: make_pipeline(SelectKBest(f_classif, k=5), clone(est))
    base_rf = clone(BASELINE).set_params(random_state=BASELINE_SEED.get(task, seed))
    out = [
        ("baseline (HealthCure RF)", "app", base_rf),
        ("tuned random forest", "app", tuned_rf),
        ("logistic regression", "app", logreg),
        ("SVM (RBF)", "app", svm),
        ("gradient boosting", "app", hgb),
        ("soft-voting ensemble (LR+SVM+RF)", "app", ensemble),
    ]
    if task == "diabetes":
        # 0 means "not measured" for glucose, blood pressure, skin, insulin and BMI:
        # mark as missing and impute the training-fold median. Baseline left as it was.
        impute = lambda est: make_pipeline(FunctionTransformer(zeros_to_nan),
                                           SimpleImputer(strategy="median"), clone(est))
        out = [out[0], ("baseline RF + missing-value imputation", "app", impute(base_rf))] + \
              [(n, fs, impute(e)) for n, fs, e in out[1:]]
    if task == "breastcancer" and n_all > n_app:
        out += [
            ("logistic regression", "select5", select5(logreg)),
            ("soft-voting ensemble (LR+SVM+RF)", "select5", select5(ensemble)),
            ("baseline config RF", "all", clone(BASELINE).set_params(random_state=seed)),
            ("logistic regression", "all", logreg),
            ("SVM (RBF)", "all", svm),
            ("gradient boosting", "all", hgb),
            ("soft-voting ensemble (LR+SVM+RF)", "all", ensemble),
        ]
    elif task == "heart" and n_all > n_app:
        out += [
            ("baseline config RF", "all", clone(base_rf)),
            ("logistic regression", "all", logreg),
            ("SVM (RBF)", "all", svm),
            ("gradient boosting", "all", hgb),
            ("soft-voting ensemble (LR+SVM+RF)", "all", ensemble),
        ]
    return [(f"{name} [{fs}]", fs, est) for name, fs, est in out]


def fold_scores(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, prob),
        "sensitivity": recall_score(y_true, pred, pos_label=1),
        "specificity": recall_score(y_true, pred, pos_label=0),
    }


def corrected_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated CV."""
    d = np.asarray(diffs)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["breastcancer", "diabetes", "heart"])
    ap.add_argument("--csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="training/results")
    ap.add_argument("--save", action="store_true", help="fit the best model on all data and save it")
    ap.add_argument("--save-features", choices=["app", "select5", "all"], default="app",
                    help="feature set of the model to save; anything but 'app' needs a matching web form")
    args = ap.parse_args()

    X_app, X_all, y, spec, all_names = load_table(args.task, args.csv)
    X_by = {"app": X_app, "select5": X_all, "all": X_all}
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    splits = list(cv.split(X_app, y))
    print(f"{args.task}: {len(y)} rows, app features {spec['features']}, {X_all.shape[1]} columns in total, "
          f"positives {y.mean():.1%}; {args.repeats}x{args.folds}-fold CV = {len(splits)} folds")

    per_fold = {}
    registry = models(args.seed, X_app.shape[1], X_all.shape[1], args.task)
    for name, fs, model in registry:
        X = X_by[fs]
        t0 = time.time()
        rows = []
        for tr, te in splits:
            m = clone(model).fit(X[tr], y[tr])
            rows.append(fold_scores(y[te], m.predict_proba(X[te])[:, 1]))
        per_fold[name] = rows
        auc = np.mean([r["roc_auc"] for r in rows])
        bal = np.mean([r["balanced_accuracy"] for r in rows])
        print(f"  {name:44s} ROC-AUC {auc:.4f}  bal.acc {bal:.4f}  ({time.time() - t0:.0f}s)")

    n_test = len(splits[0][1])
    n_train = len(y) - n_test
    base = "baseline (HealthCure RF) [app]"
    summary = {}
    for name, rows in per_fold.items():
        s = {}
        for metric in rows[0]:
            vals = np.array([r[metric] for r in rows])
            s[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
        if name != base:
            for metric in ("roc_auc", "balanced_accuracy"):
                diffs = [a[metric] - b[metric] for a, b in zip(rows, per_fold[base])]
                t, p = corrected_ttest(diffs, n_train, n_test)
                s[metric]["vs_baseline"] = {"mean_diff": float(np.mean(diffs)), "t": t, "p": p}
        summary[name] = s

    app_names = [n for n in summary if n.endswith("[app]") and n != base]
    best = max(app_names, key=lambda n: summary[n]["roc_auc"]["mean"])
    best_any = max((n for n in summary if n != base), key=lambda n: summary[n]["roc_auc"]["mean"])
    result = {"task": args.task, "n": int(len(y)), "features": spec["features"],
              "reported_accuracy_single_split": REPORTED.get(args.task),
              "all_columns": all_names, "best_model_any_features": best_any,
              "cv": f"{args.repeats}x repeated stratified {args.folds}-fold", "seed": args.seed,
              "best_model": best, "models": summary}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.task}.json"), "w") as f:
        json.dump(result, f, indent=2)
    table = markdown(result, base)
    with open(os.path.join(args.out, f"{args.task}.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print("\n" + table)

    if args.save:
        fs = args.save_features
        candidates = [n for n, f, _ in registry if f == fs and n != base]
        name = max(candidates, key=lambda n: summary[n]["roc_auc"]["mean"])
        est = {n: e for n, _, e in registry}[name]
        X_fit = X_by[fs]
        cols = [str(c) for c in (spec["features"] if fs == "app" else all_names)]
        if fs == "select5":
            sel = SelectKBest(f_classif, k=5).fit(X_all, y)
            cols = [str(c) for c, keep in zip(all_names, sel.get_support()) if keep]
            X_fit = X_all[:, sel.get_support()]
            est = est.steps[-1][1]  # drop the selector; the app sends just these 5 columns
        final = clone(est).fit(X_fit, y)
        out_dir = os.path.join("training", "outputs", args.task)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"best_model_{fs}.pkl")
        joblib.dump(final, path)
        with open(os.path.join(out_dir, f"best_model_{fs}.json"), "w") as f:
            json.dump({"model": name, "features_in_order": cols,
                       "cv_roc_auc": summary[name]["roc_auc"]["mean"]}, f, indent=2)
        print(f"saved {path}: {name}, inputs in order {cols}")

def markdown(r, base):
    metrics = ["roc_auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "macro_f1"]
    lines = [f"## {r['task']}: baseline vs improved models",
             "",
             f"{r['n']} samples. **[app]** = the {len(r['features'])} inputs the web form collects "
             f"({', '.join(r['features'])})"
             + (f"; **[select5]** = the 5 best of all {len(r['all_columns'])} columns, chosen inside each "
                f"training fold; **[all]** = all {len(r['all_columns'])} columns"
                if len(r["all_columns"]) > len(r["features"]) else "")
             + f". {r['cv']} cross-validation "
             f"(seed {r['seed']}), the same folds for every model. Mean ± standard deviation over folds.",
             "",
             "| model | " + " | ".join(m.replace("_", " ") for m in metrics) + " | Δ ROC-AUC vs baseline (p) |",
             "|---" * (len(metrics) + 2) + "|"]
    for name, s in r["models"].items():
        cells = [f"{s[m]['mean']:.4f} ± {s[m]['std']:.4f}" for m in metrics]
        vb = s["roc_auc"].get("vs_baseline")
        delta = "—" if vb is None else f"{vb['mean_diff']:+.4f} (p={vb['p']:.3f})"
        label = f"**{name}**" if name in (r["best_model"], r["best_model_any_features"]) else name
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {delta} |")
    if r.get("reported_accuracy_single_split"):
        lines += ["", f"For reference, HealthCure reported {r['reported_accuracy_single_split']:.2%} accuracy "
                  "for this task on a single train/test split."]
    lines += ["",
              "p-values: corrected resampled t-test (Nadeau & Bengio, 2003) on the per-fold paired differences.",
              f"Best model on the app's inputs (servable today): **{r['best_model']}**.",
              f"Best model overall: **{r['best_model_any_features']}**.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
