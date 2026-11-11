"""Evaluation metrics shared by the image and tabular training scripts."""
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

CI_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "roc_auc_ovr_macro")


def _scores(y_true, y_prob, labels):
    y_pred = y_prob.argmax(axis=1)
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)}
    try:
        if len(labels) == 2:
            out["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            out["roc_auc_ovr_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                                     average="macro", labels=labels)
    except ValueError:  # a class is missing from y_true
        pass
    return out


def evaluate(y_true, y_prob, class_names, ci=True, n_boot=1000, seed=0):
    """Metrics for class probabilities y_prob (n_samples, n_classes), with bootstrap 95% CIs."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    labels = list(range(len(class_names)))
    y_pred = y_prob.argmax(axis=1)
    out = {"n_test": int(len(y_true)), **_scores(y_true, y_prob, labels),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
           "class_names": list(class_names),
           "per_class": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                              output_dict=True, zero_division=0)}
    if ci:
        rng = np.random.default_rng(seed)
        boots = {k: [] for k in CI_METRICS}
        for _ in range(n_boot):
            i = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[i])) < len(labels):
                continue
            for k, v in _scores(y_true[i], y_prob[i], labels).items():
                boots[k].append(v)
        for k, vals in boots.items():
            if vals and k in out:
                out[f"{k}_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def save(metrics, out_dir, name="metrics.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print_summary(metrics)
    print(f"saved {path}")
    return path


def print_summary(m):
    parts = []
    for k in ("n_test",) + CI_METRICS:
        if k not in m:
            continue
        v = m[k]
        s = f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        if f"{k}_ci95" in m:
            lo, hi = m[f"{k}_ci95"]
            s += f" [{lo:.3f}, {hi:.3f}]"
        parts.append(s)
    print("  ".join(parts))
    print("confusion matrix (rows = true, cols = predicted):", m["class_names"])
    for row in m["confusion_matrix"]:
        print("   ", row)
