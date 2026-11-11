"""Results table from every metrics*.json under a folder (default training/outputs).

  python -m training.compare                      # all tasks
  python -m training.compare training/outputs/pneumonia
"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("training", "outputs")


def cell(m, k):
    if k not in m:
        return "-"
    s = f"{m[k]:.3f}"
    if f"{k}_ci95" in m:
        s += " ({:.3f}–{:.3f})".format(*m[f"{k}_ci95"])
    return s


print("| run | evaluation | n | accuracy (95% CI) | balanced acc. (95% CI) | macro F1 | ROC-AUC (95% CI) |")
print("|---|---|---|---|---|---|---|")
for path in sorted(glob.glob(os.path.join(root, "**", "metrics*.json"), recursive=True)):
    m = json.load(open(path))
    run = os.path.relpath(os.path.dirname(path), root).replace(os.sep, "/")
    auc = "roc_auc" if "roc_auc" in m else "roc_auc_ovr_macro"
    ev = m.get("evaluation", "")
    ev = "k-fold CV" if "CV" in ev else ("hold-out test" if ev else "")
    print(f"| {run} | {ev} | {m['n_test']} | {cell(m, 'accuracy')} | {cell(m, 'balanced_accuracy')} | "
          f"{cell(m, 'macro_f1')} | {cell(m, auc)} |")
