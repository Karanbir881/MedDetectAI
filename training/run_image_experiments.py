"""Run the baseline-vs-improved comparison for the imaging tasks.

  python -m training.run_image_experiments --data "C:\\Users\\Karan\\Downloads\\meddetect\\meddetectAI"
  python -m training.run_image_experiments --data ... --tasks braintumor covid --quick

  improved   our CNN, MobileNetV2 + fine-tuning, EfficientNetB0 + fine-tuning
             (VGG16 with and without fine-tuning for brain tumour)
Small datasets (COVID, brain tumour) use 5-fold CV over all images; pneumonia and
Alzheimer's use their official test sets. Results: training/outputs/<task>/, and a
combined table in training/results/images.md.
"""
import argparse
import os
import subprocess
import sys

LAYOUT = {
    # task: (train dir, test dir or None for CV), relative to --data
    "braintumor": (os.path.join("Brain Tumor Detection", "brain_tumor_dataset"), None),
    "covid": ("Covid Detection", None),  # its COVID/ and NORMAL/ folders hold all 392 images
    "pneumonia": (os.path.join("Pneumonia Detection", "chest_xray", "train"),
                  os.path.join("Pneumonia Detection", "chest_xray", "test")),
    "alzheimer": (os.path.join("Alzheimer Detection", "Alzheimer_Dataset", "train"),
                  os.path.join("Alzheimer Detection", "Alzheimer_Dataset", "test")),
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True)
ap.add_argument("--tasks", nargs="+", default=list(LAYOUT), choices=list(LAYOUT))
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--fine-tune", type=int, default=10)
ap.add_argument("--quick", action="store_true", help="skip EfficientNetB0 (the slowest on a CPU)")
args = ap.parse_args()

failed = []
for task in args.tasks:
    train, test = LAYOUT[task]
    io = ["--train-dir", os.path.join(args.data, train)]
    io += ["--cv", "5"] if test is None else ["--test-dir", os.path.join(args.data, test)]
    ft = ["--fine-tune", str(args.fine_tune)]
    if task == "braintumor":
        archs = [["existing"], ["baseline"], ["vgg16"], ["vgg16", *ft, "--unfreeze", "4"]]
    else:
        archs = [["existing"], ["baseline"], ["cnn"], ["mobilenetv2", *ft]]
        if not args.quick:
            archs.append(["efficientnetb0", *ft])
    for arch in archs:
        cmd = [sys.executable, "-m", "training.train_image", "--task", task, *io,
               "--epochs", str(args.epochs), "--arch", *arch]
        if test is not None and arch[0] != "existing":
            cmd.append("--save-model")
        print("\n$", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(f"{task} {' '.join(arch)}")

os.makedirs(os.path.join("training", "results"), exist_ok=True)
table = subprocess.run([sys.executable, "-m", "training.compare", os.path.join("training", "outputs")],
                       capture_output=True, text=True).stdout
with open(os.path.join("training", "results", "images.md"), "w", encoding="utf-8") as f:
    f.write("## Imaging tasks: baseline vs improved models\n\n" + table)
print("\n" + table)
if failed:
    print("FAILED runs:", *failed, sep="\n  ")
