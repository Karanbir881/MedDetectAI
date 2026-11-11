"""Train and evaluate the imaging models (COVID-19, pneumonia, Alzheimer's, brain tumour).

Images are read with OpenCV and passed through preprocessing.model_input, the same
function the web app uses, so what is evaluated here is exactly what the app serves.
Class indices follow preprocessing.MODEL_CLASSES (alphabetical, like the HealthCure models).

Each directory holds one sub-folder per class (folder names are matched ignoring case).
Two evaluation modes:
  * hold-out:  --train-dir X --test-dir Y   (an official test set; best when it is large)
  * k-fold:    --train-dir X --cv 5         (all images in X; best for small datasets)
A validation set for early stopping comes from --val-dir if it has >= 50 images,
otherwise it is split from the training images.

Architectures (--arch):
  existing      evaluate the saved HealthCure model as is (may have seen the test images)
  baseline      HealthCure's architecture, re-initialised and retrained (the fair baseline)
  cnn           our CNN: BatchNorm, global pooling, augmentation, class weights
  mobilenetv2 | efficientnetb0 | vgg16   ImageNet transfer learning (+ --fine-tune)
"""
import argparse
import hashlib
import os
import random
import sys

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocessing import IMAGE_SIZES, MODEL_CLASSES, model_input  # noqa: E402
from training import metrics  # noqa: E402

SAVED_MODELS = {"covid": "covid.h5", "pneumonia": "pneumonia_model.h5",
                "alzheimer": "alzheimer_model.h5", "braintumor": "braintumor.h5"}
MODELS_DIR = os.path.join(ROOT, "models")
EXTS = {".png", ".jpg", ".jpeg"}


# ------------------------------------------------------------------- data
def list_images(root, classes):
    """Image paths and labels from root/<class>/ (class folder names matched ignoring case)."""
    if not os.path.isdir(root):
        raise SystemExit(f"folder not found: {root}")
    subdirs = {d.lower(): d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    paths, labels = [], []
    for idx, name in enumerate(classes):
        if name.lower() not in subdirs:
            raise SystemExit(f"no '{name}' folder in {root} (found {sorted(subdirs.values())})")
        folder = os.path.join(root, subdirs[name.lower()])
        found = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in EXTS)
        paths += found
        labels += [idx] * len(found)
    return paths, np.array(labels)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def dedupe(paths, y, exclude=()):
    """Drop byte-identical copies (keep the first) and any image whose hash is in `exclude`.

    An image that appears under two different classes is dropped entirely.
    Without this, copies of one scan can end up in both training and test.
    """
    hashes = [file_hash(p) for p in paths]
    labels_of = {}
    for h, label in zip(hashes, y):
        labels_of.setdefault(h, set()).add(int(label))
    seen, keep = set(exclude), []
    for i, h in enumerate(hashes):
        if h in seen or len(labels_of[h]) > 1:
            continue
        seen.add(h)
        keep.append(i)
    dropped = len(paths) - len(keep)
    if dropped:
        print(f"removed {dropped} duplicate images ({len(keep)} left)")
    return [paths[i] for i in keep], np.asarray(y)[keep], set(hashes)


def load_arrays(task, paths):
    """Model-ready arrays, stored as float16 to halve memory (Keras casts to float32)."""
    size = IMAGE_SIZES[task]
    X = np.empty((len(paths), size, size, 3), dtype=np.float16)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            raise SystemExit(f"unreadable image {p}")
        X[i] = model_input(task, img)[0]
    return X


# ----------------------------------------------------------------- models
def baseline_model(task):
    """HealthCure's architecture for `task`, freshly initialised, with its original optimizer."""
    if task == "braintumor":
        # braintumor.h5: VGG16 (ImageNet, frozen) -> Flatten -> Dropout(0.5) -> Dense(1, sigmoid)
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        base.trainable = False
        model = tf.keras.Sequential([base, tf.keras.layers.Flatten(), tf.keras.layers.Dropout(0.5),
                                     tf.keras.layers.Dense(1, activation="sigmoid")])
        return model, tf.keras.optimizers.Adam(1e-3)
    saved = tf.keras.models.load_model(os.path.join(MODELS_DIR, SAVED_MODELS[task]), compile=False)
    model = tf.keras.models.clone_model(saved)  # same layers, new random weights
    # optimizers in the notebooks: pneumonia RMSprop, the others Adam (lr 1e-3)
    opt = tf.keras.optimizers.RMSprop(1e-3) if task == "pneumonia" else tf.keras.optimizers.Adam(1e-3)
    return model, opt


def build_model(task, arch, n_classes, size):
    """Returns (model, backbone); backbone is None for models without a pretrained base."""
    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    if arch == "baseline":
        model, opt = baseline_model(task)
        model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        return model, None
    out_units, out_act = (1, "sigmoid") if n_classes == 2 else (n_classes, "softmax")
    inp = tf.keras.Input((size, size, 3))
    base = None
    if arch == "cnn":
        x = inp
        for filters in (32, 64, 128, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif arch == "vgg16":  # braintumor only: inputs are already VGG16-preprocessed
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        base.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D()(base(inp, training=False))
    elif arch == "mobilenetv2":  # inputs in [0, 1]; MobileNetV2 expects [-1, 1]
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                                 input_shape=(size, size, 3))
        base.trainable = False
        x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(base(x, training=False))
    elif arch == "efficientnetb0":  # inputs in [0, 1]; Keras EfficientNet rescales [0, 255] itself
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                    input_shape=(size, size, 3))
        base.trainable = False
        x = tf.keras.layers.Rescaling(255.0)(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(base(x, training=False))
    else:
        raise SystemExit(f"unknown arch {arch}")
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(out_units, activation=out_act)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    return model, base


def fine_tune(model, base, n_layers, loss):
    """Unfreeze the top n_layers of the backbone (BatchNorm layers stay frozen); low learning rate."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss, metrics=["accuracy"])


def probabilities(model, X, n_classes):
    p = model.predict(X, batch_size=32, verbose=0)
    return np.hstack([1 - p, p]) if n_classes == 2 else p


def train_one(args, X_tr, y_tr, X_va, y_va, n_classes, log_dir):
    """Train one model; returns it."""
    model, base = build_model(args.task, args.arch, n_classes, IMAGE_SIZES[args.task])
    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    os.makedirs(log_dir, exist_ok=True)
    cbs = lambda name: [tf.keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
                        tf.keras.callbacks.CSVLogger(os.path.join(log_dir, name))]
    augment = tf.keras.Sequential([tf.keras.layers.RandomRotation(0.05),
                                   tf.keras.layers.RandomZoom(0.1),
                                   tf.keras.layers.RandomTranslation(0.05, 0.05)])
    ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(len(y_tr), seed=args.seed)
    ds = ds.batch(args.batch_size).map(lambda x, y: (tf.cast(x, tf.float32), y))
    if not args.no_augment:
        ds = ds.map(lambda x, y: (augment(x, training=True), y))
    ds = ds.prefetch(2)
    val = (X_va.astype("float32"), y_va)
    cw = None
    if not args.no_class_weights:
        w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_tr)
        cw = dict(enumerate(w))
    model.fit(ds, validation_data=val, epochs=args.epochs, callbacks=cbs("history.csv"),
              class_weight=cw, verbose=2)
    if args.fine_tune:
        if base is None:
            raise SystemExit("--fine-tune needs a pretrained backbone (mobilenetv2, efficientnetb0, vgg16)")
        fine_tune(model, base, args.unfreeze, loss)
        model.fit(ds, validation_data=val, epochs=args.fine_tune, callbacks=cbs("history_finetune.csv"),
                  class_weight=cw, verbose=2)
    return model


def split_val(args, paths, y):
    """Validation set: --val-dir when it has >= 50 images, else a stratified split of `paths`."""
    if args.val_dir:
        vp, vy = list_images(args.val_dir, MODEL_CLASSES[args.task])
        if len(vy) >= 50:
            return paths, y, vp, vy
        print(f"--val-dir has only {len(vy)} images; splitting validation from training instead")
    tp, vp, ty, vy = train_test_split(paths, y, test_size=args.val_size, stratify=y, random_state=args.seed)
    return tp, ty, vp, vy


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(MODEL_CLASSES))
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir")
    ap.add_argument("--val-dir")
    ap.add_argument("--cv", type=int, default=0, metavar="K", help="k-fold CV over --train-dir instead of a test set")
    ap.add_argument("--arch", default="cnn",
                    choices=["existing", "baseline", "cnn", "vgg16", "mobilenetv2", "efficientnetb0"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--fine-tune", type=int, default=0, metavar="EPOCHS")
    ap.add_argument("--unfreeze", type=int, default=30, help="backbone layers to unfreeze when fine-tuning")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--out", default=os.path.join("training", "outputs"))
    ap.add_argument("--save-model", action="store_true", help="hold-out mode: also save the trained model")
    args = ap.parse_args()

    if args.task == "braintumor" and args.arch in ("cnn", "mobilenetv2", "efficientnetb0"):
        raise SystemExit("braintumor inputs are VGG16-preprocessed: use --arch vgg16, baseline or existing")
    if args.task != "braintumor" and args.arch == "vgg16":
        raise SystemExit("--arch vgg16 is only for braintumor")
    if args.arch == "baseline":  # replicate the original training: no augmentation, no class weights
        args.no_augment = args.no_class_weights = True
    if not args.cv and not args.test_dir:
        raise SystemExit("give --test-dir or --cv K")
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)

    classes = MODEL_CLASSES[args.task]
    n_classes = len(classes)
    run = args.arch + (f"_ft{args.fine_tune}" if args.fine_tune else "")
    out_dir = os.path.join(args.out, args.task, run)
    os.makedirs(out_dir, exist_ok=True)
    paths, y = list_images(args.train_dir, classes)
    if args.test_dir:
        te_paths, te_y = list_images(args.test_dir, classes)
        te_paths, te_y, te_hashes = dedupe(te_paths, te_y)
        paths, y, _ = dedupe(paths, y, exclude=te_hashes)  # no test image may appear in training
    else:
        paths, y, _ = dedupe(paths, y)
    settings = dict(arch=args.arch, fine_tune_epochs=args.fine_tune, augment=not args.no_augment,
                    class_weights=not args.no_class_weights, seed=args.seed,
                    train_dir=args.train_dir, test_dir=args.test_dir)
    existing = None
    if args.arch == "existing":
        existing = tf.keras.models.load_model(os.path.join(MODELS_DIR, SAVED_MODELS[args.task]), compile=False)

    if args.cv:
        print(f"{args.task}: {len(y)} images {np.bincount(y).tolist()}, {args.cv}-fold CV, arch {run}")
        folds = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed).split(paths, y)
        X_all = load_arrays(args.task, paths)
        oof = np.zeros((len(y), n_classes))
        per_fold = []
        for k, (tr, te) in enumerate(folds):
            if existing is not None:
                model = existing
            else:
                tp, ty, vp, vy = split_val(args, [paths[i] for i in tr], y[tr])
                idx = {p: i for i, p in enumerate(paths)}
                X_tr = X_all[[idx[p] for p in tp]]
                X_va = (X_all[[idx[p] for p in vp]] if all(p in idx for p in vp)
                        else load_arrays(args.task, vp))
                model = train_one(args, X_tr, ty, X_va, vy, n_classes, os.path.join(out_dir, f"fold{k}"))
            oof[te] = probabilities(model, X_all[te].astype("float32"), n_classes)
            fm = metrics.evaluate(y[te], oof[te], classes, ci=False)
            per_fold.append(fm)
            print(f"  fold {k}: acc {fm['accuracy']:.4f}  bal.acc {fm['balanced_accuracy']:.4f}")
            if existing is None:
                tf.keras.backend.clear_session()
        m = metrics.evaluate(y, oof, classes)  # pooled out-of-fold predictions, with bootstrap CIs
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "roc_auc_ovr_macro"):
            vals = [f[key] for f in per_fold if key in f]
            if vals:
                m[f"{key}_fold_mean"], m[f"{key}_fold_std"] = float(np.mean(vals)), float(np.std(vals, ddof=1))
        m.update(settings, evaluation=f"{args.cv}-fold stratified CV, pooled out-of-fold predictions")
    else:
        print(f"{args.task}: train {len(y)} {np.bincount(y).tolist()}, test {len(te_y)} "
              f"{np.bincount(te_y).tolist()}, arch {run}")
        X_te = load_arrays(args.task, te_paths).astype("float32")
        if existing is not None:
            model = existing
        else:
            tp, ty, vp, vy = split_val(args, paths, y)
            model = train_one(args, load_arrays(args.task, tp), ty, load_arrays(args.task, vp), vy,
                              n_classes, out_dir)
        m = metrics.evaluate(te_y, probabilities(model, X_te, n_classes), classes)
        m.update(settings, evaluation="official hold-out test set")
        if args.save_model and existing is None:
            model.save(os.path.join(out_dir, "model.h5"))
            print(f"saved {out_dir}/model.h5 (copy over models/{SAVED_MODELS[args.task]} to serve it)")
    if args.arch == "existing":
        m["note"] = "saved HealthCure model; it may have been trained on some of these test images"
    metrics.save(m, out_dir)


if __name__ == "__main__":
    main()
