"""
MedDetect AI - Flask web app for seven disease detections.
This is a research prototype, not a medical device.
"""
import os
import pickle
import uuid

import cv2
import joblib
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from tensorflow.keras.models import load_model

from preprocessing import MODEL_CLASSES, model_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ------------------------------------------------------------------ tasks
# Everything the pages show about each detector. `fields` are the clinical inputs,
# in the order the model expects them; the names are the form field names.
F = lambda name, label, help, **kw: dict(name=name, label=label, help=help, **kw)
TASKS = {
    "covid": dict(
        title="COVID-19", kind="image", form="covid", post="resultc",
        input="Chest X-ray", model="Convolutional neural network",
        about="Looks for patterns of COVID-19 pneumonia on a frontal chest X-ray.",
        limits="Trained on a few hundred images from public collections that mix hospitals and "
               "scanners, so it can pick up the image source rather than the disease.",
        upload_hint="A frontal (PA or AP) chest X-ray, PNG or JPG.",
        labels={"COVID": "Signs of COVID-19", "NORMAL": "No signs of COVID-19"},
        positive={"COVID"}),
    "pneumonia": dict(
        title="Pneumonia", kind="image", form="pneumonia", post="resultp",
        input="Chest X-ray", model="Convolutional neural network",
        about="Looks for signs of pneumonia on a frontal chest X-ray.",
        limits="Trained on paediatric chest X-rays from one hospital; adult scans may be read less "
               "reliably.",
        upload_hint="A frontal chest X-ray, PNG or JPG.",
        labels={"NORMAL": "No signs of pneumonia", "PNEUMONIA": "Signs of pneumonia"},
        positive={"PNEUMONIA"}),
    "braintumor": dict(
        title="Brain tumour", kind="image", form="brain_tumor", post="resultbt",
        input="Brain MRI", model="VGG16 transfer learning",
        about="Crops the scan to the brain, then checks it for a tumour.",
        limits="Trained on about 230 MRI slices; it answers yes or no and does not locate or "
               "classify the tumour.",
        upload_hint="One axial brain MRI slice, PNG or JPG.",
        labels={"NO": "No tumour detected", "YES": "Tumour detected"},
        positive={"YES"}),
    "alzheimer": dict(
        title="Alzheimer's disease", kind="image", form="alzheimer", post="resulta",
        input="Brain MRI", model="Convolutional neural network",
        about="Estimates the stage of dementia from one MRI slice, from none to moderate.",
        limits="Very few moderate cases in the training data (52 images), so that stage is the "
               "least reliable.",
        upload_hint="One axial brain MRI slice, PNG or JPG.",
        labels={"NonDemented": "No dementia", "VeryMildDemented": "Very mild dementia",
                "MildDemented": "Mild dementia", "ModerateDemented": "Moderate dementia"},
        order=["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"],
        positive={"VeryMildDemented", "MildDemented", "ModerateDemented"}),
    "diabetes": dict(
        title="Diabetes", kind="clinical", form="diabetes", post="resultd",
        input="8 clinical measurements", model="Random forest",
        about="Estimates the chance of diabetes from routine measurements.",
        limits="Trained on 768 adult women of Pima heritage; at the default threshold it misses "
               "about 4 in 10 people who have diabetes.",
        labels={0: "No signs of diabetes", 1: "Signs of diabetes"}, positive={1},
        fields=[
            F("pregnancies", "Pregnancies", "Number of times pregnant", min=0, step=1),
            F("glucose", "Glucose", "Plasma glucose 2 h into an oral glucose tolerance test", unit="mg/dL", min=0),
            F("bloodpressure", "Blood pressure", "Diastolic", unit="mm Hg", min=0),
            F("skin", "Skin thickness", "Triceps skinfold", unit="mm", min=0),
            F("insulin", "Insulin", "2-hour serum insulin", unit="µU/mL", min=0),
            F("bmi", "BMI", "Weight in kg divided by height in m squared", unit="kg/m²", min=0, step="any"),
            F("diabetespedigree", "Diabetes pedigree", "Family-history score, usually 0.1–2.5", min=0, step="any"),
        ]),
    "heart": dict(
        title="Heart disease", kind="clinical", form="heartdisease", post="resulth",
        input="7 clinical measurements", model="Random forest",
        about="Estimates the chance of coronary heart disease from exercise-test and angiography results.",
        limits="Trained on 300 patients from the Cleveland Clinic dataset.",
        labels={0: "No signs of heart disease", 1: "Signs of heart disease"}, positive={1},
        fields=[
            F("tcp", "Chest pain type", "As recorded at the visit", options=[
                ("0", "Typical angina"), ("1", "Atypical angina"),
                ("2", "Non-anginal pain"), ("3", "Asymptomatic")]),
            F("mhra", "Maximum heart rate", "Highest rate reached during the exercise test", unit="bpm", min=0, step="any"),
            F("eia", "Exercise-induced angina", "Chest pain during the exercise test",
              options=[("0", "No"), ("1", "Yes")]),
            F("op", "ST depression", "ST depression induced by exercise relative to rest (oldpeak)", unit="mm", step="any"),
            F("nmv", "Major vessels", "Number of major vessels coloured by fluoroscopy",
              options=[(str(i), str(i)) for i in range(5)]),
            F("thal", "Thallium stress test", "Result code as recorded in the dataset",
              options=[("1", "1"), ("2", "2"), ("3", "3")]),
        ]),
    "breastcancer": dict(
        title="Breast cancer", kind="clinical", form="breast_cancer", post="resultbc",
        input="5 tumour measurements", model="Random forest",
        about="Classifies a breast mass as benign or malignant from measurements of cell nuclei in "
              "a fine-needle aspirate image.",
        limits="Trained on 569 samples from the Wisconsin Diagnostic dataset. Values are the mean "
               "over the nuclei in the image.",
        labels={0: "Benign", 1: "Malignant"}, positive={1},
        fields=[
            F("concave_points_mean", "Concave points", "Mean number of concave portions of the contour", step="any", min=0),
            F("area_mean", "Area", "Mean nucleus area", unit="px²", step="any", min=0),
            F("radius_mean", "Radius", "Mean distance from centre to the contour", unit="px", step="any", min=0),
            F("perimeter_mean", "Perimeter", "Mean nucleus perimeter", unit="px", step="any", min=0),
            F("concavity_mean", "Concavity", "Mean severity of concave portions of the contour", step="any", min=0),
        ]),
}
for slug, t in TASKS.items():
    t["slug"] = slug
    t["uses_age"] = slug in ("diabetes", "heart")  # age is a model input, not just patient detail

# ----------------------------------------------------------------- models
MODELS = {
    "covid": load_model(os.path.join(MODEL_DIR, "covid.h5")),
    "braintumor": load_model(os.path.join(MODEL_DIR, "braintumor.h5")),
    "alzheimer": load_model(os.path.join(MODEL_DIR, "alzheimer_model.h5")),
    "pneumonia": load_model(os.path.join(MODEL_DIR, "pneumonia_model.h5")),
    "breastcancer": joblib.load(os.path.join(MODEL_DIR, "cancer_model.pkl")),
}
with open(os.path.join(MODEL_DIR, "diabetes.sav"), "rb") as f:
    MODELS["diabetes"] = pickle.load(f)
with open(os.path.join(MODEL_DIR, "heart_disease.pickle.dat"), "rb") as f:
    MODELS["heart"] = pickle.load(f)

# ------------------------------------------------------------------ flask
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads
# Set MEDDETECT_SECRET_KEY in the environment for any shared deployment.
app.secret_key = os.environ.get("MEDDETECT_SECRET_KEY") or os.urandom(24)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class InputError(ValueError):
    """Raised when a form or upload can't be used for a prediction."""


# ---------------------------------------------------------------- helpers
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload():
    """Save the uploaded image under a random name; return (filename, BGR image)."""
    file = request.files.get("file")
    if not file or not file.filename:
        raise InputError("Choose an image to upload.")
    if not allowed_file(file.filename):
        raise InputError("Upload a PNG or JPG image.")
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    img = cv2.imread(path)
    if img is None:
        os.remove(path)
        raise InputError("That file couldn't be read as an image. Upload a PNG or JPG.")
    return filename, img


def patient_info():
    """Fields shown on the result page. Nothing here is stored."""
    form = request.form
    return dict(first_name=form.get("firstname", "").strip(), last_name=form.get("lastname", "").strip(),
                sex=form.get("gender", ""), age=form.get("age", "").strip())


def numbers(task):
    """The task's clinical fields as floats, in model order (age last where the model uses it)."""
    names = [f["name"] for f in task["fields"]]
    if task["uses_age"]:
        names.append("age")
    if task["slug"] == "heart":  # model order: nmv, tcp, eia, thal, op, mhra, age
        names = ["nmv", "tcp", "eia", "thal", "op", "mhra", "age"]
    values, labels = [], {f["name"]: f["label"] for f in task["fields"]}
    labels["age"] = "Age"
    for name in names:
        raw = request.form.get(name, "").strip()
        try:
            values.append(float(raw))
        except ValueError:
            raise InputError(f"Enter a number for {labels.get(name, name).lower()}.")
    return np.array(values, dtype=float).reshape(1, -1)


def image_probabilities(slug, img):
    """Probability of each class in MODEL_CLASSES[slug] order."""
    p = np.ravel(MODELS[slug].predict(model_input(slug, img)))
    return np.array([1 - p[0], p[0]]) if p.size == 1 else p


def clinical_probabilities(slug, x):
    model = MODELS[slug]
    p = np.ravel(model.predict_proba(x))
    return p / p.sum()


def result_view(task, probs, class_keys, filename=None, inputs=None):
    """Template data for one prediction."""
    best = class_keys[int(np.argmax(probs))]
    order = task.get("order", class_keys)
    rows = [dict(label=task["labels"][k], p=float(probs[class_keys.index(k)]),
                 positive=k in task["positive"], chosen=k == best) for k in order]
    return render_template("result.html", task=task, finding=task["labels"][best],
                           positive=best in task["positive"], rows=rows, filename=filename,
                           inputs=inputs, patient=patient_info())


def predict(slug):
    task = TASKS[slug]
    try:
        if task["kind"] == "image":
            filename, img = save_upload()
            probs = image_probabilities(slug, img)
            return result_view(task, probs, MODEL_CLASSES[slug], filename=filename)
        x = numbers(task)
        probs = clinical_probabilities(slug, x)
        shown = [(f["label"], request.form.get(f["name"]), f.get("unit", "")) for f in task["fields"]]
        for f in task["fields"]:  # show option labels rather than codes
            if "options" in f:
                i = [n for n, _, _ in shown].index(f["label"])
                code = request.form.get(f["name"])
                shown[i] = (f["label"], dict(f["options"]).get(code, code), "")
        return result_view(task, probs, [0, 1], inputs=shown)
    except InputError as err:
        flash(str(err))
        return redirect(url_for(task["form"]))


def form_page(slug):
    return render_template("detect.html", task=TASKS[slug])


# ------------------------------------------------------------------ pages
@app.route("/")
def home():
    imaging = [t for t in TASKS.values() if t["kind"] == "image"]
    clinical = [t for t in TASKS.values() if t["kind"] == "clinical"]
    return render_template("home.html", imaging=imaging, clinical=clinical)


@app.route("/covid")
def covid():
    return form_page("covid")


@app.route("/pneumonia")
def pneumonia():
    return form_page("pneumonia")


@app.route("/braintumor")
def brain_tumor():
    return form_page("braintumor")


@app.route("/alzheimer")
def alzheimer():
    return form_page("alzheimer")


@app.route("/diabetes")
def diabetes():
    return form_page("diabetes")


@app.route("/heartdisease")
def heartdisease():
    return form_page("heart")


@app.route("/breastcancer")
def breast_cancer():
    return form_page("breastcancer")


@app.route("/resultc", methods=["POST"])
def resultc():
    return predict("covid")


@app.route("/resultp", methods=["POST"])
def resultp():
    return predict("pneumonia")


@app.route("/resultbt", methods=["POST"])
def resultbt():
    return predict("braintumor")


@app.route("/resulta", methods=["POST"])
def resulta():
    return predict("alzheimer")


@app.route("/resultd", methods=["POST"])
def resultd():
    return predict("diabetes")


@app.route("/resulth", methods=["POST"])
def resulth():
    return predict("heart")


@app.route("/resultbc", methods=["POST"])
def resultbc():
    return predict("breastcancer")


@app.errorhandler(413)
def too_large(_err):
    flash("That file is larger than 10 MB. Upload a smaller image.")
    return redirect(request.referrer or url_for("home"))


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
