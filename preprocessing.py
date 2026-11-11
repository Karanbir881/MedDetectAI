"""Preprocessing shared by the web app (app.py) and the training code (training/),
so a model is always served with exactly the inputs it was trained on.

Image functions take OpenCV BGR uint8 arrays, as returned by cv2.imread. The
HealthCure models were trained with Keras' ImageDataGenerator.flow_from_directory,
which loads images as RGB and resizes with nearest-neighbour interpolation, so
model_input converts to RGB and resizes the same way.
"""
import numpy as np

# Input size for each image task.
IMAGE_SIZES = {"covid": 224, "pneumonia": 150, "alzheimer": 176, "braintumor": 224}

# Class order of each image model's output. flow_from_directory numbers class
# folders alphabetically, which is how the HealthCure models were trained.
MODEL_CLASSES = {
    "covid": ["COVID", "NORMAL"],
    "pneumonia": ["NORMAL", "PNEUMONIA"],
    "braintumor": ["NO", "YES"],
    "alzheimer": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"],
}

# Pima diabetes: a 0 in these columns means "not measured".
DIABETES_ZERO_IS_MISSING = [1, 2, 3, 4, 5]  # Glucose, BloodPressure, SkinThickness, Insulin, BMI


def zeros_to_nan(X):
    """Mark unmeasured diabetes values (recorded as 0) as missing. Used inside saved pipelines."""
    X = np.array(X, dtype=float, copy=True)
    cols = X[:, DIABETES_ZERO_IS_MISSING]
    cols[cols == 0] = np.nan
    X[:, DIABETES_ZERO_IS_MISSING] = cols
    return X


def _resize(img, size):
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def _to_rgb(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def scaled(img, size):
    """BGR image -> RGB, resized to size x size, batch axis added, pixels in [0, 1]."""
    return _resize(_to_rgb(img), size).reshape(1, size, size, 3).astype("float32") / 255.0


def crop_brain(img, add_pixels_value=0):
    """Crop an MRI slice (BGR) to the bounding box of its largest contour.

    Same steps as the HealthCure notebook. Returns the uncropped image when no
    contour is found, instead of crashing on blank or very dark images.
    """
    import cv2
    import imutils
    # The notebook applied RGB2GRAY to cv2's BGR images; kept so crops match training.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = imutils.grab_contours(
        cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    if not cnts:
        return img.copy()
    c = max(cnts, key=cv2.contourArea)
    left, right = c[:, :, 0].min(), c[:, :, 0].max()
    top, bottom = c[:, :, 1].min(), c[:, :, 1].max()
    pad = add_pixels_value
    cropped = img[max(top - pad, 0):bottom + pad, max(left - pad, 0):right + pad].copy()
    return cropped if cropped.size else img.copy()


def vgg_input(img, size=224):
    """Crop, convert to RGB, resize and apply VGG16 preprocessing -> (1, size, size, 3)."""
    from tensorflow.keras.applications.vgg16 import preprocess_input
    img = _resize(_to_rgb(crop_brain(img)), size).astype("float32")
    return preprocess_input(img)[np.newaxis]


def model_input(task, img):
    """The exact array the model for `task` receives for one BGR image."""
    if task == "braintumor":
        return vgg_input(img, IMAGE_SIZES[task])
    return scaled(img, IMAGE_SIZES[task])
