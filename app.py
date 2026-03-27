import json
import os
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    import cv2
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)
else:
    CV2_IMPORT_ERROR = ""

try:
    import numpy as np
except Exception as exc:
    np = None
    NUMPY_IMPORT_ERROR = str(exc)
else:
    NUMPY_IMPORT_ERROR = ""

try:
    import tensorflow as tf
except Exception as exc:
    tf = None
    TF_IMPORT_ERROR = str(exc)
else:
    TF_IMPORT_ERROR = ""


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "brain_tumor_efficientnetb0.keras"
CLASS_MAP_PATH = PROJECT_DIR / "models" / "class_indices.json"
TESTING_DIR = PROJECT_DIR / "Dataset" / "Testing"

UPLOAD_DIR = PROJECT_DIR / "uploads"
RESULTS_DIR = PROJECT_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")


if tf is not None and MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None


if CLASS_MAP_PATH.exists():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as handle:
        class_indices = json.load(handle)
    class_labels = [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]
else:
    class_labels = sorted([p.name for p in TESTING_DIR.iterdir() if p.is_dir()]) if TESTING_DIR.exists() else []


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path: Path):
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_array = image_rgb.astype("float32") / 255.0
    return np.expand_dims(image_array, axis=0)


def build_gradcam_models(model_obj: Any):
    backbone = model_obj.get_layer("efficientnetb0")
    conv_model = tf.keras.models.Model(backbone.input, backbone.output)

    classifier_input = tf.keras.Input(shape=backbone.output.shape[1:])
    x = classifier_input
    for layer in model_obj.layers[2:]:
        x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    return conv_model, classifier_model


def make_gradcam_overlay(model_obj: Any, image_path: Path, output_path: Path) -> tuple[str, float]:
    input_batch = preprocess_image(image_path)

    preds = model_obj.predict(input_batch, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])

    class_name = class_labels[pred_idx] if class_labels else str(pred_idx)

    conv_model, classifier_model = build_gradcam_models(model_obj)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(input_batch)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        loss_value = predictions[:, pred_idx]

    grads = tape.gradient(loss_value, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    heatmap_np = heatmap.numpy()

    original_bgr = cv2.imread(str(image_path))
    original_bgr = cv2.resize(original_bgr, (IMG_SIZE, IMG_SIZE))

    heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap_np, (IMG_SIZE, IMG_SIZE)))
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.6, colored_heatmap, 0.4, 0)

    cv2.imwrite(str(output_path), overlay)
    return class_name, confidence


def get_probability_table(preds) -> list[dict[str, float | str]]:
    rows = []
    for idx, score in enumerate(preds[0]):
        label = class_labels[idx] if idx < len(class_labels) else str(idx)
        rows.append({"label": label, "score": float(score)})
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


@app.route("/", methods=["GET"])
def index():
    model_ready = model is not None and len(class_labels) > 0 and tf is not None
    tf_warning = None
    if tf is None:
        tf_warning = "TensorFlow is not installed in this environment. Install dependencies and restart."
    return render_template("index.html", model_ready=model_ready, error=tf_warning)


@app.route("/predict", methods=["POST"])
def predict():
    if np is None:
        return render_template(
            "index.html",
            error=f"NumPy is missing: {NUMPY_IMPORT_ERROR}",
            model_ready=False,
        )

    if cv2 is None:
        return render_template(
            "index.html",
            error=f"OpenCV is missing: {CV2_IMPORT_ERROR}",
            model_ready=False,
        )

    if tf is None:
        return render_template(
            "index.html",
            error=f"TensorFlow is missing: {TF_IMPORT_ERROR}",
            model_ready=False,
        )

    if model is None:
        return render_template(
            "index.html",
            error="Model not found. Train first using: python train_model.py",
            model_ready=False,
        )

    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please upload an image file.", model_ready=True)

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Only PNG, JPG, and JPEG files are allowed.",
            model_ready=True,
        )

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    upload_path = UPLOAD_DIR / unique_name
    overlay_path = RESULTS_DIR / f"overlay_{unique_name}"

    file.save(upload_path)

    input_batch = preprocess_image(upload_path)
    preds = model.predict(input_batch, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    pred_label = class_labels[pred_idx] if class_labels else str(pred_idx)
    pred_conf = float(preds[0][pred_idx])

    make_gradcam_overlay(model, upload_path, overlay_path)
    table = get_probability_table(preds)

    return render_template(
        "result.html",
        prediction=pred_label,
        confidence=pred_conf,
        probabilities=table,
        uploaded_image=f"/uploads/{unique_name}",
        gradcam_image=f"/results/{overlay_path.name}",
    )


@app.route("/uploads/<path:filename>")
def uploads(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/results/<path:filename>")
def results(filename: str):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": class_labels,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
