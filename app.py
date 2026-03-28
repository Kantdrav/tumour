import json
import os
import uuid
import logging
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cv2
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)
    logger.error(f"OpenCV import failed: {CV2_IMPORT_ERROR}")
else:
    CV2_IMPORT_ERROR = ""

try:
    import numpy as np
except Exception as exc:
    np = None
    NUMPY_IMPORT_ERROR = str(exc)
    logger.error(f"NumPy import failed: {NUMPY_IMPORT_ERROR}")
else:
    NUMPY_IMPORT_ERROR = ""

try:
    import tensorflow as tf
except Exception as exc:
    tf = None
    TF_IMPORT_ERROR = str(exc)
    logger.error(f"TensorFlow import failed: {TF_IMPORT_ERROR}")
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
app.logger.setLevel(logging.INFO)

# Lazy loading: model and cached models loaded on first request
model = None
conv_model_cached = None
classifier_model_cached = None
model_load_error = None

# Load class labels from JSON or filesystem
if CLASS_MAP_PATH.exists():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as handle:
        class_indices = json.load(handle)
    class_labels = [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]
    logger.info(f"Loaded {len(class_labels)} classes from {CLASS_MAP_PATH}")
else:
    class_labels = sorted([p.name for p in TESTING_DIR.iterdir() if p.is_dir()]) if TESTING_DIR.exists() else []
    logger.info(f"Loaded {len(class_labels)} classes from filesystem")


def load_model_and_cache():
    """Load model and build Grad-CAM models lazily on first use."""
    global model, conv_model_cached, classifier_model_cached, model_load_error
    
    if model is not None:
        return True  # Already loaded
    
    if model_load_error is not None:
        return False  # Already tried and failed
    
    if tf is None:
        model_load_error = f"TensorFlow is not available: {TF_IMPORT_ERROR}"
        logger.error(model_load_error)
        return False
    
    if not MODEL_PATH.exists():
        model_load_error = f"Model file not found at {MODEL_PATH}"
        logger.error(model_load_error)
        return False
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Pre-build Grad-CAM models for faster inference
        try:
            logger.info("Building Grad-CAM models...")
            backbone = model.get_layer("efficientnetb0")
            conv_model_cached = tf.keras.models.Model(backbone.input, backbone.output)
            classifier_input = tf.keras.Input(shape=backbone.output.shape[1:])
            x = classifier_input
            for layer in model.layers[2:]:
                x = layer(x)
            classifier_model_cached = tf.keras.models.Model(classifier_input, x)
            logger.info("Grad-CAM models built successfully")
        except Exception as e:
            logger.warning(f"Grad-CAM models failed to build: {e}")
            conv_model_cached = None
            classifier_model_cached = None
        
        return True
    except Exception as e:
        model_load_error = f"Failed to load model: {str(e)}"
        logger.error(model_load_error, exc_info=True)
        return False


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path: Path):
    """Preprocess image for model input."""
    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Failed to read image at {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
        image_array = image_rgb.astype("float32") / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}", exc_info=True)
        raise


def make_gradcam_overlay(
    input_batch: Any,
    image_path: Path,
    output_path: Path,
    pred_idx: int,
) -> bool:
    """Generate Grad-CAM visualization overlay."""
    # Use cached models instead of rebuilding
    if conv_model_cached is None or classifier_model_cached is None:
        logger.warning("Grad-CAM models not available")
        return False

    try:
        with tf.GradientTape() as tape:
            conv_outputs = conv_model_cached(input_batch)
            tape.watch(conv_outputs)
            predictions = classifier_model_cached(conv_outputs)
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

        success = bool(cv2.imwrite(str(output_path), overlay))
        if success:
            logger.info(f"Grad-CAM overlay saved to {output_path}")
        return success
    except Exception as exc:
        logger.error(f"Grad-CAM generation failed: {exc}", exc_info=True)
        return False


def get_probability_table(preds) -> list[dict[str, float | str]]:
    rows = []
    for idx, score in enumerate(preds[0]):
        label = class_labels[idx] if idx < len(class_labels) else str(idx)
        rows.append({"label": label, "score": float(score)})
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


@app.route("/", methods=["GET"])
def index():
    """Serve the main upload page."""
    # Don't load model here, just check if dependencies exist
    model_ready = tf is not None and cv2 is not None and np is not None and len(class_labels) > 0
    error_msg = None
    
    if not model_ready:
        if tf is None:
            error_msg = f"TensorFlow not available: {TF_IMPORT_ERROR}"
        elif cv2 is None:
            error_msg = f"OpenCV not available: {CV2_IMPORT_ERROR}"
        elif np is None:
            error_msg = f"NumPy not available: {NUMPY_IMPORT_ERROR}"
        elif len(class_labels) == 0:
            error_msg = "No classes found. Train the model first with: python train_model.py"
        
        logger.warning(f"Index page accessed with error: {error_msg}")
    
    return render_template("index.html", model_ready=model_ready, error=error_msg)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image prediction requests."""
    logger.info("Prediction request received")
    
    # Check dependencies
    errors = []
    if np is None:
        errors.append(f"NumPy missing: {NUMPY_IMPORT_ERROR}")
    if cv2 is None:
        errors.append(f"OpenCV missing: {CV2_IMPORT_ERROR}")
    if tf is None:
        errors.append(f"TensorFlow missing: {TF_IMPORT_ERROR}")
    
    if errors:
        error_msg = " | ".join(errors)
        logger.error(f"Missing dependencies: {error_msg}")
        return render_template("index.html", error=error_msg, model_ready=False)
    
    # Validate file input
    file = request.files.get("file")
    if file is None or file.filename == "":
        logger.warning("No file provided")
        return render_template("index.html", error="Please upload an image file.", model_ready=True)

    if not allowed_file(file.filename):
        logger.warning(f"Invalid file extension: {file.filename}")
        return render_template(
            "index.html",
            error="Only PNG, JPG, and JPEG files are allowed.",
            model_ready=True,
        )
    
    # Load model on first prediction
    logger.info("Loading model for prediction...")
    if not load_model_and_cache():
        error_msg = model_load_error or "Failed to load model"
        logger.error(f"Model load failed: {error_msg}")
        return render_template(
            "index.html",
            error=error_msg,
            model_ready=False,
        )
    
    # Save uploaded file
    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    upload_path = UPLOAD_DIR / unique_name
    overlay_path = RESULTS_DIR / f"overlay_{unique_name}"
    
    try:
        file.save(upload_path)
        logger.info(f"File saved to {upload_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        return render_template(
            "index.html",
            error="Error saving uploaded file. Try again.",
            model_ready=True,
        )
    
    # Preprocess and predict
    try:
        logger.info("Preprocessing image...")
        input_batch = preprocess_image(upload_path)
        
        logger.info("Running prediction...")
        preds = model.predict(input_batch, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        pred_label = class_labels[pred_idx] if class_labels else str(pred_idx)
        pred_conf = float(preds[0][pred_idx])
        
        logger.info(f"Prediction: {pred_label} ({pred_conf:.4f})")
        
        # Generate Grad-CAM overlay
        gradcam_ready = make_gradcam_overlay(input_batch, upload_path, overlay_path, pred_idx)
        table = get_probability_table(preds)
        
        gradcam_image = f"/results/{overlay_path.name}" if gradcam_ready else f"/uploads/{unique_name}"
        gradcam_note = None if gradcam_ready else "Grad-CAM unavailable for this request; showing original image."
        
        logger.info("Prediction completed successfully")
        return render_template(
            "result.html",
            prediction=pred_label,
            confidence=pred_conf,
            probabilities=table,
            uploaded_image=f"/uploads/{unique_name}",
            gradcam_image=gradcam_image,
            gradcam_note=gradcam_note,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template(
            "index.html",
            error=f"Error processing image: {str(e)}",
            model_ready=True,
        )


@app.route("/uploads/<path:filename>")
def uploads(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/results/<path:filename>")
def results(filename: str):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint with detailed status."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_load_error": model_load_error,
        "dependencies_available": {
            "tensorflow": tf is not None,
            "opencv": cv2 is not None,
            "numpy": np is not None,
        },
        "classes_count": len(class_labels),
        "classes": class_labels,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    
    logger.info("=" * 60)
    logger.info("Starting Brain Tumor Detection Application")
    logger.info(f"Port: {port}, Debug: {debug}")
    logger.info(f"Dependencies available - TF: {tf is not None}, OpenCV: {cv2 is not None}, NumPy: {np is not None}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Model exists: {MODEL_PATH.exists()}")
    logger.info(f"Classes loaded: {len(class_labels)} ({', '.join(class_labels)})")
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=port, debug=debug)
