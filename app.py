import json
import os
import uuid
import logging
import importlib
import urllib.request
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
    tflite_runtime_module = importlib.import_module("tflite_runtime.interpreter")
    TFLiteInterpreter = tflite_runtime_module.Interpreter
except Exception:
    TFLiteInterpreter = None


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "brain_tumor_efficientnetb0.keras"
TFLITE_MODEL_PATH = PROJECT_DIR / "models" / "brain_tumor_efficientnetb0_quantized.tflite"
RUNTIME_TFLITE_MODEL_PATH = Path("/tmp") / "brain_tumor_efficientnetb0_quantized.tflite"
CLASS_MAP_PATH = PROJECT_DIR / "models" / "class_indices.json"
TESTING_DIR = PROJECT_DIR / "Dataset" / "Testing"

UPLOAD_DIR = PROJECT_DIR / "uploads"
RESULTS_DIR = PROJECT_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IS_RENDER = bool(os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_HOSTNAME"))
REPO_OWNER = os.environ.get("GITHUB_REPO_OWNER", "Kantdrav")
REPO_NAME = os.environ.get("GITHUB_REPO_NAME", "tumour")
REPO_BRANCH = os.environ.get("GITHUB_REPO_BRANCH", "main")
DEFAULT_TFLITE_URL = (
    f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{REPO_BRANCH}"
    "/models/brain_tumor_efficientnetb0_quantized.tflite"
)
TFLITE_DOWNLOAD_URL = os.environ.get("TFLITE_MODEL_URL", DEFAULT_TFLITE_URL)

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
app.logger.setLevel(logging.INFO)

# Lazy loading: model and cached models loaded on first request
model = None
tflite_interpreter = None
conv_model_cached = None
classifier_model_cached = None
model_load_error = None
use_tflite = False  # Track if we're using TFLite or Keras
tf_module = None
TF_IMPORT_ERROR = ""
active_tflite_path = TFLITE_MODEL_PATH


def get_tf_module():
    """Lazily import TensorFlow only if Keras fallback is needed."""
    global tf_module, TF_IMPORT_ERROR
    if tf_module is not None:
        return tf_module

    try:
        tf_module = importlib.import_module("tensorflow")
        return tf_module
    except Exception as exc:
        TF_IMPORT_ERROR = str(exc)
        logger.error("TensorFlow lazy import failed: %s", TF_IMPORT_ERROR)
        return None


def ensure_tflite_model_exists() -> bool:
    """Ensure quantized model exists; auto-download on Render if missing."""
    global active_tflite_path

    if TFLITE_MODEL_PATH.exists():
        active_tflite_path = TFLITE_MODEL_PATH
        return True

    if IS_RENDER and RUNTIME_TFLITE_MODEL_PATH.exists() and RUNTIME_TFLITE_MODEL_PATH.stat().st_size > 0:
        active_tflite_path = RUNTIME_TFLITE_MODEL_PATH
        return True

    if not IS_RENDER:
        return False

    try:
        logger.warning("TFLite model missing. Attempting download from %s", TFLITE_DOWNLOAD_URL)
        RUNTIME_TFLITE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TFLITE_DOWNLOAD_URL, str(RUNTIME_TFLITE_MODEL_PATH))
        exists_now = RUNTIME_TFLITE_MODEL_PATH.exists() and RUNTIME_TFLITE_MODEL_PATH.stat().st_size > 0
        if exists_now:
            active_tflite_path = RUNTIME_TFLITE_MODEL_PATH
            logger.info("Downloaded TFLite model to %s", active_tflite_path)
            return True
        logger.error("Downloaded file is invalid or empty: %s", RUNTIME_TFLITE_MODEL_PATH)
        return False
    except Exception as exc:
        logger.error("Failed to download TFLite model: %s", exc)
        return False

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
    """Load model (TFLite or Keras) lazily on first use."""
    global model, tflite_interpreter, conv_model_cached, classifier_model_cached, model_load_error, use_tflite
    
    if model is not None or tflite_interpreter is not None:
        return True  # Already loaded
    
    if model_load_error is not None:
        return False  # Already tried and failed
    
    # Try TFLite first (70-90% smaller, 2-3x faster)
    tflite_available = ensure_tflite_model_exists()
    if tflite_available:
        try:
            logger.info(f"Loading quantized TFLite model from {active_tflite_path}...")
            if TFLiteInterpreter is not None:
                tflite_interpreter = TFLiteInterpreter(model_path=str(active_tflite_path))
            else:
                tf_fallback = get_tf_module()
                if tf_fallback is None:
                    raise RuntimeError(f"No TFLite interpreter available. TensorFlow error: {TF_IMPORT_ERROR}")
                tflite_interpreter = tf_fallback.lite.Interpreter(model_path=str(active_tflite_path))
            tflite_interpreter.allocate_tensors()
            use_tflite = True
            logger.info("✓ TFLite model loaded successfully (70-90% smaller, 2-3x faster)")
            return True
        except Exception as e:
            logger.warning(f"TFLite loading failed, falling back to Keras: {e}")
            tflite_interpreter = None
            use_tflite = False
    
    if IS_RENDER:
        model_load_error = (
            "Optimized TFLite model not available. "
            f"Tried local path and download URL: {TFLITE_DOWNLOAD_URL}"
        )
        logger.error(model_load_error)
        return False

    # Fallback to Keras model (local/dev only)
    if not MODEL_PATH.exists():
        model_load_error = f"Model file not found at {MODEL_PATH}"
        logger.error(model_load_error)
        return False

    tf_keras = get_tf_module()
    if tf_keras is None:
        model_load_error = f"TensorFlow is not available for Keras fallback: {TF_IMPORT_ERROR}"
        logger.error(model_load_error)
        return False
    
    try:
        logger.info(f"Loading Keras model from {MODEL_PATH}...")
        model = tf_keras.keras.models.load_model(MODEL_PATH)
        use_tflite = False
        logger.info("✓ Keras model loaded successfully")
        
        # Pre-build Grad-CAM models for faster inference
        try:
            logger.info("Building Grad-CAM models...")
            backbone = model.get_layer("efficientnetb0")
            conv_model_cached = tf_keras.keras.models.Model(backbone.input, backbone.output)
            classifier_input = tf_keras.keras.Input(shape=backbone.output.shape[1:])
            x = classifier_input
            for layer in model.layers[2:]:
                x = layer(x)
            classifier_model_cached = tf_keras.keras.models.Model(classifier_input, x)
            logger.info("✓ Grad-CAM models built successfully")
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
    tf_for_gradcam = get_tf_module()
    if tf_for_gradcam is None:
        logger.warning("Grad-CAM skipped because TensorFlow is unavailable")
        return False

    # Use cached models instead of rebuilding
    if conv_model_cached is None or classifier_model_cached is None:
        logger.warning("Grad-CAM models not available")
        return False

    try:
        with tf_for_gradcam.GradientTape() as tape:
            conv_outputs = conv_model_cached(input_batch)
            tape.watch(conv_outputs)
            predictions = classifier_model_cached(conv_outputs)
            loss_value = predictions[:, pred_idx]

        grads = tape.gradient(loss_value, conv_outputs)
        pooled_grads = tf_for_gradcam.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., None]
        heatmap = tf_for_gradcam.squeeze(heatmap)

        heatmap = tf_for_gradcam.maximum(heatmap, 0)
        heatmap /= tf_for_gradcam.reduce_max(heatmap) + tf_for_gradcam.keras.backend.epsilon()
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


def run_prediction(input_batch):
    """Run prediction using TFLite or Keras model."""
    if use_tflite and tflite_interpreter is not None:
        # Use TFLite for prediction
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        
        # Scale input to uint8 if needed
        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_batch_scaled = (input_batch / input_scale + input_zero_point).astype(np.uint8)
        else:
            input_batch_scaled = input_batch.astype(np.float32)
        
        tflite_interpreter.set_tensor(input_details[0]['index'], input_batch_scaled)
        tflite_interpreter.invoke()
        
        preds = tflite_interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output if needed
        if output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            preds = (preds.astype(np.float32) - output_zero_point) * output_scale
        
        # Ensure output shape is (1, num_classes)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)
        
        return preds
    else:
        # Use Keras model
        return model.predict(input_batch, verbose=0)


@app.route("/", methods=["GET"])
def index():
    """Serve the main upload page."""
    # Don't load model here, just check if dependencies exist
    model_ready = (
        cv2 is not None
        and np is not None
        and len(class_labels) > 0
        and (TFLITE_MODEL_PATH.exists() or MODEL_PATH.exists())
    )
    error_msg = None
    
    if not model_ready:
        if cv2 is None:
            error_msg = f"OpenCV not available: {CV2_IMPORT_ERROR}"
        elif np is None:
            error_msg = f"NumPy not available: {NUMPY_IMPORT_ERROR}"
        elif len(class_labels) == 0:
            error_msg = "No classes found. Train the model first with: python train_model.py"
        elif not (TFLITE_MODEL_PATH.exists() or MODEL_PATH.exists()):
            error_msg = "No model found. Train first using: python train_model.py"
        
        logger.warning(f"Index page accessed with error: {error_msg}")
    
    return render_template("index.html", model_ready=model_ready, error=error_msg)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handle image prediction requests."""
    if request.method == "GET":
        return render_template(
            "index.html",
            error="Use the upload form on the home page to submit an image.",
            model_ready=True,
        )

    logger.info("Prediction request received")
    
    # Check dependencies
    errors = []
    if np is None:
        errors.append(f"NumPy missing: {NUMPY_IMPORT_ERROR}")
    if cv2 is None:
        errors.append(f"OpenCV missing: {CV2_IMPORT_ERROR}")
    
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
        
        logger.info(f"Running prediction (using {'TFLite' if use_tflite else 'Keras'})...")
        preds = run_prediction(input_batch)
        pred_idx = int(np.argmax(preds[0]))
        pred_label = class_labels[pred_idx] if class_labels else str(pred_idx)
        pred_conf = float(preds[0][pred_idx])
        
        logger.info(f"Prediction: {pred_label} ({pred_conf:.4f})")
        
        # Generate Grad-CAM overlay (only works with Keras model)
        gradcam_ready = False
        gradcam_note = None
        if use_tflite:
            gradcam_note = "Grad-CAM visualization not available with TFLite (quantized model); showing original image."
        else:
            gradcam_ready = make_gradcam_overlay(input_batch, upload_path, overlay_path, pred_idx)
            gradcam_note = None if gradcam_ready else "Grad-CAM unavailable for this request; showing original image."
        
        table = get_probability_table(preds)
        
        gradcam_image = f"/results/{overlay_path.name}" if gradcam_ready else f"/uploads/{unique_name}"
        
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
    model_type = "TFLite (optimized)" if use_tflite else "Keras" if model is not None else "Not loaded"
    return {
        "status": "ok",
        "model_loaded": model is not None or tflite_interpreter is not None,
        "model_type": model_type,
        "tflite_file_exists": TFLITE_MODEL_PATH.exists(),
        "active_tflite_path": str(active_tflite_path),
        "runtime_tflite_file_exists": RUNTIME_TFLITE_MODEL_PATH.exists(),
        "tflite_download_url": TFLITE_DOWNLOAD_URL,
        "model_load_error": model_load_error,
        "dependencies_available": {
            "tensorflow_loaded": tf_module is not None,
            "tflite_runtime": TFLiteInterpreter is not None,
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
    logger.info("Running on Render: %s", IS_RENDER)
    logger.info(
        "Dependencies available - TFLite runtime: %s, OpenCV: %s, NumPy: %s",
        TFLiteInterpreter is not None,
        cv2 is not None,
        np is not None,
    )
    logger.info(f"Keras model path: {MODEL_PATH} (exists: {MODEL_PATH.exists()})")
    logger.info(f"TFLite model path: {TFLITE_MODEL_PATH} (exists: {TFLITE_MODEL_PATH.exists()})")
    logger.info(f"Classes loaded: {len(class_labels)} ({', '.join(class_labels)})")
    logger.info("Note: Model loads lazily on first prediction request for memory efficiency")
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=port, debug=debug)
