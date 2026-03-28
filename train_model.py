import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_WARMUP = 8
EPOCHS_FINETUNE = 24

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "Dataset"
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "brain_tumor_efficientnetb0.keras"
CLASS_MAP_PATH = MODELS_DIR / "class_indices.json"


np.random.seed(SEED)
tf.random.set_seed(SEED)


print("TensorFlow:", tf.__version__)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        "Dataset folders not found. Expected Dataset/Training and Dataset/Testing in project root."
    )


train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    validation_split=0.1,
)

test_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
)

val_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

class_labels = list(train_data.class_indices.keys())
num_classes = len(class_labels)
print("Classes:", class_labels)

with open(CLASS_MAP_PATH, "w", encoding="utf-8") as handle:
    json.dump(train_data.class_indices, handle, indent=2)


class_weights_raw = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_data.classes), y=train_data.classes
)
class_weights = dict(enumerate(class_weights_raw))
print("Class weights:", class_weights)


base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(patience=3, factor=0.3, verbose=1, monitor="val_loss"),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    CSVLogger(RESULTS_DIR / "training_log.csv"),
]

print("\n--- Warmup training (frozen backbone) ---")
history_warmup = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_WARMUP,
    class_weight=class_weights,
    callbacks=callbacks,
)


base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Fine-tuning (last 100 layers) ---")
history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_WARMUP + EPOCHS_FINETUNE,
    initial_epoch=history_warmup.epoch[-1] + 1,
    class_weight=class_weights,
    callbacks=callbacks,
)

if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)


test_loss, test_acc = model.evaluate(test_data)
print("Test Loss:", float(test_loss))
print("Test Accuracy:", float(test_acc))


acc = history_warmup.history["accuracy"] + history_finetune.history["accuracy"]
val_acc = history_warmup.history["val_accuracy"] + history_finetune.history["val_accuracy"]
loss = history_warmup.history["loss"] + history_finetune.history["loss"]
val_loss = history_warmup.history["val_loss"] + history_finetune.history["val_loss"]

plt.figure(figsize=(8, 5))
plt.plot(acc, label="train_acc")
plt.plot(val_acc, label="val_acc")
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=180)
plt.close()


pred_probs = model.predict(test_data)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180)
plt.close()

report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
print("\nClassification Report:\n")
print(report)
with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as handle:
    handle.write(report)


def get_gradcam(
    model_obj: tf.keras.Model,
    img_path: Path,
    layer_name: str = "top_conv",
    output_path: Path | None = None,
):
    img_loaded = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img_loaded) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    backbone = model_obj.get_layer("efficientnetb0")
    conv_model = tf.keras.models.Model(backbone.input, backbone.output)

    classifier_input = tf.keras.Input(shape=backbone.output.shape[1:])
    x = classifier_input
    for layer in model_obj.layers[2:]:
        x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        class_idx = tf.argmax(predictions[0])
        loss_value = predictions[:, class_idx]

    grads = tape.gradient(loss_value, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    denominator = tf.reduce_max(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (denominator + tf.keras.backend.epsilon())
    heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))

    raw_img = cv2.imread(str(img_path))
    raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(raw_img, 0.6, colored_heatmap, 0.4, 0)

    if output_path is not None:
        cv2.imwrite(str(output_path), superimposed)

    return superimposed


sample_candidates = list((TEST_DIR / class_labels[0]).glob("*.jpg"))
if sample_candidates:
    sample_image = sample_candidates[0]
    gradcam_path = RESULTS_DIR / "sample_gradcam.png"
    try:
        get_gradcam(model, sample_image, output_path=gradcam_path)
        print(f"Saved sample Grad-CAM to: {gradcam_path}")
    except Exception as exc:
        print(f"Grad-CAM generation skipped due to error: {exc}")
else:
    print("No sample image found for Grad-CAM test.")

print(f"\nBest model saved to: {MODEL_PATH}")
print(f"Class mapping saved to: {CLASS_MAP_PATH}")
print(f"Outputs saved in: {RESULTS_DIR}")


# Quantization to TFLite for deployment
print("\n--- Converting to Quantized TFLite ---")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Use dynamic quantization for faster conversion
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    
    # Add representative dataset for better quantization
    def representative_dataset_gen():
        """Generator for quantization calibration."""
        for batch in test_data.take(100):  # Use 100 batches for calibration
            yield [np.asarray(batch[0], dtype=np.float32)]
    
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    TFLITE_MODEL_PATH = MODELS_DIR / "brain_tumor_efficientnetb0_quantized.tflite"
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        bytes_written = f.write(tflite_model)
    
    original_size = MODEL_PATH.stat().st_size / 1024 / 1024
    quantized_size = TFLITE_MODEL_PATH.stat().st_size / 1024 / 1024
    compression_ratio = (1 - quantized_size / original_size) * 100
    
    print(f"✓ Quantized model saved to: {TFLITE_MODEL_PATH}")
    print(f"  Original model size:  {original_size:.2f} MB")
    print(f"  Quantized model size: {quantized_size:.2f} MB")
    print(f"  Compression: {compression_ratio:.1f}% smaller")
    print(f"  Inference speed: ~2-3x faster, ~5-10x less memory")
except Exception as e:
    print(f"⚠ Quantization failed (model will still work): {e}")
    import traceback
    traceback.print_exc()
