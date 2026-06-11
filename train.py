"""
train.py - MobileNet Emotion Detection Model Training
Includes REST API endpoint to trigger training remotely.
"""

import os
import threading
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from flask import Flask, jsonify, request

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
NUM_CLASSES    = 7
BATCH_SIZE     = 32
EPOCHS         = 25
TRAIN_DIR      = "train"
VALIDATION_DIR = "validation"
MODEL_SAVE_PATH = "final_emotion_detection_model.keras"

CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Global training status (thread-safe enough for a single-user dev server)
training_status = {"status": "idle", "epoch": 0, "val_accuracy": None, "message": ""}


# ── Model definition ──────────────────────────────────────────────────────

def build_model():
    base_model = MobileNet(weights='imagenet', include_top=False,
                           input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=out)


# ── Training logic ────────────────────────────────────────────────────────

class StatusCallback(tf.keras.callbacks.Callback):
    """Updates the global status dict after every epoch."""
    def on_epoch_end(self, epoch, logs=None):
        training_status["epoch"] = epoch + 1
        training_status["val_accuracy"] = round(logs.get("val_accuracy", 0), 4)
        training_status["message"] = (
            f"Epoch {epoch+1}/{EPOCHS} — "
            f"loss: {logs.get('loss', 0):.4f} — "
            f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
        )


def run_training():
    global training_status
    training_status["status"] = "running"
    training_status["message"] = "Initialising training..."

    try:
        model = build_model()

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_gen = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=IMG_SIZE,
            batch_size=BATCH_SIZE, class_mode='categorical'
        )
        val_gen = val_datagen.flow_from_directory(
            VALIDATION_DIR, target_size=IMG_SIZE,
            batch_size=BATCH_SIZE, class_mode='categorical'
        )

        callbacks = [
            StatusCallback(),
            ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss',
                            save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_accuracy', patience=5,
                              factor=0.2, min_lr=0.0001, verbose=1)
        ]

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=val_gen.samples // BATCH_SIZE
        )

        model.save(MODEL_SAVE_PATH)
        training_status["status"] = "completed"
        training_status["message"] = f"Training complete. Model saved to {MODEL_SAVE_PATH}"

    except Exception as e:
        training_status["status"] = "error"
        training_status["message"] = str(e)


# ── REST API endpoints ────────────────────────────────────────────────────

@app.route('/api/train/start', methods=['POST'])
def start_training():
    """
    POST /api/train/start
    Kick off model training in a background thread.
    """
    if training_status["status"] == "running":
        return jsonify({"error": "Training is already in progress."}), 409

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return jsonify({"message": "Training started.", "status": "running"}), 202


@app.route('/api/train/status', methods=['GET'])
def get_status():
    """
    GET /api/train/status
    Returns current training status, epoch, and latest val_accuracy.
    """
    return jsonify(training_status)


@app.route('/api/train/model-info', methods=['GET'])
def model_info():
    """
    GET /api/train/model-info
    Returns model metadata and whether a saved model exists.
    """
    return jsonify({
        "model_architecture": "MobileNet (ImageNet weights) + custom head",
        "num_classes": NUM_CLASSES,
        "class_labels": CLASS_LABELS,
        "input_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}x3",
        "batch_size": BATCH_SIZE,
        "epochs_configured": EPOCHS,
        "model_saved": os.path.exists(MODEL_SAVE_PATH),
        "model_path": MODEL_SAVE_PATH
    })


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)
    print("Training API running at http://localhost:5001")
    print("  POST /api/train/start       — start training")
    print("  GET  /api/train/status      — check progress")
    print("  GET  /api/train/model-info  — model config")
    app.run(debug=False, port=5001)
