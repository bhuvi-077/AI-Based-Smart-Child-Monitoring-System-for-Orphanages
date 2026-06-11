"""
test4.py - Real-Time Emotion Detection with Grad-CAM + GPS Alerts
Includes REST API endpoints alongside the webcam loop.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.cm as cm
import random
import os
import base64
import threading
from datetime import datetime
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH   = "final_emotion_detection_model.keras"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
ALERT_LOG    = "alerts_log.txt"
HEATMAP_DIR  = "gradcam_alerts"
ALERT_EMOTIONS = {"Sad", "Fear", "Angry"}
CONFIDENCE_THRESHOLD = 0.40
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

os.makedirs(HEATMAP_DIR, exist_ok=True)

# Shared state (read by API, written by webcam thread)
latest_detection = {
    "emotion": None,
    "confidence": None,
    "gps": None,
    "heatmap_file": None,
    "timestamp": None,
    "alert_triggered": False
}
detection_lock = threading.Lock()

# ── Load ML model ─────────────────────────────────────────────────────────
classifier      = load_model(MODEL_PATH)
face_classifier = cv2.CascadeClassifier(CASCADE_PATH)


# ── GPS simulation ────────────────────────────────────────────────────────

def get_gps_coordinates():
    return {
        "lat": round(random.uniform(12.9000, 13.1000), 6),
        "lon": round(random.uniform(77.5000, 77.7000), 6)
    }


# ── Alert ─────────────────────────────────────────────────────────────────

def trigger_alert(emotion, confidence, gps, heatmap_file):
    print(f"\n🚨 ALERT — Emotion: {emotion} | Conf: {confidence:.2f} | GPS: {gps}")
    with open(ALERT_LOG, "a") as f:
        f.write(
            f"Emotion: {emotion}, Confidence: {confidence:.2f}, "
            f"GPS: {gps}, Heatmap: {heatmap_file}\n"
        )


# ── Grad-CAM ──────────────────────────────────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_colors = cm.ScalarMappable(cmap="jet").cmap(np.arange(256))[:, :3]
    jet_heatmap = (jet_colors[heatmap_uint8] * 255).astype(np.uint8)
    jet_bgr = cv2.cvtColor(jet_heatmap, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(jet_bgr, alpha, img, 1 - alpha, 0)


# ── Webcam thread ─────────────────────────────────────────────────────────

def run_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = cv2.resize(gray[y:y+h, x:x+w], (224, 224))
            roi_rgb  = cv2.merge([roi_gray, roi_gray, roi_gray]).astype('float32') / 255.0
            roi_input = np.expand_dims(roi_rgb, axis=0)

            preds      = classifier.predict(roi_input, verbose=0)[0]
            label      = CLASS_LABELS[preds.argmax()]
            confidence = float(preds.max())

            # Grad-CAM
            heatmap     = make_gradcam_heatmap(roi_input, classifier, "conv_pw_13_relu")
            face_color  = cv2.cvtColor(cv2.resize(roi_gray, (224, 224)), cv2.COLOR_GRAY2BGR)
            gradcam_img = overlay_gradcam(face_color, heatmap)
            cv2.imshow("Grad-CAM Heatmap", gradcam_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            gps           = get_gps_coordinates()
            heatmap_file  = None
            alert_fired   = False

            if confidence > CONFIDENCE_THRESHOLD and label in ALERT_EMOTIONS:
                heatmap_file = (
                    f"{HEATMAP_DIR}/{label}_{confidence:.2f}_"
                    f"{gps['lat']}_{gps['lon']}.png"
                )
                cv2.imwrite(heatmap_file, gradcam_img)
                trigger_alert(label, confidence, gps, heatmap_file)
                alert_fired = True

            # Update shared state
            with detection_lock:
                latest_detection.update({
                    "emotion": label,
                    "confidence": round(confidence, 4),
                    "gps": gps,
                    "heatmap_file": heatmap_file,
                    "timestamp": datetime.now().isoformat(),
                    "alert_triggered": alert_fired
                })

        cv2.imshow("Emotion Detector + GPS Alerts", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── REST API endpoints ────────────────────────────────────────────────────

@app.route('/api/detection/latest', methods=['GET'])
def get_latest():
    """
    GET /api/detection/latest
    Returns the most recent emotion detection result.
    """
    with detection_lock:
        data = dict(latest_detection)
    if data["emotion"] is None:
        return jsonify({"message": "No detection yet. Is the webcam running?"}), 404
    return jsonify(data)


@app.route('/api/detection/predict', methods=['POST'])
def predict_image():
    """
    POST /api/detection/predict
    Body: { "image": "<base64-encoded JPEG/PNG>" }
    Returns emotion prediction without needing the webcam.
    """
    body = request.get_json()
    if not body or "image" not in body:
        return jsonify({"error": "Provide 'image' as a base64 string"}), 400

    try:
        img_bytes = base64.b64decode(body["image"])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"message": "No face detected in the image"}), 200

        results = []
        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (224, 224))
            roi_rgb = cv2.merge([roi, roi, roi]).astype('float32') / 255.0
            preds = classifier.predict(np.expand_dims(roi_rgb, 0), verbose=0)[0]
            results.append({
                "emotion": CLASS_LABELS[preds.argmax()],
                "confidence": round(float(preds.max()), 4),
                "all_scores": {
                    CLASS_LABELS[i]: round(float(preds[i]), 4)
                    for i in range(len(CLASS_LABELS))
                },
                "gps": get_gps_coordinates()
            })

        return jsonify({"detections": results, "face_count": len(results)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """
    GET /api/alerts
    Returns all logged alerts from alerts_log.txt.
    """
    alerts = []
    try:
        with open(ALERT_LOG, "r") as f:
            for i, line in enumerate(f.readlines(), start=1):
                if "GPS" not in line:
                    continue
                parts    = line.split(",")
                emotion  = parts[0].replace("Emotion: ", "").strip()
                conf     = float(parts[1].replace("Confidence: ", "").strip())
                gps_str  = line.split("GPS: ")[-1].split(", Heatmap")[0].strip()
                heatmap  = line.split("Heatmap:")[-1].strip()
                alerts.append({
                    "id": i,
                    "emotion": emotion,
                    "confidence": conf,
                    "gps": eval(gps_str),
                    "heatmap_file": heatmap
                })
    except FileNotFoundError:
        pass
    return jsonify({"alerts": alerts, "total": len(alerts)})


@app.route('/api/alerts/clear', methods=['DELETE'])
def clear_alerts():
    """
    DELETE /api/alerts/clear
    Clears the alerts log file.
    """
    open(ALERT_LOG, "w").close()
    return jsonify({"message": "Alert log cleared."})


@app.route('/api/heatmap/<filename>', methods=['GET'])
def get_heatmap(filename):
    """
    GET /api/heatmap/<filename>
    Returns a saved Grad-CAM heatmap image.
    """
    path = os.path.join(HEATMAP_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype='image/png')


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Start webcam in a background thread so the API stays responsive
    webcam_thread = threading.Thread(target=run_webcam, daemon=True)
    webcam_thread.start()

    print("Emotion Detection API running at http://localhost:5000")
    print("  GET  /api/detection/latest       — latest webcam result")
    print("  POST /api/detection/predict      — predict from base64 image")
    print("  GET  /api/alerts                 — all triggered alerts")
    print("  DELETE /api/alerts/clear         — clear alert log")
    print("  GET  /api/heatmap/<filename>     — fetch a heatmap image")
    app.run(debug=False, port=5000)
