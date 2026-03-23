import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.cm as cm
import random, os

# -------------------------------
# Load model and Haar Cascade
# -------------------------------
model_path = "final_emotion_detection_model.keras"
cascade_path = "haarcascade_frontalface_default.xml"

classifier = load_model(model_path)
face_classifier = cv2.CascadeClassifier(cascade_path)

# Emotion labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -------------------------------
# GPS Simulation
# -------------------------------
def get_gps_coordinates():
    lat = round(random.uniform(12.9000, 13.1000), 6)
    lon = round(random.uniform(77.5000, 77.7000), 6)
    return {"lat": lat, "lon": lon}

# -------------------------------
# Alert Function
# -------------------------------
def trigger_alert(emotion, confidence, gps, heatmap_file):
    print("\n🚨 ALERT TRIGGERED 🚨")
    print(f"Emotion: {emotion} | Confidence: {confidence:.2f}")
    print(f"Child Location → Latitude: {gps['lat']} , Longitude: {gps['lon']}")
    print(f"📷 Heatmap saved: {heatmap_file}\n")

    # Save log with heatmap reference
    with open("alerts_log.txt", "a") as f:
        f.write(f"Emotion: {emotion}, Confidence: {confidence:.2f}, GPS: {gps}, Heatmap: {heatmap_file}\n")

# -------------------------------
# Grad-CAM Functions
# -------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    jet = cm.ScalarMappable(cmap="jet").cmap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.cvtColor((jet_heatmap * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(jet_heatmap, alpha, img, 1 - alpha, 0)
    return superimposed_img

# -------------------------------
# Webcam Loop
# -------------------------------
cap = cv2.VideoCapture(0)

# Ensure folder exists
if not os.path.exists("gradcam_alerts"):
    os.makedirs("gradcam_alerts")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)

        # Convert grayscale → RGB
        roi_rgb = cv2.merge([roi_gray, roi_gray, roi_gray])
        roi_rgb = roi_rgb.astype('float') / 255.0
        roi_rgb = np.expand_dims(roi_rgb, axis=0)

        # Prediction
        preds = classifier.predict(roi_rgb)[0]
        label = class_labels[preds.argmax()]
        confidence = preds.max()

        # Grad-CAM
        heatmap = make_gradcam_heatmap(roi_rgb, classifier, last_conv_layer_name="conv_pw_13_relu")
        face_color = cv2.cvtColor(cv2.resize(roi_gray, (224, 224)), cv2.COLOR_GRAY2BGR)
        gradcam_img = overlay_gradcam(face_color, heatmap)
        cv2.imshow("Grad-CAM Heatmap", gradcam_img)

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 🚨 Alert condition
        if confidence > 0.40 and label in ["Sad", "Fear", "Angry"]:
            gps = get_gps_coordinates()
            filename = f"gradcam_alerts/{label}_{confidence:.2f}_{gps['lat']}_{gps['lon']}.png"
            cv2.imwrite(filename, gradcam_img)
            trigger_alert(label, confidence, gps, filename)

    cv2.imshow("Emotion Detector + GPS Alerts", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
