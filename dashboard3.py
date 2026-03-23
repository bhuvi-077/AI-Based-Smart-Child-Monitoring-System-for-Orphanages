import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
from PIL import Image
import base64

st.set_page_config(page_title="Caregiver Dashboard", layout="wide")
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Call background
add_bg_from_local("image3.jpg") 
st.title("👩‍👧 Caregiver Dashboard - Child Monitoring System")

# -------------------------------
# Load Alerts Log
# -------------------------------
def load_alerts(log_file="alerts_log.txt"):
    alerts = []
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines, start=1):
            if "GPS" in line:
                parts = line.split(",")
                emotion = parts[0].replace("Emotion: ", "").strip()
                conf = parts[1].replace("Confidence: ", "").strip()
                gps_str = line.split("GPS: ")[-1].split(", Heatmap")[0].strip()
                gps_dict = eval(gps_str)
                heatmap_file = line.split("Heatmap:")[-1].strip()
                alerts.append({
                    "child_id": f"Child-{i}",
                    "emotion": emotion,
                    "confidence": float(conf),
                    "lat": gps_dict["lat"],
                    "lon": gps_dict["lon"],
                    "heatmap": heatmap_file
                })
    except FileNotFoundError:
        st.warning("⚠️ No alerts_log.txt file found. Run Module 3/5 first to generate alerts.")
    return alerts

alerts = load_alerts()

# -------------------------------
# Latest Alert
# -------------------------------
if alerts:
    last_alert = alerts[-1]
    st.subheader("📌 Latest Alert")
    st.success(f"🧒 {last_alert['child_id']} → Emotion: {last_alert['emotion']} "
               f"(Confidence: {last_alert['confidence']:.2f}) "
               f"| GPS: ({last_alert['lat']}, {last_alert['lon']})")

    # Alert History Table
    st.subheader("📊 Alert History")
    df = pd.DataFrame(alerts)
    st.dataframe(df)

    # GPS Map
    st.subheader("🗺️ Child Location Map")
    m = folium.Map(location=[last_alert["lat"], last_alert["lon"]], zoom_start=15)
    folium.Marker([last_alert["lat"], last_alert["lon"]],
                  popup=f"{last_alert['child_id']} - {last_alert['emotion']}").add_to(m)
    st_folium(m, width=700, height=400)

    # Heatmap Gallery
    st.subheader("📷 Grad-CAM Heatmap Gallery")
    heatmap_dir = "gradcam_alerts"
    if os.path.exists(heatmap_dir):
        images = os.listdir(heatmap_dir)
        if images:
            cols = st.columns(3)
            for i, img_file in enumerate(images[-9:]):  # last 9 images
                img_path = os.path.join(heatmap_dir, img_file)
                img = Image.open(img_path)
                with cols[i % 3]:
                    st.image(img, caption=img_file, use_container_width=True)
        else:
            st.info("No heatmaps generated yet.")
    else:
        st.info("Heatmap folder does not exist.")

else:
    st.info("✅ No alerts available yet. The system is running normally.")
