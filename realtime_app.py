import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# ==== Konfigurasi halaman ====
st.set_page_config(page_title="Klasifikasi Jenis Beras", layout="centered")

# ==== Styling CSS ====
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==== Load model dan scaler ====
model = joblib.load("model/svm_model.pkl")
scaler = joblib.load("model/scaler.pkl")
class_labels = model.classes_.tolist()

# ==== Fungsi preprocessing ====

def adjust_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 10)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

def convertToHSV_withMask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 20), (180, 255, 255))
    return cv2.bitwise_and(hsv, hsv, mask=mask)

def extract_color_moments(image_hsv_masked):
    features = []
    for channel in cv2.split(image_hsv_masked):
        c = channel.flatten().astype(np.float32)
        mean = np.mean(c)
        std = np.std(c)
        skew = np.mean((c - mean) ** 3) / (std**3 + 1e-10)
        features.extend([mean, std, skew])
    return features

def fullPreprocessingHu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def extract_hu_moments(processed_image, min_area_threshold=50, top_n=3):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area_threshold]
    if not valid:
        return [0.0] * 7
    avg_area = np.mean([cv2.contourArea(c) for c in valid])
    closest = sorted(valid, key=lambda c: abs(cv2.contourArea(c) - avg_area))[:top_n]
    hu_feats = []
    for c in closest:
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu_feats.append(-np.sign(hu) * np.log10(np.abs(hu) + 1e-10))
    return np.mean(hu_feats, axis=0).tolist()

def predict_rice(image):
    resized = cv2.resize(image, (500, 500))
    bg = adjust_background(resized)
    hsv = convertToHSV_withMask(bg)
    color_feats = extract_color_moments(hsv)
    hu_feats = extract_hu_moments(fullPreprocessingHu(bg))
    final = scaler.transform([color_feats + hu_feats])
    prediction = model.predict(final)[0]
    probs = model.predict_proba(final)[0]
    prob_dict = {label: float(f"{p*100:.2f}") for label, p in zip(class_labels, probs)}
    return prediction, prob_dict

# ==== UI ====

st.markdown("""
<div class="title-section">
    <h3>Klasifikasi Jenis Beras menggunakan SVM</h3>
</div>
<h5>Upload Citra Beras</h5>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.markdown("<div class='center-content'>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Gambar yang Diunggah", width=300)

    klasifikasi = st.button("Klasifikasi Gambar", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    if klasifikasi:
        pred, probs = predict_rice(image)
        st.markdown(f"<div class='result-box'><strong>Jenis Beras:</strong> {pred}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='accuracy'><strong>Akurasi Prediksi:</strong> {max(probs.values()):.2f}%</div>", unsafe_allow_html=True)
        st.markdown("<h4 class='centered-text'>Distribusi Probabilitas:</h4>", unsafe_allow_html=True)
        df = pd.DataFrame(list(probs.items()), columns=["Jenis Beras", "Probabilitas (%)"])
        st.dataframe(df, use_container_width=True)
