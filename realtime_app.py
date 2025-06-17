import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# ==== Load Model dan Scaler ====
model = joblib.load("model/svm_model.pkl")
scaler = joblib.load("model/scaler.pkl")
class_labels = model.classes_.tolist()

# ==== Preprocessing & Ekstraksi Fitur ====

def adjust_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 10)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def convertToHSV_withMask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 20), (180, 255, 255))
    hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    return hsv_masked

def extract_color_moments(image_hsv_masked):
    features = []
    for channel in cv2.split(image_hsv_masked):
        channel = channel.flatten().astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean((channel - mean) ** 3) / (std**3 + 1e-10)
        features.extend([mean, std, skewness])
    return features

def fullPreprocessingHu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return morph

def extract_hu_moments(processed_image, min_area_threshold=50, top_n=3):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_threshold]
    if len(contours) == 0:
        return [0.0] * 7
    contour_areas = [cv2.contourArea(c) for c in contours]
    mean_area = np.mean(contour_areas)
    closest = sorted(contours, key=lambda c: abs(cv2.contourArea(c) - mean_area))[:top_n]
    hu_features = []
    for c in closest:
        moments = cv2.moments(c)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        hu_features.append(hu_log)
    return np.mean(hu_features, axis=0).tolist()

def predict_rice(image):
    resized = cv2.resize(image, (500, 500))
    adjusted = adjust_background(resized)
    hsv_masked = convertToHSV_withMask(adjusted)
    color_features = extract_color_moments(hsv_masked)
    hu_input = fullPreprocessingHu(adjusted)
    hu_features = extract_hu_moments(hu_input)
    final_features = np.array(color_features + hu_features).reshape(1, -1)
    scaled = scaler.transform(final_features)
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    prob_dict = {label: float(f"{prob*100:.2f}") for label, prob in zip(class_labels, probabilities)}
    return prediction, prob_dict

# ==== Streamlit UI ====

st.set_page_config(page_title="Klasifikasi Jenis Beras", layout="centered")
st.title("📁 Deteksi Jenis Beras dengan Upload Gambar")

uploaded = st.file_uploader("Upload Citra Beras", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Gambar yang Diunggah", width=300)

    pred, probs = predict_rice(image)

    st.success(f"✅ Jenis Beras: {pred}")
    st.metric("Probabilitas Tertinggi", f"{max(probs.values()):.2f}%")

    st.markdown("### Distribusi Probabilitas:")
    st.dataframe(pd.DataFrame(probs.items(), columns=["Jenis Beras", "Probabilitas (%)"]))
