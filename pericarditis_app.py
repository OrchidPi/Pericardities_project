import joblib
import streamlit as st
import pandas as pd
from PIL import Image
import torch
import cv2
import numpy as np
import json
from easydict import EasyDict as edict
from model.classifier_vit import VIT
import sys
import os
import io
import shap
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper
from streamlit_cropperjs import st_cropperjs
import tempfile
import gdown


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
st.title("Pericarditis Risk Calculator")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Demographic Features")
    gender = st.radio("Gender", ["Male", "Female", "Unknown"], index=2)
    male = 1 if gender == "Male" else 0
    female = 1 if gender == "Female" else 0

    age = st.number_input("Enter Age", min_value=0, max_value=120, step=1, value=40)

    # Default all age groups to 0
    age_lt_40, age_40_80, age_gt_80 = 0, 0, 0
    if age:
        if age < 40:
            age_lt_40 = 1
        elif age <= 80:
            age_40_80 = 1
        else:
            age_gt_80 = 1


with col2:
    st.subheader("Clinical History")
    cancer = st.radio("Cancer", ["Yes", "No", "Unknown"], index=2)
    autoimmune = st.radio("Autoimmune Disease", ["Yes", "No", "Unknown"], index=2)
    af = st.radio("Atrial Fibrillation (AF)", ["Yes", "No", "Unknown"], index=2)
    smoking = st.radio("Smoking", ["Yes", "Never", "Unknown"], index=2)
    hyperlipidemia = st.radio("Hyperlipidemia", ["Yes", "No", "Unknown"], index=2)
    cad = st.radio("Coronary Artery Disease (CAD)", ["Yes", "No", "Unknown"], index=2)
    
with col3:
    st.subheader(" ")
    diabetes = st.radio("Diabetes", ["Yes", "No", "Unknown"], index=2)
    hypertension = st.radio("Hypertension", ["Yes", "No", "Unknown"], index=2)
    mi = st.radio("Myocardial Infarction (MI)", ["Yes", "No", "Unknown"], index=2)
    stroke = st.radio("Stroke", ["Yes", "No", "Unknown"], index=2)
    hf = st.radio("Heart Failure (HF)", ["Yes", "No", "Unknown"], index=2)


def map_value(x):
    return {"Yes": 1, "No": 0, "Never": 0, "Unknown": -1}[x]

def pad_to_square(image, target_size):
    h, w, c = image.shape
    target_h, target_w = target_size
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    image_padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                          mode='constant', constant_values=255)

    return image_padded

input_dict = {
    'Hypertension': map_value(hypertension),
    'Hyperlipidemia': map_value(hyperlipidemia),
    'Diabetes': map_value(diabetes),
    'CAD': map_value(cad),
    'HF': map_value(hf),
    'AF': map_value(af),
    'MI': map_value(mi),
    'Smoking': map_value(smoking),
    'Stroke': map_value(stroke),
    'Cancer': map_value(cancer),
    'Autoimmune': map_value(autoimmune),
    'Male': male,
    'Female': female,
    'Age < 40': age_lt_40,
    'Age 40-80': age_40_80,
    'Age > 80': age_gt_80
}

ckpt_path = "checkpoints/ECG_model.ckpt"
ckpt_url = "https://drive.google.com/uc?id=1HzCghcteqo7OG_DBjiGZCFepSprTe-Pf"

# Ensure checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# Download only if not already downloaded
if not os.path.exists(ckpt_path):
    gdown.download(ckpt_url, ckpt_path, quiet=False)

with open('./cfg.json') as f:
    cfg = edict(json.load(f))
    ECG_model = VIT(cfg)
    ckpt = torch.load(ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=False)
    ECG_model.load_state_dict(ckpt['state_dict'], strict=False)


col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"], key="uploaded_pic")

with col2:
    st.markdown("##### Reference Format")
    st.image("ecg_reference.png", caption="Expected Format")

# After file is uploaded, show cropper
if uploaded_file is not None:
    st.markdown("### ✂️ Please crop the ECG image to remove text or device info (keep waveform area only)")

    # Read file bytes
    image_bytes = uploaded_file.read()

    # Use st_cropperjs (JavaScript-based)
    cropped_image = st_cropperjs(pic=image_bytes, btn_text="Confirm Crop", key="ecg_cropperjs")

    if cropped_image is not None:
        st.image(cropped_image, caption="Cropped ECG")

    image = np.array(cropped_img)
    h, w, _ = image.shape

    # Continue preprocessing as before
    intermediate_size = (w, w)
    final_size = (512, 512)
    padded_image = pad_to_square(image, intermediate_size)
    final_image = cv2.resize(padded_image, final_size, interpolation=cv2.INTER_AREA)

    # Normalize and convert to tensor
    max_v = final_image.max()
    min_v = final_image.min()
    norm_image = (final_image - min_v) / (max_v - min_v)
    norm_image = norm_image.transpose(2, 0, 1)
    image_tensor = torch.tensor(norm_image, dtype=torch.float32).unsqueeze(0)

    st.success("✅ Cropped image ready for prediction.")

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict Pericarditis Risk"):
        tabular_model = joblib.load("random_forest_pericarditis_model.pkl")
        tabular_model_calibration = joblib.load("calibrated_random_forest_model.pkl")
        # explainer = shap.TreeExplainer(tabular_model)
        explainer = shap.TreeExplainer(tabular_model)

    
        # Tabular model
        tabular_prob = tabular_model.predict_proba(input_df)[0][1]
        shap_values = explainer.shap_values(input_df)

        plt.figure()
        shap.force_plot(
            explainer.expected_value[0],
            shap_values[:,:,0][0],
            input_df.iloc[0],
            matplotlib=True,
            show=False
        )

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)

        st.image(buf, caption="SHAP Force Plot for Tabular Model")


        # ECG model
        with torch.no_grad():
            ecg_output, _ = ECG_model(image_tensor)
            ecg_prob = torch.sigmoid(ecg_output.view(-1)).cpu().numpy()[0]

        calibrator_ecg = joblib.load("ecg_logit_calibrator.pkl")
        calibrated_ecg_probs = calibrator_ecg.predict_proba(ecg_output.reshape(-1, 1))[:, 1]

        # Fusion model
        fusion_model = joblib.load("calibrated_fusion_model.pkl")
        tabular_prob_calibration = tabular_model_calibration.predict_proba(input_df)[0][1]
        fusion_input = np.column_stack([tabular_prob_calibration, calibrated_ecg_probs])[0].reshape(1, -1)
        fusion_prob = fusion_model.predict_proba(fusion_input)[0][1]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pericarditis Risk by Tabular Model", f"{tabular_prob_calibration*100:.2f}%")
            st.write(f"Optimal Threshold: 13.13%")
            st.write(f"Positive predictive value (PPV): 27.54%")
        with col2:
            st.metric("Pericarditis Risk by ECG Model", f"{calibrated_ecg_probs[0]*100:.2f}%")
            st.write(f"Optimal Threshold: 13.44%")
            st.write(f"Positive predictive value (PPV): 33.80%")
        with col3:
            st.metric("Pericarditis Risk by Fusion Model", f"{fusion_prob*100:.2f}%")
            st.write(f"Optimal Threshold: 9.72%")
            st.write(f"Positive predictive value (PPV): 34.84%")

