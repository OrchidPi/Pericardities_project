import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import torch
import cv2
import numpy as np
import json
from easydict import EasyDict as edict
from model.classifier_vit import VIT
import sys
import os

# Add the root directory (the one containing 'model') to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


st.title("Pericarditis Risk Calculator")

st.subheader("Gender")
gender = st.radio("Gender", ["Male", "Female"])
male = 1 if gender == "Male" else 0
female = 1 if gender == "Female" else 0

st.subheader("Age Group")
age_group = st.radio("Age Group", ["<40", "40–80", ">80"])
age_lt_40 = 1 if age_group == "<40" else 0
age_40_80 = 1 if age_group == "40–80" else 0
age_gt_80 = 1 if age_group == ">80" else 0

st.subheader("Behavioral")
smoking = st.radio("Smoking", ["Yes", "No"])

st.subheader("Comorbidities")
hypertension = st.radio("Hypertension", ["Yes", "No"])
hyperlipidemia = st.radio("Hyperlipidemia", ["Yes", "No"])
diabetes = st.radio("Diabetes", ["Yes", "No"])
cad = st.radio("Coronary Artery Disease (CAD)", ["Yes", "No"])
hf = st.radio("Heart Failure (HF)", ["Yes", "No"])
af = st.radio("Atrial Fibrillation (AF)", ["Yes", "No"])
mi = st.radio("Myocardial Infarction (MI)", ["Yes", "No"])
stroke = st.radio("Stroke", ["Yes", "No"])
cancer = st.radio("Cancer", ["Yes", "No"])
autoimmune = st.radio("Autoimmune Disease", ["Yes", "No"])

def map_value(x):
    return {"Yes": 1, "No": 0}[x]

def pad_to_square(image, target_size):
    h, w, c = image.shape
    target_h, target_w = target_size

    # Compute required padding for height (Top & Bottom)
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top

    # Compute required padding for width (Left & Right) (should be 0 in your case)
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    # Apply padding
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


with open('/media/Datacenter_storage/jialu/003/Rimita_project/ECG_encoder/logdir_ecg_final/' + 'cfg.json') as f:
    cfg = edict(json.load(f))
    ECG_model = VIT(cfg)
    ckpt_path = os.path.join('/media/Datacenter_storage/jialu/003/Rimita_project/ECG_encoder/logdir_ecg_final/', 'best2.ckpt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ECG_model.module.load_state_dict(ckpt['state_dict'], strict=False)

input_df = pd.DataFrame([input_dict])

uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)

    h, w, c = image.shape
    intermediate_size = (w, w)
    final_size = (512, 512)
    padded_image = pad_to_square(image, intermediate_size)
    final_image = cv2.resize(padded_image, final_size, interpolation=cv2.INTER_AREA)

    # Normalize and convert to tensor
    max_v = final_image.max()
    min_v = final_image.min()
    norm_image = (final_image - min_v) / (max_v - min_v)
    norm_image = norm_image.transpose(2, 0, 1)
    image_tensor = torch.tensor(norm_image, dtype=torch.float32).unsqueeze(0)  # Add batch dim

    if st.button("Predict Pericarditis Risk"):
        tabular_model = joblib.load("random_forest_pericarditis_model.pkl")
        fusion_model = joblib.load("LogisticRegression_model.pkl")

        # Predict with tabular model
        tabular_prob = tabular_model.predict_proba(input_df)[0][1]

        # ECG model
        with torch.no_grad():
            ecg_output, _ = ECG_model(image_tensor)
            ecg_prob = torch.sigmoid(ecg_output.view(-1)).cpu().numpy()[0]

        # Fusion prediction
        fusion_input = np.array([[tabular_prob, ecg_prob]])
        fusion_prob = fusion_model.predict_proba(fusion_input)[0][1]

        st.metric("Risk by Tabular Model", f"{tabular_prob*100:.2f}%")
        st.metric("Risk by ECG Model", f"{ecg_prob*100:.2f}%")
        st.metric("Risk by Fusion Model", f"{fusion_prob*100:.2f}%")


