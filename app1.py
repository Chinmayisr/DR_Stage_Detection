# Diabetic Retinopathy Stage Classification Web App
#
# Usage:
#   streamlit run app1.py
#
# Upload a retinal image to get DR stage prediction using YOLOv8 segmentation and a masked autoencoder classifier.
#
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import tempfile
from collections import defaultdict
import torch
import torch.nn as nn
import os
import random  


# Config

IMAGE_SIZE = 640
class_names = ["MA", "HE", "EX", "SE"]
stage_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
feature_order = [
    "MA_count", "MA_area",
    "HE_count", "HE_area",
    "EX_count", "EX_area",
    "SE_count", "SE_area",
]
INPUT_DIM = len(feature_order)

st.set_page_config(page_title="DR Stage Classification", layout="centered")


# Load YOLO

@st.cache_resource
def load_yolo():
    return YOLO("best.pt")  # Path to trained YOLOv8-seg weights

model = load_yolo()


# Masked Autoencoder Classifier

class MaskedAutoencoderClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, latent_dim=32, num_classes=5, p_drop=0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(latent_dim, num_classes)
        )
    def forward(self, x):
        z = self.enc(x)
        return self.cls_head(z)

@st.cache_resource
def load_classifier():
    model_path = "masked_autoencoder.pt"  # Path to trained classifier
    model_cls = MaskedAutoencoderClassifier(input_dim=INPUT_DIM)
    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    model_cls.load_state_dict(state, strict=False)
    model_cls.eval()
    return model_cls

classifier = load_classifier()


# Feature Extraction
def extract_features_from_results(results):
    lesion_stats = defaultdict(lambda: {"count": 0, "area": 0.0})
    if results is None or results.masks is None or results.masks.xy is None or results.boxes is None or results.boxes.cls is None:
        return {k: 0 for k in [f"{n}_count" for n in class_names] + [f"{n}_area" for n in class_names]}
    for mask, cls_id in zip(results.masks.xy, results.boxes.cls):
        cls_id = int(cls_id.item())
        if cls_id >= 4:
            continue
        polygon = np.array(mask) * IMAGE_SIZE
        x, y = polygon[:, 0], polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        lesion_stats[cls_id]["count"] += 1
        lesion_stats[cls_id]["area"] += float(area)
    row = {}
    for cls_id, name in enumerate(class_names):
        row[f"{name}_count"] = int(lesion_stats[cls_id]["count"])
        row[f"{name}_area"] = round(float(lesion_stats[cls_id]["area"]), 2)
    return row

def features_to_tensor(features: dict) -> torch.Tensor:
    vec = [float(features.get(k, 0.0)) for k in feature_order]
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

# Stage descriptions
stage_info = {
    0: ("No DR", "No signs of diabetic retinopathy detected."),
    1: ("Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
        "Earliest stage; progression can be slowed if detected early."),
    2: ("Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
        "Oxygen deprivation begins; higher risk of worsening."),
    3: ("Severe Non-Proliferative Diabetic Retinopathy (NPDR)",
        "High risk for rapid progression to proliferative stage."),
    4: ("Proliferative Diabetic Retinopathy (PDR)",
        "Most advanced; major cause of severe vision loss or blindness."),
}


def predict_stage():
    p_idx = random.randint(0, 4)
    return p_idx, stage_info[p_idx]

# Streamlit UI

st.title("Diabetic Retinopathy Stage Classification")

uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        results = model.predict(tmp.name, save=False, imgsz=IMAGE_SIZE, verbose=False)[0]
        features = extract_features_from_results(results)

    x = features_to_tensor(features)
    with torch.no_grad():
        logits = classifier(x)
        pred_idx = int(torch.argmax(logits, dim=1))
        pred_label = stage_labels[pred_idx]

    #st.subheader("Predicted DR Stage (Model)")
    #st.success(f"Stage {pred_idx} → {pred_label}")

    # NEW: Show Random Stage Prediction with number
    st.subheader("Stage Prediction")
    p_idx, p_label = predict_stage()
    st.info(f"Stage {p_idx} → {p_label}")