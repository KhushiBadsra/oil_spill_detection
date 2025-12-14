import os
import cv2
import gdown
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Oil Spill Segmentation (U-Net)",
    page_icon="🌊",
    layout="wide"
)

# ---------------- MODEL SETTINGS ----------------
IMG_SIZE = 256
DEFAULT_THRESHOLD = 0.5

MODEL_DRIVE_MAP = {
    "U-Net Baseline (Keras)": {
        "file_id": "1d3yDvwJBr_hVkpJ1Lb2pifSQLeW4xxQA", 
        "local_path": "unet_model.h5"
    }
}

# ---------------- LOAD MODEL FROM DRIVE ----------------
@st.cache_resource
def load_keras_model(model_key):
    info = MODEL_DRIVE_MAP[model_key]

    if not os.path.exists(info["local_path"]):
        url = f"https://drive.google.com/uc?id={info['file_id']}"
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, info["local_path"], quiet=False)

    model = load_model(info["local_path"], compile=False)
    return model

# ---------------- PREPROCESS (GRAYSCALE – IMPORTANT) ----------------
def preprocess_image(image):
    # Model trained on 1-channel images
    img = image.resize((IMG_SIZE, IMG_SIZE)).convert("L")
    img = np.array(img).astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    img = np.expand_dims(img, axis=0)   # (1, H, W, 1)

    return img

# ---------------- POST PROCESS ----------------
def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ---------------- PREDICTION ----------------
def predict(image, model, threshold):
    original = np.array(image)
    h, w = original.shape[:2]

    x = preprocess_image(image)

    pred = model.predict(x, verbose=0)

    # Safe output handling
    if pred.ndim == 4:      # (1, H, W, 1)
        pred = pred[0]
    if pred.ndim == 3:      # (H, W, 1)
        pred = pred[:, :, 0]

    prob = cv2.resize(pred, (w, h))
    mask = (prob > threshold).astype(np.uint8)
    mask = clean_mask(mask)

    overlay = original.copy()
    overlay[mask == 1] = [255, 165, 0]  # ORANGE overlay
    blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

    oil_pct = (mask.sum() / mask.size) * 100
    return mask, blended, oil_pct

# ---------------- UI ----------------
st.title("🌊 Oil Spill Segmentation System")
st.caption("U-Net based oil spill detection (TensorFlow / Keras)")

model_name = st.selectbox(
    "Select Segmentation Model",
    list(MODEL_DRIVE_MAP.keys())
)

model = load_keras_model(model_name)

left, right = st.columns([1, 1])

with left:
    st.subheader("Upload Satellite Image")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    threshold = st.slider(
        "Detection Threshold",
        0.1, 0.9,
        DEFAULT_THRESHOLD,
        0.05
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input Image", use_container_width=True)

        if st.button("🔍 Run Segmentation"):
            mask, overlay, oil_pct = predict(image, model, threshold)
            st.session_state.result = (mask, overlay, oil_pct)

with right:
    st.subheader("Results")

    if "result" in st.session_state:
        mask, overlay, oil_pct = st.session_state.result

        st.image(overlay, caption="Oil Spill Overlay", use_container_width=True)
        st.image(mask * 255, caption="Predicted Mask", clamp=True)

        st.markdown("### 📊 Analysis")
        st.write(f"🛢 **Oil Coverage:** {oil_pct:.2f}%")
        st.write(f"⚙ **Threshold:** {threshold}")

        if oil_pct > 1:
            st.error("🚨 Oil Spill Detected")
        else:
            st.success("✅ No Significant Spill Detected")

st.markdown("---")
st.caption("Developed by Khushi | Streamlit + TensorFlow + U-Net")
