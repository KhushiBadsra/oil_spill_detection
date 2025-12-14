import os
import io
import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import streamlit as st
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown

# Set matplotlib cache directory to avoid import issues
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_cache')

import matplotlib.pyplot as plt

# App configuration
st.set_page_config(
    page_title="U-Net Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

# Model settings
DEVICE = torch.device("cpu")
IMG_SIZE = (256, 256)
DEFAULT_THRESHOLD = 0.5

FILE_ID = "1-MkZMXNjh2kHSPgh7-FOZ505iSZbxVmo"
MODEL_PATH = "best_model.pth"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    with st.spinner("⬇️ Downloading model from Google Drive..."):
        gdown.download(url, MODEL_PATH, quiet=False)


# U-Net building block with two conv layers
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

# Main U-Net model with encoder-decoder architecture
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder: upsampling path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path with skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# Image preprocessing for the model
base_transform = A.Compose([
    A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# Load the trained U-Net model
@st.cache_resource
def load_unet_model():
    try:
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at '{MODEL_PATH}'. Please place the model in the correct directory.")
            return None
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ U-Net model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

# Run inference and create visualization
@torch.no_grad()
def predict_and_visualize(model, image, threshold):
    try:
        original = np.array(image.convert("RGB"))
        oh, ow = original.shape[:2]

        # Preprocess
        transformed = base_transform(image=original)
        x = transformed["image"].unsqueeze(0).to(DEVICE)

        # Inference
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Resize + mask
        prob_resized = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = (prob_resized >= float(threshold)).astype(np.uint8)

        # Overlay
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay[mask == 1] = [255, 0, 0]
        blended = cv2.addWeighted(
            overlay, 0.7,
            cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 0.3, 0
        )

        # Stats
        total_px = mask.size
        oil_px = int(mask.sum())
        oil_pct = 100.0 * oil_px / max(total_px, 1)

        conf_max = float(prob_resized.max())
        conf_mean = float(prob_resized[mask == 1].mean()) if oil_px > 0 else 0.0

        # -------- SINGLE CLEAN FIGURE --------
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(blended)
        ax.set_title("Oil Spill Detection Overlay", fontsize=14, fontweight="bold")
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        result_image = Image.open(buf)

        severity = (
            "HIGH" if oil_pct > 10 else
            "MODERATE" if oil_pct > 5 else
            "LOW" if oil_pct > 1 else
            "MINIMAL"
        )

        status_icon = "🚨" if oil_pct > 1 else "✅"
        status_text = "OIL SPILL DETECTED" if oil_pct > 1 else "No significant oil detected"

        results_text = f"""
        | Metric | Value |
        | :--- | :--- |
        | **Status** | **{status_icon} {status_text}** |
        | **Severity Level** | {severity} |
        | **Oil Coverage Area** | {oil_pct:.2f}% |
        | **Detected Pixels** | {oil_px:,} |
        | **Total Pixels** | {total_px:,} |
        | **Peak Confidence** | {conf_max:.3f} |
        | **Avg Spill Confidence** | {conf_mean:.3f} |
        | **Image Resolution** | {ow} × {oh} |
        """

        return result_image, results_text

    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")
        return None, None


# Main application interface
st.title("🌊 Marine Oil Spill AI")
st.caption("Satellite-based oil spill detection using deep learning")

model = load_unet_model()

if model:
    col1, col2 = st.columns([1.1, 1.4])

    # ---------------- LEFT PANEL ----------------
    with col1:
        st.markdown("### 📤 Upload Satellite Image")

        uploaded_file = st.file_uploader(
            "Supported formats: JPG, PNG",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🚀 Run AI Detection", use_container_width=True):
                with st.spinner("AI is analyzing ocean surface..."):
                    result_plot, result_stats = predict_and_visualize(
                        model,
                        input_image,
                        DEFAULT_THRESHOLD   # SAME LOGIC
                    )

                if result_plot:
                    st.session_state.result_plot = result_plot
                    st.session_state.result_stats = result_stats
                    st.session_state.done = True
                    st.rerun()


    # ---------------- RIGHT PANEL ----------------
    with col2:
        st.markdown("### 📊 Detection Result")

        if "done" in st.session_state:
            stats = st.session_state.result_stats
            oil_pct = float(stats.split("Oil Coverage")[1].split("%")[0].split()[-1])

            if oil_pct > 1:
                st.markdown(
                    "<div style='padding:14px;border-radius:12px;background:#ffe5e5;color:#b00020;font-weight:600;'>"
                    "🚨 Oil Spill Detected</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='padding:14px;border-radius:12px;background:#e8f5e9;color:#1b5e20;font-weight:600;'>"
                    "✅ No Significant Oil Spill Found</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.image(st.session_state.result_plot, use_container_width=True)

            with st.expander("📈 Detailed Analysis"):
                st.markdown(st.session_state.result_stats)

        else:
            st.info("Upload an image and run detection to view results.")

else:
    st.error("Model could not be loaded.")
