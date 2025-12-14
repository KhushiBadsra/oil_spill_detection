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
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🌊",
    layout="wide"
)

DEVICE = torch.device("cpu")
IMG_SIZE = (256, 256)
DEFAULT_THRESHOLD = 0.5   # 🔒 FIXED — SAME ACCURACY
FILE_ID = "1-MkZMXNjh2kHSPgh7-FOZ505iSZbxVmo"
MODEL_PATH = "best_model.pth"

# ---------------- DOWNLOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    with st.spinner("⬇️ Loading trained U-Net model..."):
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- MODEL ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)

        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = TF.resize(x, skip.shape[2:])
            x = self.ups[i+1](torch.cat((skip, x), dim=1))

        return self.final(x)

# ---------------- TRANSFORM (SAME) ----------------
transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),
    ToTensorV2()
])

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = UNET().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ---------------- PREDICT ----------------
@torch.no_grad()
def predict(image, model):
    orig = np.array(image.convert("RGB"))
    h, w = orig.shape[:2]

    x = transform(image=orig)["image"].unsqueeze(0).to(DEVICE)
    prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
    prob = cv2.resize(prob, (w,h))

    mask = (prob > DEFAULT_THRESHOLD).astype(np.uint8)

    # Optional smoothing (visual only, accuracy same)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    overlay = orig.copy()
    overlay[mask==1] = [255,0,0]
    blended = cv2.addWeighted(orig, 0.65, overlay, 0.35, 0)

    oil_pct = (mask.sum()/mask.size)*100
    return blended, mask, oil_pct

# ---------------- UI ----------------
st.title("🌊 Oil Spill Detection System")
st.caption("AI-powered oil spill segmentation using U-Net")

model = load_model()

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("📤 Upload Satellite Image")
    uploaded = st.file_uploader("Supported formats: JPG, PNG", ["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Input Image", use_container_width=True)

        if st.button("🔍 Run Oil Spill Detection", use_container_width=True):
            blended, mask, oil_pct = predict(image, model)
            st.session_state.result = (blended, mask, oil_pct)

with col2:
    st.subheader("📊 Detection Results")

    if "result" in st.session_state:
        blended, mask, oil_pct = st.session_state.result

        st.image(blended, caption="Detected Oil Spill Overlay", use_container_width=True)

        if oil_pct > 1:
            st.error(f"🚨 OIL SPILL DETECTED ({oil_pct:.2f}% area)")
        else:
            st.success(f"✅ No Significant Spill ({oil_pct:.2f}% area)")

        st.markdown("### 📈 Summary")
        st.write(f"**Oil Coverage:** {oil_pct:.2f}%")
        st.write("**Model:** U-Net (PyTorch)")
        st.write("**Inference:** CPU")

    else:
        st.info("Upload an image and run detection")

st.markdown("---")
st.markdown("<center>Developed by Khushi • Streamlit + PyTorch</center>", unsafe_allow_html=True)
