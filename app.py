import os
import cv2
import gdown
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🌊",
    layout="wide"
)

# ---------------- MODEL DOWNLOAD ----------------
FILE_ID = "1-MkZMXNjh2kHSPgh7-FOZ505iSZbxVmo"   # 👈 CHANGE THIS
MODEL_PATH = "oil_spill_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇️ Downloading trained model..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# ---------------- U-NET ARCHITECTURE ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        self.d3 = Down(256, 512)
        self.d4 = Down(512, 1024)
        self.u1 = Up(1024, 512)
        self.u2 = Up(512, 256)
        self.u3 = Up(256, 128)
        self.u4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

# ✅ handles both full-model and state_dict safely
if isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint, strict=False)
else:
    model = checkpoint.to(device)

model.eval()

# ---------------- PREPROCESS ----------------
def preprocess(img, size=256):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return tfm(img).unsqueeze(0)

# ---------------- PREDICTION ----------------
def predict(image):
    x = preprocess(image).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))
        mask = (prob > 0.5).float().cpu().numpy()[0, 0]
    return mask

def overlay(image, mask):
    img = np.array(image.resize(mask.shape[::-1]))
    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 0]
    return Image.fromarray(cv2.addWeighted(img, 0.6, overlay, 0.4, 0))

# ---------------- UI ----------------
st.title("🌊 Oil Spill Detection System")
st.caption("U-Net based oil spill segmentation")

uploaded = st.file_uploader("Upload satellite image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    with st.spinner("🔍 Running inference..."):
        mask = predict(img)

    spill_pct = (mask.sum() / mask.size) * 100

    if spill_pct > 2:
        st.error(f"🚨 Oil Spill Detected ({spill_pct:.2f}% area)")
    else:
        st.success(f"✅ No Significant Spill ({spill_pct:.2f}% area)")

    st.image(overlay(img, mask), caption="Oil Spill Overlay", use_container_width=True)

    buf = BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")

    st.download_button(
        "⬇️ Download Mask",
        data=buf.getvalue(),
        file_name="oil_spill_mask.png",
        mime="image/png"
    )

st.markdown("---")
st.markdown(
    "<center><b>Developed by Khushi</b> • Streamlit + PyTorch</center>",
    unsafe_allow_html=True
)
