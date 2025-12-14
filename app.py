import os
import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import gdown

# ---------------- ENV FIXES ----------------
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Oil Spill Segmentation (U-Net)",
    page_icon="🌊",
    layout="wide"
)

# ---------------- GOOGLE DRIVE MODEL ----------------
FILE_ID = "1-MkZMXNjh2kHSPgh7-FOZ505iSZbxVmo"
MODEL_PATH = "best_unet_oilspill.pth"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    with st.spinner("⬇️ Downloading trained model..."):
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- U-NET ARCHITECTURE (EXACT MATCH) ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------- PREDICTION ----------------
def predict_mask(pil_img, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)
        mask = (prob > 0.5).float().cpu().numpy()[0, 0]

    return mask

def overlay_image(img, mask, alpha=0.4):
    img = img.resize(mask.shape[::-1])
    img_np = np.array(img).astype(np.uint8)
    overlay = img_np.copy()
    overlay[mask == 1] = [255, 0, 0]
    return Image.fromarray(cv2.addWeighted(img_np, 1 - alpha, overlay, alpha, 0))

# ---------------- UI ----------------
st.title("🌊 Oil Spill Segmentation System")
st.caption("U-Net based satellite image analysis")

uploaded = st.file_uploader("Upload Satellite Image", ["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    mask = predict_mask(img)

    spill_percent = (mask.sum() / mask.size) * 100

    if spill_percent > 2:
        st.error(f"🚨 Oil Spill Detected ({spill_percent:.2f}% area)")
    else:
        st.success(f"✅ No Significant Spill ({spill_percent:.2f}% area)")

    overlay = overlay_image(img, mask)
    st.image(overlay, caption="Oil Spill Overlay", use_container_width=True)

    buf = BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")
    st.download_button("⬇️ Download Mask", buf.getvalue(), "mask.png", "image/png")

st.markdown("---")
st.markdown("<center>Developed by Khushi • Streamlit + PyTorch</center>", unsafe_allow_html=True)
