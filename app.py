import os
import io
import time
import glob
import json
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="NeuroSeg AI — Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CUSTOM CSS — medical dark theme
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    /* ---- Global ---- */
    .stApp {
        background: linear-gradient(160deg, #0a0a1a 0%, #0d1b2a 35%, #1b2838 70%, #0a0a1a 100%);
        color: #e0e6ed;
        font-family: 'Inter', sans-serif;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0a0a1a 100%);
        border-right: 1px solid rgba(0, 200, 180, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00c8b4;
    }

    /* ---- Hero ---- */
    .hero-container { text-align: center; padding: 1.2rem 1rem 0.3rem 1rem; }
    .hero-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #00c8b4, #00b4d8, #7b2ff7, #ff006e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem; letter-spacing: -0.5px;
    }
    .hero-sub { font-size: 1rem; color: #8899aa; margin-bottom: 0.3rem; }
    .hero-badge {
        display: inline-block; margin-top: 0.4rem;
        background: rgba(0,200,180,0.08); border: 1px solid rgba(0,200,180,0.25);
        border-radius: 20px; padding: 0.25rem 1rem; font-size: 0.75rem;
        color: #00c8b4; letter-spacing: 1px; text-transform: uppercase;
    }

    /* ---- Glass cards ---- */
    .glass-card {
        background: rgba(13,27,42,0.7);
        border: 1px solid rgba(0,200,180,0.1);
        border-radius: 16px; padding: 1rem;
        backdrop-filter: blur(16px);
        transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,200,180,0.12);
        border-color: rgba(0,200,180,0.3);
    }
    .card-label {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.5px; color: #00c8b4; margin-bottom: 0.4rem;
    }

    /* ---- Stat pills ---- */
    .stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.5rem 0; }
    .stat-pill {
        background: rgba(0,200,180,0.08); border: 1px solid rgba(0,200,180,0.2);
        border-radius: 24px; padding: 0.4rem 0.9rem; font-size: 0.8rem;
        color: #c0cdd8; display: inline-flex; align-items: center; gap: 0.35rem;
    }
    .stat-pill .val { font-weight: 700; color: #00c8b4; font-family: 'JetBrains Mono', monospace; }

    /* ---- Metric cards ---- */
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.6rem; margin: 0.8rem 0; }
    .metric-card {
        background: rgba(0,200,180,0.06); border: 1px solid rgba(0,200,180,0.15);
        border-radius: 14px; padding: 0.8rem 1rem; text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: rgba(0,200,180,0.4); }
    .metric-card .mc-label {
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.2px; color: #6b8299; margin-bottom: 0.25rem;
    }
    .metric-card .mc-value {
        font-size: 1.4rem; font-weight: 800; color: #00c8b4;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .mc-unit { font-size: 0.7rem; color: #556677; margin-top: 0.1rem; }

    /* ---- Accuracy bar ---- */
    .acc-bar-wrap { margin: 0.3rem 0; }
    .acc-bar-label { font-size: 0.75rem; color: #8899aa; margin-bottom: 0.2rem; display: flex; justify-content: space-between; }
    .acc-bar-track {
        height: 8px; border-radius: 4px;
        background: rgba(255,255,255,0.06); overflow: hidden;
    }
    .acc-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
    .acc-bar-fill.teal { background: linear-gradient(90deg, #00c8b4, #00b4d8); }
    .acc-bar-fill.purple { background: linear-gradient(90deg, #7b2ff7, #a855f7); }
    .acc-bar-fill.pink { background: linear-gradient(90deg, #ff006e, #ff4d8d); }

    /* ---- Chips ---- */
    .chip-row { display: flex; gap: 0.5rem; margin: 0.4rem 0 0.7rem 0; }
    .chip {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px; padding: 0.28rem 0.85rem; font-size: 0.75rem; color: #667788;
    }
    .chip.active { background: rgba(0,200,180,0.15); border-color: #00c8b4; color: #fff; }

    /* ---- Column headers ---- */
    .col-header {
        font-size: 0.95rem; font-weight: 700; color: #fff; text-align: center;
        padding: 0.45rem 0 0.25rem 0; border-bottom: 2px solid; margin-bottom: 0.5rem;
    }
    .col-header.mri  { border-color: #00c8b4; }
    .col-header.pred { border-color: #ff006e; }
    .col-header.xai  { border-color: #7b2ff7; }

    /* ---- Section divider ---- */
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #fff;
        padding: 0.6rem 0 0.3rem 0; border-left: 3px solid #00c8b4;
        padding-left: 0.8rem; margin: 1.2rem 0 0.6rem 0;
    }

    /* ---- Tab styling ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(0,200,180,0.06); border-radius: 10px 10px 0 0;
        border: 1px solid rgba(0,200,180,0.1); color: #8899aa; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,200,180,0.15) !important;
        border-color: #00c8b4 !important; color: #fff !important;
    }

    /* ---- Download button ---- */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00c8b4, #00b4d8) !important;
        color: #0a0a1a !important; font-weight: 700 !important;
        border: none !important; border-radius: 10px !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #00b4d8, #7b2ff7) !important;
        color: #fff !important;
    }

    /* ---- Upload area ---- */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(0,200,180,0.25) !important;
        border-radius: 14px !important; padding: 0.5rem !important;
    }

    /* ---- Footer ---- */
    .footer {
        text-align: center; color: #334455; font-size: 0.72rem;
        padding: 1.5rem 0 1rem 0; border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 1.5rem;
    }

    /* ---- Hide defaults ---- */
    header[data-testid="stHeader"] { background: transparent; }
    .stDeployButton { display: none; }
    [data-testid="stMetricValue"] { color: #00c8b4; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #7b2ff7; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.8px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURATION
# ============================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(APP_DIR, "samples")
MODEL_PATH = os.path.join(APP_DIR, "brain_tumor_model.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(APP_DIR), "brain_tumor_model.pth")

MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
CLASS_NAMES = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
CLASS_COLORS = ["teal", "purple", "pink"]


# ============================================
# GENERATE 5 DEMO SAMPLES FROM DATASET
# ============================================
def generate_demo_samples(data_dir, num_samples=5):
    """Extract real patient .npy samples from BraTS NIfTI dataset on first run."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    existing = sorted([f for f in os.listdir(SAMPLE_DIR) if f.endswith("_image.npy")])
    if len(existing) >= num_samples:
        return

    try:
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
            Spacingd, NormalizeIntensityd, MapTransform,
        )
        from monai.data import Dataset as MonaiDataset

        class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
            def __call__(self, data):
                d = dict(data)
                for key in self.keys:
                    img = d[key]
                    if img.ndim == 4 and img.shape[0] == 1:
                        img = img.squeeze(0)
                    result = [
                        np.logical_or(img == 1, np.logical_or(img == 2, img == 3)),
                        np.logical_or(img == 1, img == 3),
                        img == 3,
                    ]
                    d[key] = np.stack(result, axis=0).astype(np.float32)
                return d

        for search_path in [data_dir, os.path.join(APP_DIR, data_dir),
                            os.path.join(os.path.dirname(APP_DIR), data_dir)]:
            images_dir = os.path.join(search_path, "imagesTr")
            labels_dir = os.path.join(search_path, "labelsTr")
            if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
                break
        else:
            return

        all_images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
        all_labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
        if len(all_images) < num_samples:
            return

        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])

        indices = np.linspace(0, len(all_images) - 1, num_samples, dtype=int)
        data_dicts = [{"image": all_images[i], "label": all_labels[i]} for i in indices]
        ds = MonaiDataset(data=data_dicts, transform=val_transforms)

        for idx in range(len(ds)):
            img_name = f"patient_{idx + 1}_image.npy"
            lbl_name = f"patient_{idx + 1}_label.npy"
            if os.path.exists(os.path.join(SAMPLE_DIR, img_name)):
                continue
            data = ds[idx]
            np.save(os.path.join(SAMPLE_DIR, img_name), data["image"].numpy())
            np.save(os.path.join(SAMPLE_DIR, lbl_name), data["label"].numpy())
    except Exception:
        pass


generate_demo_samples("Task01_BrainTumour")


# ============================================
# DISCOVER SAMPLE FILES
# ============================================
def get_sample_patients():
    if not os.path.isdir(SAMPLE_DIR):
        return {}
    patients = {}
    for f in sorted(os.listdir(SAMPLE_DIR)):
        if f.endswith("_image.npy"):
            name = f.replace("_image.npy", "")
            label_path = os.path.join(SAMPLE_DIR, f"{name}_label.npy")
            patients[name] = {
                "image": os.path.join(SAMPLE_DIR, f),
                "label": label_path if os.path.exists(label_path) else None,
            }
    return patients


# ============================================
# MODEL
# ============================================
class SegmentationModel:
    def __init__(self, model_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        ).to(self.device)
        if os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        else:
            st.error(f"Model not found at `{model_weights_path}`")

    def predict(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(0)
            output = sliding_window_inference(
                inputs=image_tensor, roi_size=(128, 128, 32),
                sw_batch_size=1, predictor=self.model, overlap=0.5,
            )
            output = torch.sigmoid(output)
            return (output > 0.5).float().squeeze(0)


class XAIModel:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def generate_saliency_map(self, image_tensor, target_channel=0):
        torch.cuda.empty_cache()
        image_tensor = image_tensor.to(self.device)
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(0)
        original_shape = image_tensor.shape[2:]
        small_input = F.interpolate(image_tensor, scale_factor=0.5, mode="trilinear", align_corners=False)
        padded_input = self._pad_to_divisible(small_input)
        padded_input.requires_grad_(True)
        output = self.model(padded_input)
        target = output[:, target_channel].mean()
        self.model.zero_grad()
        target.backward()
        grad = padded_input.grad
        if grad is None:
            return np.zeros(original_shape)
        saliency = torch.abs(grad).sum(dim=1, keepdim=True)
        d, h, w = small_input.shape[2:]
        saliency = saliency[:, :, :d, :h, :w]
        saliency_full = F.interpolate(saliency, size=original_shape, mode="trilinear", align_corners=False)
        saliency_np = saliency_full.squeeze().detach().cpu().numpy()
        saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
        return saliency_np

    def _pad_to_divisible(self, tensor, divisor=16):
        spatial_dims = tensor.shape[2:]
        paddings = []
        for dim in reversed(spatial_dims):
            rem = dim % divisor
            paddings.extend([0, 0] if rem == 0 else [0, divisor - rem])
        return F.pad(tensor, paddings)


@st.cache_resource
def load_pipeline():
    seg_model = SegmentationModel(MODEL_PATH)
    xai_model = XAIModel(seg_model.model)
    return seg_model, xai_model


# ============================================
# METRIC HELPERS
# ============================================
def compute_dice(pred_ch, label_ch):
    pred_flat = pred_ch.flatten()
    label_flat = label_ch.flatten()
    intersection = (pred_flat * label_flat).sum()
    return float((2.0 * intersection) / (pred_flat.sum() + label_flat.sum() + 1e-8))


def compute_metrics(prediction, label_tensor=None):
    pred_np = prediction.cpu().numpy()
    metrics = {}
    for ch, name in enumerate(CLASS_NAMES):
        ch_data = pred_np[ch]
        metrics[name] = {
            "volume_voxels": int(ch_data.sum()),
            "volume_cm3": round(float(ch_data.sum()) * 0.001, 2),
            "coverage_pct": round(float(ch_data.sum()) / max(ch_data.size, 1) * 100, 4),
        }
        if label_tensor is not None:
            lbl_np = label_tensor.numpy() if hasattr(label_tensor, 'numpy') else label_tensor
            metrics[name]["dice_score"] = round(compute_dice(ch_data, lbl_np[ch]), 4)
    metrics["total_tumor_voxels"] = int(pred_np.sum())
    metrics["total_tumor_cm3"] = round(float(pred_np.sum()) * 0.001, 2)
    if label_tensor is not None:
        dice_scores = [metrics[n]["dice_score"] for n in CLASS_NAMES]
        metrics["mean_dice"] = round(float(np.mean(dice_scores)), 4)
    return metrics


def generate_report_text(patient_id, metrics, img_shape, proc_time, device_str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    has_dice = "mean_dice" in metrics

    lines = [
        "=" * 65,
        "         NEUROSEG AI — CLINICAL SEGMENTATION REPORT",
        "=" * 65,
        "",
        f"  Report Generated : {now}",
        f"  Patient ID       : {patient_id}",
        f"  Model            : SegResNet (MONAI)",
        f"  Device           : {device_str}",
        f"  Inference Time   : {proc_time:.2f} seconds",
        "",
        "-" * 65,
        "  SCAN INFORMATION",
        "-" * 65,
        f"  Volume Shape     : {img_shape[1]} x {img_shape[2]} x {img_shape[3]}",
        f"  MRI Modalities   : T1, T1ce, T2, FLAIR (4 channels)",
        f"  Voxel Spacing    : 1.0 mm isotropic",
        "",
        "-" * 65,
        "  SEGMENTATION RESULTS",
        "-" * 65,
    ]

    for name in CLASS_NAMES:
        m = metrics[name]
        lines.append(f"  {name}:")
        lines.append(f"    Volume         : {m['volume_voxels']:,} voxels ({m['volume_cm3']:.2f} cm³)")
        lines.append(f"    Coverage       : {m['coverage_pct']:.4f}%")
        if "dice_score" in m:
            lines.append(f"    Dice Score     : {m['dice_score']:.4f} ({m['dice_score']*100:.2f}%)")
        lines.append("")

    lines.append(f"  Total Tumor Volume : {metrics['total_tumor_voxels']:,} voxels ({metrics['total_tumor_cm3']:.2f} cm³)")

    if has_dice:
        lines.extend([
            "",
            "-" * 65,
            "  ACCURACY METRICS",
            "-" * 65,
            f"  Mean Dice Score  : {metrics['mean_dice']:.4f} ({metrics['mean_dice']*100:.2f}%)",
        ])
        for name in CLASS_NAMES:
            d = metrics[name]["dice_score"]
            bar_len = int(d * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"  {name:26s}: [{bar}] {d:.4f}")

    lines.extend([
        "",
        "-" * 65,
        "  CLINICAL NOTES",
        "-" * 65,
        "  - XAI saliency maps highlight regions of highest model attention.",
        "  - Tumour sub-regions follow BraTS convention:",
        "      WT = Labels 1+2+3 | TC = Labels 1+3 | ET = Label 3",
        "  - This is an AI-assisted analysis. Final diagnosis should be",
        "    confirmed by a qualified radiologist.",
        "",
        "=" * 65,
        "  NeuroSeg AI — Powered by SegResNet, MONAI & BraTS 2021",
        "=" * 65,
    ])
    return "\n".join(lines)


# ============================================
# PLOTTING
# ============================================
BG_COLOR = "#0a0a1a"

def make_fig(img_slice, overlay=None, cmap_overlay="spring", alpha=0.6, cmap_base="gray"):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.imshow(img_slice, cmap=cmap_base)
    if overlay is not None:
        ax.imshow(overlay, cmap=cmap_overlay, alpha=alpha)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def make_multiclass_fig(img_slice, prediction, slice_idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG_COLOR)
    colors = ["Greens", "Purples", "Reds"]
    for i, (ax, name, cmap) in enumerate(zip(axes, CLASS_NAMES, colors)):
        ax.set_facecolor(BG_COLOR)
        ax.imshow(img_slice, cmap="gray", alpha=0.5)
        mask = prediction[i, :, :, slice_idx].cpu().numpy()
        overlay = np.ma.masked_where(mask == 0, mask)
        ax.imshow(overlay, cmap=cmap, alpha=0.7)
        ax.set_title(name, color="#e0e6ed", fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.05)
    return fig


# ============================================
# UI
# ============================================

# --- Hero ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">NeuroSeg AI</div>
    <div class="hero-sub">
        3D Brain Tumor Segmentation with Explainable AI
    </div>
    <span class="hero-badge">SegResNet &bull; MONAI &bull; BraTS 2021</span>
</div>
""", unsafe_allow_html=True)

# --- Load model ---
seg_model, xai_model = load_pipeline()
patients = get_sample_patients()

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 0.8rem 0;">
        <span style="font-size:2rem;">🧠</span><br>
        <span style="font-size:0.7rem;color:#00c8b4;letter-spacing:2px;text-transform:uppercase;font-weight:700;">Diagnostic Console</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    input_mode = st.radio("Input Source", ["Demo Patients", "Upload MRI"], horizontal=True)

# ============================================
# INPUT HANDLING
# ============================================
img_tensor = None
label_tensor = None
patient_id = "Unknown"

if input_mode == "Demo Patients":
    if not patients:
        st.warning("No demo samples found. Place `patient_X_image.npy` files in `samples/` folder.")
        st.stop()
    with st.sidebar:
        patient_name = st.selectbox(
            "Select Patient",
            list(patients.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        patient_id = patient_name.replace("_", " ").title()

    img_np = np.load(patients[patient_name]["image"])
    img_tensor = torch.from_numpy(img_np).float()
    if patients[patient_name]["label"] is not None:
        label_tensor = torch.from_numpy(np.load(patients[patient_name]["label"])).float()

else:
    with st.sidebar:
        st.markdown("#### Upload Patient MRI")
        uploaded_img = st.file_uploader("MRI Image (.npy)", type=["npy"], key="img_upload")
        uploaded_lbl = st.file_uploader("Label — optional (.npy)", type=["npy"], key="lbl_upload")

    if uploaded_img is not None:
        img_np = np.load(io.BytesIO(uploaded_img.read()))
        img_tensor = torch.from_numpy(img_np).float()
        patient_id = uploaded_img.name.replace("_image.npy", "").replace("_", " ").title()
        if uploaded_lbl is not None:
            label_tensor = torch.from_numpy(np.load(io.BytesIO(uploaded_lbl.read()))).float()
    else:
        st.info("Upload a `.npy` MRI file (shape: 4, H, W, D) to begin analysis.")
        st.stop()

# Validate shape
if img_tensor.ndim != 4 or img_tensor.shape[0] != 4:
    st.error(f"Invalid MRI shape: {img_tensor.shape}. Expected (4, H, W, D) with 4 modalities.")
    st.stop()

# --- Sidebar: modality + slice ---
with st.sidebar:
    st.markdown("---")
    modality_idx = st.selectbox("MRI Modality", list(range(4)), format_func=lambda i: MODALITY_NAMES[i], index=1)


# ============================================
# INFERENCE
# ============================================
@st.cache_data(show_spinner=False)
def run_inference(_img_tensor_np):
    img_t = torch.from_numpy(_img_tensor_np).float()
    t0 = time.time()
    prediction = seg_model.predict(img_t)
    saliency = xai_model.generate_saliency_map(img_t)
    proc_time = time.time() - t0
    return prediction, saliency, proc_time


with st.spinner("Running 3D sliding-window inference & computing XAI saliency..."):
    prediction, saliency_map, proc_time = run_inference(img_tensor.numpy())

max_slices = img_tensor.shape[-1]
metrics = compute_metrics(prediction, label_tensor)

with st.sidebar:
    slice_idx = st.slider("Slice Depth (Z-axis)", 0, max_slices - 1, max_slices // 2)

# ============================================
# SIDEBAR STATS
# ============================================
with st.sidebar:
    st.markdown("---")
    h, w, d = img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3]

    st.markdown("### Scan Info")
    st.markdown(f"""
    <div class="stat-row">
        <span class="stat-pill">Volume <span class="val">{h}&times;{w}&times;{d}</span></span>
        <span class="stat-pill">Channels <span class="val">4</span></span>
        <span class="stat-pill">Slices <span class="val">{d}</span></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Tumor Volume")
    st.markdown(f"""
    <div class="stat-row">
        <span class="stat-pill">Total <span class="val">{metrics['total_tumor_voxels']:,} vx</span></span>
        <span class="stat-pill">~<span class="val">{metrics['total_tumor_cm3']:.1f} cm&sup3;</span></span>
    </div>
    """, unsafe_allow_html=True)

    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        m = metrics[name]
        st.markdown(f"""
        <div class="stat-row">
            <span class="stat-pill">{name.split('(')[1].replace(')','')} <span class="val">{m['volume_voxels']:,}</span></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Performance")
    st.markdown(f"""
    <div class="stat-row">
        <span class="stat-pill">Time <span class="val">{proc_time:.1f}s</span></span>
        <span class="stat-pill">Device <span class="val">{str(seg_model.device).upper()}</span></span>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# MAIN TABS
# ============================================
tab_viz, tab_metrics, tab_report = st.tabs(["Visualization", "Accuracy & Metrics", "Download Report"])

# ---- TAB 1: Visualization ----
with tab_viz:
    # Modality chips
    chips_html = '<div class="chip-row">'
    for i, name in enumerate(MODALITY_NAMES):
        cls = "chip active" if i == modality_idx else "chip"
        chips_html += f'<span class="{cls}">{name}</span>'
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    # 3-column main view
    col1, col2, col3 = st.columns(3, gap="medium")
    base_slice = img_tensor[modality_idx, :, :, slice_idx].cpu().numpy()

    with col1:
        st.markdown('<div class="col-header mri">Input MRI</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = make_fig(base_slice)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="col-header pred">Tumor Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        mask = prediction[0, :, :, slice_idx].cpu().numpy()
        overlay = np.ma.masked_where(mask == 0, mask)
        fig = make_fig(base_slice, overlay=overlay, cmap_overlay="hot", alpha=0.65)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="col-header xai">XAI Saliency</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = make_fig(base_slice, overlay=saliency_map[:, :, slice_idx], cmap_overlay="inferno", alpha=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Per-class segmentation row
    st.markdown('<div class="section-title">Per-Class Segmentation Overlay</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig = make_multiclass_fig(base_slice, prediction, slice_idx)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # All modalities row
    st.markdown(f'<div class="section-title">All MRI Modalities — Slice {slice_idx}</div>', unsafe_allow_html=True)
    mcols = st.columns(4, gap="small")
    for i, mcol in enumerate(mcols):
        with mcol:
            st.markdown(f'<div class="glass-card"><div class="card-label">{MODALITY_NAMES[i]}</div>', unsafe_allow_html=True)
            fig = make_fig(img_tensor[i, :, :, slice_idx].cpu().numpy())
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)


# ---- TAB 2: Accuracy & Metrics ----
with tab_metrics:
    st.markdown('<div class="section-title">Segmentation Metrics</div>', unsafe_allow_html=True)

    has_dice = "mean_dice" in metrics

    # Metric cards row
    metric_cards_html = '<div class="metric-grid">'
    metric_cards_html += f'''
        <div class="metric-card">
            <div class="mc-label">Total Volume</div>
            <div class="mc-value">{metrics["total_tumor_voxels"]:,}</div>
            <div class="mc-unit">voxels</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Volume (cm³)</div>
            <div class="mc-value">{metrics["total_tumor_cm3"]:.1f}</div>
            <div class="mc-unit">cubic cm</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Inference</div>
            <div class="mc-value">{proc_time:.1f}s</div>
            <div class="mc-unit">{str(seg_model.device).upper()}</div>
        </div>
    '''
    if has_dice:
        metric_cards_html += f'''
        <div class="metric-card">
            <div class="mc-label">Mean Dice</div>
            <div class="mc-value">{metrics["mean_dice"]:.4f}</div>
            <div class="mc-unit">{metrics["mean_dice"]*100:.2f}%</div>
        </div>
        '''
    metric_cards_html += '</div>'
    st.markdown(metric_cards_html, unsafe_allow_html=True)

    # Per-class breakdown
    st.markdown('<div class="section-title">Per-Class Breakdown</div>', unsafe_allow_html=True)

    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        m = metrics[name]
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:0.8rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                <span style="font-weight:700;color:#fff;font-size:0.95rem;">{name}</span>
                <span class="stat-pill">{m['volume_voxels']:,} voxels &bull; {m['volume_cm3']:.2f} cm&sup3;</span>
            </div>
        """, unsafe_allow_html=True)

        if "dice_score" in m:
            pct = m["dice_score"] * 100
            st.markdown(f"""
            <div class="acc-bar-wrap">
                <div class="acc-bar-label">
                    <span>Dice Similarity Coefficient</span>
                    <span style="font-weight:700;color:#00c8b4;">{m['dice_score']:.4f} ({pct:.2f}%)</span>
                </div>
                <div class="acc-bar-track">
                    <div class="acc-bar-fill {color}" style="width:{pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="acc-bar-wrap">
                <div class="acc-bar-label">
                    <span>Volume Coverage</span>
                    <span style="font-weight:700;color:#7b2ff7;">{m['coverage_pct']:.4f}%</span>
                </div>
                <div class="acc-bar-track">
                    <div class="acc-bar-fill {color}" style="width:{min(m['coverage_pct'] * 20, 100)}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not has_dice:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:1.5rem;margin-top:1rem;">
            <span style="font-size:1.5rem;">📋</span><br>
            <span style="color:#8899aa;font-size:0.9rem;">
                Upload a ground-truth label file to compute Dice accuracy scores.
            </span>
        </div>
        """, unsafe_allow_html=True)


# ---- TAB 3: Download Report ----
with tab_report:
    st.markdown('<div class="section-title">Clinical Segmentation Report</div>', unsafe_allow_html=True)

    report_text = generate_report_text(
        patient_id, metrics, img_tensor.shape, proc_time, str(seg_model.device).upper()
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.code(report_text, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        st.download_button(
            label="Download Report (.txt)",
            data=report_text,
            file_name=f"NeuroSeg_Report_{patient_id.replace(' ', '_')}.txt",
            mime="text/plain",
        )

    with col_dl2:
        metrics_json = json.dumps(metrics, indent=2)
        st.download_button(
            label="Download Metrics (.json)",
            data=metrics_json,
            file_name=f"NeuroSeg_Metrics_{patient_id.replace(' ', '_')}.json",
            mime="application/json",
        )

    with col_dl3:
        # Generate a composite visualization image for download
        fig_dl, axes_dl = plt.subplots(1, 3, figsize=(18, 6))
        fig_dl.patch.set_facecolor(BG_COLOR)
        mid_slice = max_slices // 2
        base_dl = img_tensor[1, :, :, mid_slice].cpu().numpy()

        axes_dl[0].set_facecolor(BG_COLOR)
        axes_dl[0].imshow(base_dl, cmap="gray")
        axes_dl[0].set_title("Input MRI (T1ce)", color="#e0e6ed", fontsize=12, fontweight="bold")
        axes_dl[0].axis("off")

        axes_dl[1].set_facecolor(BG_COLOR)
        axes_dl[1].imshow(base_dl, cmap="gray", alpha=0.5)
        m_dl = prediction[0, :, :, mid_slice].cpu().numpy()
        axes_dl[1].imshow(np.ma.masked_where(m_dl == 0, m_dl), cmap="hot", alpha=0.65)
        axes_dl[1].set_title("Tumor Prediction", color="#e0e6ed", fontsize=12, fontweight="bold")
        axes_dl[1].axis("off")

        axes_dl[2].set_facecolor(BG_COLOR)
        axes_dl[2].imshow(base_dl, cmap="gray", alpha=0.5)
        axes_dl[2].imshow(saliency_map[:, :, mid_slice], cmap="inferno", alpha=0.6)
        axes_dl[2].set_title("XAI Saliency", color="#e0e6ed", fontsize=12, fontweight="bold")
        axes_dl[2].axis("off")

        fig_dl.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05)
        buf = io.BytesIO()
        fig_dl.savefig(buf, format="png", dpi=150, facecolor=BG_COLOR)
        plt.close(fig_dl)
        buf.seek(0)

        st.download_button(
            label="Download Visualization (.png)",
            data=buf,
            file_name=f"NeuroSeg_Viz_{patient_id.replace(' ', '_')}.png",
            mime="image/png",
        )


# ---- Footer ----
st.markdown("""
<div class="footer">
    NeuroSeg AI &mdash; 3D Brain Tumor Segmentation with Explainable AI<br>
    Built with Streamlit &bull; MONAI &bull; PyTorch &bull; BraTS 2021<br>
    &copy; 2026 &mdash; For research and educational purposes only
</div>
""", unsafe_allow_html=True)
