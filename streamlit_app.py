import re
import time
from pathlib import Path
from io import BytesIO

import requests
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.models import (
    convnext_small, ConvNeXt_Small_Weights,
    resnet101, ResNet101_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
)

from data_utils import load_car_annotations, get_mean_std_from_weights


MODEL_REGISTRY = {
    "convnext_small": {
        "builder": convnext_small,
        "weights": ConvNeXt_Small_Weights.IMAGENET1K_V1,
    },
    "resnet101": {
        "builder": resnet101,
        "weights": ResNet101_Weights.IMAGENET1K_V1,
    },
    "efficientnet_v2_m": {
        "builder": efficientnet_v2_m,
        "weights": EfficientNet_V2_M_Weights.IMAGENET1K_V1,
    },
}


def infer_model_key(filename: str) -> str | None:
    for key in MODEL_REGISTRY:
        if filename.startswith(key):
            return key
    return None


def infer_img_size(filename: str) -> int | None:
    m = re.search(r"_(\d+)_best\\.pth$", filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def build_model(model_key: str, num_classes: int, weights):
    model = MODEL_REGISTRY[model_key]["builder"](weights=weights)
    if model_key == "convnext_small":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif model_key == "resnet101":
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_key == "efficientnet_v2_m":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")
    return model


def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def square_crop(img: Image.Image, x: int, y: int, size: int) -> Image.Image:
    w, h = img.size
    size = max(1, min(size, w, h))
    x = max(0, min(x, w - size))
    y = max(0, min(y, h - size))
    return img.crop((x, y, x + size, y + size))


def pad_to_square(img: Image.Image, fill_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    if w == h:
        return img.copy()
    canvas = Image.new("RGB", (side, side), color=fill_color)
    left = (side - w) // 2
    top = (side - h) // 2
    canvas.paste(img, (left, top))
    return canvas


def build_transform(img_size: int, weights):
    mean, std = get_mean_std_from_weights(weights)
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ])


@st.cache_resource
def load_checkpoint_model(ckpt_path: str, model_key: str, num_classes: int, device: str):
    weights = MODEL_REGISTRY[model_key]["weights"]
    model = build_model(model_key, num_classes, weights)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, weights


def maybe_warmup_model(model, x: torch.Tensor, device: str, warmup_key: str, warmup_iters: int = 3) -> None:
    warmed_keys = st.session_state.setdefault("_warmed_model_keys", set())
    if warmup_key in warmed_keys:
        return

    with torch.no_grad():
        for _ in range(max(1, warmup_iters)):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

    warmed_keys.add(warmup_key)


def timed_forward(model, x: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return logits, dt


st.set_page_config(page_title="Car Model Inference", layout="wide")
st.title("Car Model Recognition - Multi-Model Inference")

with st.sidebar:
    st.header("Settings")
    data_root = st.text_input("Data root", value="./stanford_cars")
    ckpt_dir = st.text_input("Checkpoints dir", value="./checkpoints")
    default_img_size = st.selectbox("Default image size", [224, 320], index=1)
    use_gpu = st.checkbox("Use GPU if available", value=True)

device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

try:
    _, _, id_to_name = load_car_annotations(data_root, verbose=False)
except Exception as e:
    st.error(f"Failed to load class names from {data_root}: {e}")
    st.stop()

st.subheader("Image Source")
col_a, col_b = st.columns(2)
with col_a:
    url = st.text_input(
        "Image URL",
        value="https://images.hgmsites.net/lrg/2012-bmw-x6-awd-4-door-35i-angular-front-exterior-view_100355322_l.jpg",
    )
with col_b:
    upload = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

img = None
if upload is not None:
    img = Image.open(upload).convert("RGB")
elif url.strip():
    try:
        img = load_image_from_url(url)
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")

if img is None:
    st.stop()

w, h = img.size
st.subheader("Crop (Square)")
use_black_padding = st.checkbox("Use full image with black padding (skip crop)", value=False)
use_cropper = st.checkbox("Use interactive cropper (requires streamlit-cropper)", value=True)

crop_x: int = 0
crop_y: int = 0
crop_size: int = min(200, w, h)
min_crop_size: int = 1 if min(w, h) < 32 else 32
crop_img: Image.Image = pad_to_square(img) if use_black_padding else square_crop(img, 0, 0, min(200, w, h))

def clamp_box(x: int, y: int, size: int):
    size = max(1, min(size, w, h))
    x = max(0, min(x, w - size))
    y = max(0, min(y, h - size))
    return x, y, size

if use_black_padding:
    crop_img = pad_to_square(img)
    st.image(crop_img, caption="Padded to Square (Black Borders)", width="stretch")
elif use_cropper:
    try:
        from streamlit_cropper import st_cropper

        st.caption("Drag and resize the crop area. Aspect ratio is locked to a square.")
        realtime_crop = st.checkbox("Realtime crop updates", value=False, key="realtime_crop")
        crop_col, preview_col = st.columns(2)

        with crop_col:
            crop_box = st_cropper(
                img,
                realtime_update=realtime_crop,
                box_color="#FFEB3B",
                aspect_ratio=(1, 1),
                return_type="box",
            )

        if crop_box:
            rx = int(round(crop_box.get("left", crop_box.get("x", 0))))
            ry = int(round(crop_box.get("top", crop_box.get("y", 0))))
            rw = int(round(crop_box.get("width", crop_size)))
            rh = int(round(crop_box.get("height", crop_size)))
            crop_size = max(1, min(rw, rh))
            crop_x, crop_y, crop_size = clamp_box(rx, ry, crop_size)
        else:
            crop_x, crop_y, crop_size = clamp_box(0, 0, crop_size)

        crop_img = square_crop(img, crop_x, crop_y, crop_size)
        with preview_col:
            st.image(crop_img, caption="Cropped", width="stretch")
    except Exception as e:
        st.warning(
            f"streamlit-cropper not available ({e}). Install with `pip install streamlit-cropper`. Falling back to sliders."
        )
        use_cropper = False

if not use_cropper:
    crop_mode = st.radio("Crop mode", ["Center", "Top-left"], horizontal=True)
    crop_size = st.slider("Crop size", min_value=min_crop_size, max_value=min(w, h), value=min(200, w, h), step=1)

    if crop_mode == "Center":
        cx = st.slider("Center X", min_value=0, max_value=w, value=w // 2, step=1)
        cy = st.slider("Center Y", min_value=0, max_value=h, value=h // 2, step=1)
        crop_x = int(cx - crop_size // 2)
        crop_y = int(cy - crop_size // 2)
    else:
        crop_x = st.slider("Crop X", min_value=0, max_value=max(0, w - crop_size), value=0, step=1)
        crop_y = st.slider("Crop Y", min_value=0, max_value=max(0, h - crop_size), value=0, step=1)

    crop_x, crop_y, crop_size = clamp_box(crop_x, crop_y, crop_size)
    crop_img = square_crop(img, crop_x, crop_y, crop_size)
    st.image(crop_img, caption="Cropped", width="stretch")

st.subheader("Model Results")
ckpt_path = Path(ckpt_dir)
ckpts = sorted([p for p in ckpt_path.glob("*_best.pth")])

if not ckpts:
    st.warning(f"No '*_best.pth' checkpoints found in {ckpt_dir}")
    st.stop()

grouped_ckpts: dict[str, list[Path]] = {k: [] for k in MODEL_REGISTRY}
for p in ckpts:
    model_key = infer_model_key(p.name)
    if model_key is None:
        st.info(f"Skipping {p.name} (unknown model type)")
        continue
    grouped_ckpts[model_key].append(p)

for model_key, group in grouped_ckpts.items():
    if not group:
        continue

    # st.markdown(f"### {model_key}")
    cols = st.columns(len(group))

    for col, p in zip(cols, group):
        with col:
            img_size = infer_img_size(p.name) or default_img_size
            model, weights = load_checkpoint_model(str(p), model_key, len(id_to_name), device)
            tfms = build_transform(img_size, weights)

            x = tfms(crop_img)
            if hasattr(x, "as_subclass"):
                x = torch.as_tensor(x)
            x = x.unsqueeze(0).to(device)

            warmup_key = f"{p.resolve()}::{img_size}::{device}"
            maybe_warmup_model(model, x, device, warmup_key=warmup_key, warmup_iters=3)
            logits, dt = timed_forward(model, x, device)
            probs = torch.softmax(logits, dim=1)[0]
            topk = torch.topk(probs, k=min(5, probs.numel()))

            st.markdown(f"**{p.name} ({dt:.4f}s)**")

            results = []
            for score, idx in zip(topk.values, topk.indices):
                name = id_to_name.get(int(idx) + 1, str(int(idx)))
                results.append({"class": name, "prob": float(score)})
            st.table(results)
