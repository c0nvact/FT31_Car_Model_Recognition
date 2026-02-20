from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.io import ImageReadMode, read_image

from cars_dataset import clamp_bbox


def build_file_map(root: Path):
    return {p.name: p for p in root.rglob("*.jpg")}


def load_car_annotations(data_root, verbose: bool = True):
    data_root = Path(data_root)
    csv_names = data_root / "names.csv"
    csv_train = data_root / "anno_train.csv"
    csv_test = data_root / "anno_test.csv"

    img_train_root = data_root / "car_data" / "car_data" / "train"
    img_test_root = data_root / "car_data" / "car_data" / "test"

    names = pd.read_csv(csv_names, header=None, names=["class_name"])
    id_to_name = {i + 1: n for i, n in enumerate(names["class_name"].tolist())}

    cols = ["file_name", "x1", "y1", "x2", "y2", "class_id"]
    train_df = pd.read_csv(csv_train, header=None, names=cols)
    test_df = pd.read_csv(csv_test, header=None, names=cols)

    train_df["class_name"] = train_df["class_id"].map(id_to_name)
    test_df["class_name"] = test_df["class_id"].map(id_to_name)

    train_file_map = build_file_map(img_train_root)
    test_file_map = build_file_map(img_test_root)

    train_df["filepath"] = train_df["file_name"].map(train_file_map)
    test_df["filepath"] = test_df["file_name"].map(test_file_map)

    if verbose:
        print("train rows:", len(train_df), "with paths:", train_df["filepath"].notna().sum())
        print("test rows:", len(test_df), "with paths:", test_df["filepath"].notna().sum())

    return train_df, test_df, id_to_name


def df_to_arrays(df):
    df = df.dropna(subset=["filepath"]).reset_index(drop=True)
    files = df["filepath"].astype(str).tolist()
    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy()
    labels = (df["class_id"].astype(int) - 1).to_numpy()
    return files, boxes, labels


def get_mean_std_from_weights(weights_obj):
    t = weights_obj.transforms()
    if hasattr(t, "transforms"):
        for tr in t.transforms:
            if tr.__class__.__name__ == "Normalize":
                return tr.mean, tr.std
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def show_full_and_crop(df, n=3, seed=42, title_prefix="", pad_fill=0):
    sample = df.dropna(subset=["filepath"]).sample(n=n, random_state=seed)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (_, row) in enumerate(sample.iterrows()):
        img = read_image(str(row["filepath"]), mode=ImageReadMode.RGB)  # CHW uint8
        _, h, w = img.shape

        x1, y1, x2, y2 = clamp_bbox(row["x1"], row["y1"], row["x2"], row["y2"], w, h)

        # Column 2: raw bbox crop
        crop = img[:, y1:y2, x1:x2]
        _, ch, cw = crop.shape

        # Column 3: square crop from FULL image based on bbox center + S=max(w,h)
        sq, S, sq_coords, pads = square_crop_from_full(img, x1, y1, x2, y2, fill_value=pad_fill)
        _, sh, sw = sq.shape  # should be S,S

        # print resolutions
        print(
            f'{title_prefix}{row["class_name"]} | full: {w}x{h}px | '
            f'bbox: {cw}x{ch}px | square-from-full: {sw}x{sh}px (S={S}) | '
            f'pad(LR/TB)={pads[0]},{pads[1]}/{pads[2]},{pads[3]} | file: {row["filepath"]}'
        )

        img_np = img.permute(1, 2, 0).numpy()
        crop_np = crop.permute(1, 2, 0).numpy()
        sq_np = sq.permute(1, 2, 0).numpy()

        ax_full, ax_crop, ax_sq = axes[i]

        ax_full.imshow(img_np)
        ax_full.set_title(f'{title_prefix}{row["class_name"]}\nfull {w}x{h}px')
        ax_full.axis("off")

        ax_crop.imshow(crop_np)
        ax_crop.set_title(f'bbox crop {cw}x{ch}px')
        ax_crop.axis("off")

        ax_sq.imshow(sq_np)
        ax_sq.set_title(f'square-from-full {S}x{S}px')
        ax_sq.axis("off")

    plt.tight_layout()


def square_crop_from_full(img_chw, x1, y1, x2, y2, fill_value: int = 0):
    """
    Square side S = max(bbox_w, bbox_h), centered on bbox center.
    Crops from the original image, then places the crop in the CENTER of an SxS canvas
    so any padding is symmetric (top-bottom, left-right).
    """
    _, h, w = img_chw.shape

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    s = int(max(bbox_w, bbox_h))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    half = s / 2.0
    sx1 = int(round(cx - half))
    sy1 = int(round(cy - half))
    sx2 = sx1 + s
    sy2 = sy1 + s

    # Intersection of desired square with image bounds
    ix1 = max(0, sx1)
    iy1 = max(0, sy1)
    ix2 = min(w, sx2)
    iy2 = min(h, sy2)

    crop = img_chw[:, iy1:iy2, ix1:ix2]
    _, ch, cw = crop.shape

    # Create SxS canvas (black)
    canvas = img_chw.new_full((img_chw.shape[0], s, s), fill_value)

    # Paste crop into center of canvas -> symmetric padding
    off_x = (s - cw) // 2
    off_y = (s - ch) // 2
    canvas[:, off_y:off_y + ch, off_x:off_x + cw] = crop

    # Compute (approx) symmetric pad amounts for printing
    pad_left = off_x
    pad_right = s - cw - off_x
    pad_top = off_y
    pad_bottom = s - ch - off_y

    return canvas, s, (sx1, sy1, sx2, sy2), (pad_left, pad_right, pad_top, pad_bottom)

