from __future__ import annotations

from typing import Sequence, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def square_crop_from_full_chw(
    img_chw: torch.Tensor,
    bbox_xyxy,
    fill_value: int = 0,
    margin: float = 0.0,
):
    """
    UPDATED logic (centered padding):
      - S = ceil(max(bbox_w, bbox_h) * (1 + margin))
      - Compute desired SxS window centered on bbox center (in full-image coords)
      - Crop the intersection with the image bounds
      - Paste that crop into the CENTER of an SxS canvas filled with fill_value
        -> padding is symmetric (top≈bottom, left≈right)
    """
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    _, H, W = img_chw.shape

    bw = x2 - x1
    bh = y2 - y1
    S = int(np.ceil(max(bw, bh) * (1.0 + float(margin))))
    S = max(S, 1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    half = S / 2.0
    sx1 = int(round(cx - half))
    sy1 = int(round(cy - half))
    sx2 = sx1 + S
    sy2 = sy1 + S

    # Intersection of desired square with image
    ix1 = max(0, sx1)
    iy1 = max(0, sy1)
    ix2 = min(W, sx2)
    iy2 = min(H, sy2)

    crop = img_chw[:, iy1:iy2, ix1:ix2]
    _, ch, cw = crop.shape

    # Center the crop in an SxS canvas -> symmetric padding
    canvas = img_chw.new_full((img_chw.shape[0], S, S), fill_value)
    off_x = (S - cw) // 2
    off_y = (S - ch) // 2
    canvas[:, off_y:off_y + ch, off_x:off_x + cw] = crop

    return canvas


class CarsCsvDataset(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        boxes,
        labels,
        transform=None,
        crop_mode: Literal["none", "bbox", "square_from_full"] = "none",
        square_margin: float = 0.0,
        pad_fill: int = 0,
    ):
        self.files = list(files)
        self.boxes = np.asarray(boxes) if boxes is not None else None
        self.labels = np.asarray(labels)
        self.transform = transform
        self.crop_mode = crop_mode
        self.square_margin = float(square_margin)
        self.pad_fill = pad_fill

        if self.crop_mode in ("bbox", "square_from_full") and self.boxes is None:
            raise ValueError("boxes must be provided when crop_mode is 'bbox' or 'square_from_full'.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = read_image(self.files[idx], mode=ImageReadMode.RGB)  # CHW uint8
        label = int(self.labels[idx])

        if self.crop_mode != "none":
            x1, y1, x2, y2 = self.boxes[idx]
            _, h, w = img.shape
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

            if self.crop_mode == "bbox":
                img = img[:, y1:y2, x1:x2]
            elif self.crop_mode == "square_from_full":
                img = square_crop_from_full_chw(
                    img,
                    (x1, y1, x2, y2),
                    fill_value=self.pad_fill,
                    margin=self.square_margin,
                )

        if self.transform:
            img = self.transform(img)

        return img, label
