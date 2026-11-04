# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


BASE = Path(os.environ.get("MORPHOPROT_DATA", "data")) / "liver"
GENE_DIR = BASE / "_scheme2_log1p_zscore_hvg3500_force_marker_in_tissue1"
SAMPLE_DIR = BASE / "samples"
OUT = BASE / "_hires_resnet18_multiscale512"
SCALES = [16, 32, 64, 96, 128]
ENCODER_INPUT_SIZE = 224


def read_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_sample_coords(sample: str) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(SAMPLE_DIR / sample / "tissue_positions_list_with_hires_coords.csv")
    key_col = "barcode_raw" if "barcode_raw" in df.columns else "barcode"
    return {str(b): (float(x), float(y)) for b, x, y in zip(df[key_col], df["pxl_col_in_hires_raw"], df["pxl_row_in_hires_raw"])}


def load_images(max_half: int) -> dict[str, np.ndarray]:
    images = {}
    for sdir in sorted(SAMPLE_DIR.iterdir()):
        if not sdir.is_dir():
            continue
        img = np.asarray(Image.open(sdir / "tissue_hires_image.png").convert("RGB"), dtype=np.uint8)
        images[sdir.name] = np.pad(img, ((max_half, max_half), (max_half, max_half), (0, 0)), mode="reflect")
    return images


def extract_split(
    barcodes: list[str],
    scale: int,
    max_half: int,
    images: dict[str, np.ndarray],
    coords: dict[str, dict[str, tuple[float, float]]],
    feature_extractor: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    half = scale // 2
    feats = np.zeros((len(barcodes), 512), dtype=np.float32)
    patches = []
    valid_idx = []
    for i, full_bc in enumerate(barcodes):
        sample, raw_bc = full_bc.split(":", 1)
        if sample not in images or raw_bc not in coords[sample]:
            continue
        x, y = coords[sample][raw_bc]
        xi = int(round(x)) + max_half
        yi = int(round(y)) + max_half
        patch = images[sample][yi - half : yi + half, xi - half : xi + half, :]
        if patch.shape[:2] != (scale, scale):
            continue
        patches.append(patch)
        valid_idx.append(i)

    if not patches:
        return feats, len(barcodes)

    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            end = min(len(patches), start + batch_size)
            # Preserve the sampled field of view; resize the whole crop, without center cropping.
            arr = np.stack([
                np.asarray(Image.fromarray(p).resize(
                    (ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE), Image.Resampling.BILINEAR))
                for p in patches[start:end]
            ]).astype(np.float32) / 255.0
            arr = np.transpose((arr - mean) / std, (0, 3, 1, 2))
            x = torch.from_numpy(arr).to(device)
            out = feature_extractor(x).squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32)
            feats[np.asarray(valid_idx[start:end], dtype=np.int64)] = out
    return feats, len(barcodes) - len(valid_idx)


def main() -> None:
    global BASE, GENE_DIR, SAMPLE_DIR, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default=str(BASE))
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    BASE = Path(args.processed_dir)
    GENE_DIR = BASE / "_scheme2_log1p_zscore_hvg3500_force_marker_in_tissue1"
    SAMPLE_DIR = BASE / "samples"
    OUT = BASE / "_hires_resnet18_multiscale512"

    OUT.mkdir(parents=True, exist_ok=True)
    train_barcodes = read_lines(GENE_DIR / "train_barcodes.txt")
    test_barcodes = read_lines(GENE_DIR / "test_barcodes.txt")

    max_half = max(SCALES) // 2
    images = load_images(max_half)
    coords = {sample: load_sample_coords(sample) for sample in images}

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).eval()
    mean = np.array(weights.transforms().mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(weights.transforms().std, dtype=np.float32).reshape(1, 1, 1, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    batch_size = args.batch_size or (256 if device.type == "cuda" else 64)
    summary = {
        "device": str(device),
        "scales": SCALES,
        "feature_dim": 512,
        "encoder_input_size": [ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE],
        "resize_interpolation": "PIL bilinear",
        "normalization": "ImageNet mean/std after resize and scaling to [0, 1]",
        "train_n": len(train_barcodes),
        "test_n": len(test_barcodes),
        "outputs": {},
    }
    for ps in SCALES:
        train_feat, miss_train = extract_split(train_barcodes, ps, max_half, images, coords, feature_extractor, mean, std, device, batch_size)
        test_feat, miss_test = extract_split(test_barcodes, ps, max_half, images, coords, feature_extractor, mean, std, device, batch_size)
        p_train = OUT / f"he_resnet18_ps{ps}_train_512.npy"
        p_test = OUT / f"he_resnet18_ps{ps}_test_512.npy"
        np.save(p_train, train_feat)
        np.save(p_test, test_feat)
        print(f"[OK] ps{ps}: train={train_feat.shape}, test={test_feat.shape}, missing_train={miss_train}, missing_test={miss_test}")
        summary["outputs"][f"ps{ps}"] = {
            "train": str(p_train),
            "test": str(p_test),
            "missing_train": int(miss_train),
            "missing_test": int(miss_test),
        }
    (OUT / "train_barcodes.txt").write_text("\n".join(train_barcodes) + "\n", encoding="utf-8")
    (OUT / "test_barcodes.txt").write_text("\n".join(test_barcodes) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
