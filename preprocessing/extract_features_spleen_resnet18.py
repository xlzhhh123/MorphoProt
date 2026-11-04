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

BASE = Path(os.environ.get("MORPHOPROT_DATA", "data")) / "spleen"
ENCODER_INPUT_SIZE = 224


def load_coords(ds_dir: Path) -> dict[str, tuple[float, float]]:
    tp = ds_dir / "tissue_positions_list_with_hires_coords.csv"
    if not tp.exists():
        raise FileNotFoundError(f"missing file: {tp}")
    df = pd.read_csv(tp)
    return {
        str(b): (float(x), float(y))
        for b, x, y in zip(df["barcode"], df["pxl_col_in_hires"], df["pxl_row_in_hires"])
    }


def load_hires_image(ds_dir: Path, pad: int) -> np.ndarray:
    img = np.asarray(Image.open(ds_dir / "tissue_hires_image.png").convert("RGB"), dtype=np.uint8)
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def extract_features(
    barcodes: list[str],
    coord_map: dict[str, tuple[float, float]],
    img_pad: np.ndarray,
    scale: int,
    max_half: int,
    feature_extractor: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    half = scale // 2
    n = len(barcodes)
    feats = np.zeros((n, 512), dtype=np.float32)

    patches = []
    valid_idx = []
    for i, bc in enumerate(barcodes):
        if bc not in coord_map:
            continue
        x, y = coord_map[bc]
        xi = int(round(x)) + max_half
        yi = int(round(y)) + max_half
        patch = img_pad[yi - half : yi + half, xi - half : xi + half, :]
        if patch.shape[0] != scale or patch.shape[1] != scale:
            continue
        patches.append(patch)
        valid_idx.append(i)

    if not patches:
        return feats, n

    with torch.no_grad():
        for s in range(0, len(patches), batch_size):
            e = min(len(patches), s + batch_size)
            # Preserve the sampled field of view; resize the whole crop, without center cropping.
            arr = np.stack([
                np.asarray(Image.fromarray(p).resize(
                    (ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE), Image.Resampling.BILINEAR))
                for p in patches[s:e]
            ]).astype(np.float32) / 255.0
            arr = np.transpose((arr - mean) / std, (0, 3, 1, 2))
            x = torch.from_numpy(arr).to(device)
            out = feature_extractor(x).squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32)
            feats[np.array(valid_idx[s:e], dtype=np.int64)] = out

    missing = n - len(valid_idx)
    return feats, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--proc-dir", type=Path, default=None,
                        help="RNA folder with train/test_barcodes.txt (default: <base>/_scheme2_log1p_zscore_hvg3500_force_marker_in_tissue1)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder (default: <base>/_hires_resnet18_multiscale512)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.proc_dir is None:
        args.proc_dir = args.base / "_scheme2_log1p_zscore_hvg3500_force_marker_in_tissue1"
    if args.out_dir is None:
        args.out_dir = args.base / "_hires_resnet18_multiscale512"

    base = args.base
    proc_dir = args.proc_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_barcodes = [x.strip() for x in (proc_dir / "train_barcodes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    test_barcodes = [x.strip() for x in (proc_dir / "test_barcodes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]

    train_ds = base / "Dataset1_Mouse_Spleen1"
    test_ds = base / "Dataset2_Mouse_Spleen2"

    scales = [16, 32, 64, 96, 128]
    max_half = max(scales) // 2

    coords_train = load_coords(train_ds)
    coords_test = load_coords(test_ds)
    img_train_pad = load_hires_image(train_ds, max_half)
    img_test_pad = load_hires_image(test_ds, max_half)

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).eval()
    mean = np.array(weights.transforms().mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(weights.transforms().std, dtype=np.float32).reshape(1, 1, 1, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)

    summary: dict[str, object] = {
        "scales": scales,
        "feature_dim": 512,
        "encoder_input_size": [ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE],
        "resize_interpolation": "PIL bilinear",
        "normalization": "ImageNet mean/std after resize and scaling to [0, 1]",
        "device": str(device),
        "train_n": len(train_barcodes),
        "test_n": len(test_barcodes),
        "outputs": {},
    }

    for ps in scales:
        f_train, miss_train = extract_features(
            train_barcodes,
            coords_train,
            img_train_pad,
            ps,
            max_half,
            feature_extractor,
            mean,
            std,
            device,
            args.batch_size,
        )
        f_test, miss_test = extract_features(
            test_barcodes,
            coords_test,
            img_test_pad,
            ps,
            max_half,
            feature_extractor,
            mean,
            std,
            device,
            args.batch_size,
        )

        p_tr = out_dir / f"he_resnet18_ps{ps}_train_512.npy"
        p_te = out_dir / f"he_resnet18_ps{ps}_test_512.npy"
        np.save(p_tr, f_train)
        np.save(p_te, f_test)

        print(f"[OK] ps{ps}: train={f_train.shape}, test={f_test.shape}, missing_train={miss_train}, missing_test={miss_test}")
        summary["outputs"][f"ps{ps}"] = {
            "train": str(p_tr),
            "test": str(p_te),
            "train_shape": [int(f_train.shape[0]), int(f_train.shape[1])],
            "test_shape": [int(f_test.shape[0]), int(f_test.shape[1])],
            "missing_train": int(miss_train),
            "missing_test": int(miss_test),
        }

    (out_dir / "train_barcodes.txt").write_text("\n".join(train_barcodes) + "\n", encoding="utf-8")
    (out_dir / "test_barcodes.txt").write_text("\n".join(test_barcodes) + "\n", encoding="utf-8")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main()
