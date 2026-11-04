"""UNI2-H features of five spot-centered H&E fields of view (tonsil).

For every spot, a square crop of side `patch_size` full-resolution pixels is taken
around the spot center (zero-padded at the image border), resized to 224 px
(bicubic, ImageNet normalization) and encoded by UNI2-H (1,536-d CLS output).
Paper fields: 256, 512, 768, 1,024 and 1,280 px at 0.215 um/px (55-275 um).

UNI2-H is a gated model: request access at https://huggingface.co/MahmoodLab/UNI2-h
and run `huggingface-cli login` before using this script.

    python preprocessing/extract_features_tonsil_uni2h.py \
        --image raw/tonsil/tonsilwoman/Human_Tonsil_tissue_image.tif \
        --positions data/tonsil/female/coord/tissue_positions_intersect_ordered.csv \
        --out-dir data/tonsil/female/image_uni2h_patch96_to_1024_step32 --prefix tonsilwoman_uni2h
(the training loader looks for <prefix>_patch<size>.npy in the image_uni2h* folders of each section)
"""
import argparse
import os

import numpy as np
import pandas as pd
import tifffile
import timm
import torch
import zarr
from PIL import Image
from timm.data.transforms_factory import create_transform

TIMM_KWARGS = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                   mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                   act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)


def crop(arr, x, y, size):
    half = size // 2
    x0, x1, y0, y1 = x - half, x + half, y - half, y + half
    h, w = arr.shape[:2]
    xx0, xx1, yy0, yy1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    if xx1 > xx0 and yy1 > yy0:
        sub = np.asarray(arr[yy0:yy1, xx0:xx1, :])
        sub = sub[..., :3] if sub.shape[-1] >= 3 else np.repeat(sub, 3, axis=-1)
        patch[yy0 - y0:yy0 - y0 + sub.shape[0], xx0 - x0:xx0 - x0 + sub.shape[1]] = sub
    return patch


def spot_coords(csv_path):
    df = pd.read_csv(csv_path)
    low = [c.lower() for c in df.columns]
    if "pxl_col_in_fullres" in low and "pxl_row_in_fullres" in low:
        xc, yc = df.columns[low.index("pxl_col_in_fullres")], df.columns[low.index("pxl_row_in_fullres")]
    else:
        yc, xc = df.columns[-2], df.columns[-1]
    return (df[df.columns[0]].astype(str).tolist(), df[xc].astype(float).round().astype(int).tolist(),
            df[yc].astype(float).round().astype(int).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--patch-sizes", nargs="*", type=int, default=[256, 512, 768, 1024, 1280])
    ap.add_argument("--batch-size", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **TIMM_KWARGS).to(device).eval()
    transform = create_transform(input_size=(3, 224, 224), is_training=False, mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225), interpolation="bicubic", crop_pct=1.0)

    os.makedirs(args.out_dir, exist_ok=True)
    barcodes, xs, ys = spot_coords(args.positions)
    pd.DataFrame({"barcode": barcodes}).to_csv(os.path.join(args.out_dir, f"{args.prefix}_barcodes.csv"), index=False)
    with tifffile.TiffFile(args.image) as tif:
        arr = zarr.open(tif.series[0].aszarr(), mode="r")
        for size in args.patch_sizes:
            out = os.path.join(args.out_dir, f"{args.prefix}_patch{size}.npy")
            if os.path.exists(out):
                continue
            feats = []
            for s in range(0, len(barcodes), args.batch_size):
                idx = range(s, min(s + args.batch_size, len(barcodes)))
                batch = torch.stack([transform(Image.fromarray(crop(arr, xs[i], ys[i], size))) for i in idx]).to(device)
                with torch.no_grad():
                    feats.append(model(batch).cpu().numpy())
            np.save(out, np.concatenate(feats, 0))
            print("saved", out)


if __name__ == "__main__":
    main()
