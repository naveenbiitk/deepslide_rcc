#!/usr/bin/env python3
import os, csv, time, argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from models.pvt_binary import PVTBinary

def tissue_ratio_rgb(rgb):
    arr = np.asarray(rgb, dtype=np.uint8)
    not_white = np.any(arr < 230, axis=-1)
    a = arr.astype(np.float32)/255.0
    sat = (a.max(-1) - a.min(-1)) / (a.max(-1) + 1e-6)
    return float((not_white & (sat > 0.08)).mean())

def make_grid(W,H, tile, stride):
    xs = list(range(0, max(1, W-tile+1), stride))
    ys = list(range(0, max(1, H-tile+1), stride))
    if xs[-1] != W-tile: xs.append(max(0,W-tile))
    if ys[-1] != H-tile: ys.append(max(0,H-tile))
    return xs, ys

@torch.no_grad()
def infer_one_png(path, model, device, tfm, tile=224, stride=112, tissue_min_frac=0.15, bs=128):
    img = Image.open(path).convert("RGB")
    W,H = img.size
    xs, ys = make_grid(W,H,tile,stride)
    crops = []
    for y0 in ys:
        for x0 in xs:
            c = img.crop((x0,y0,x0+tile,y0+tile))
            if tissue_ratio_rgb(c) >= tissue_min_frac:
                crops.append(c)
    if len(crops)==0:
        return np.nan, 0, np.nan

    # warm sync
    if device.type=="cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()

    probs = []
    for i in range(0, len(crops), bs):
        batch = torch.stack([tfm(c) for c in crops[i:i+bs]]).to(device, non_blocking=True)
        pr = torch.sigmoid(model(batch)).squeeze(1).float().cpu().numpy()
        probs.append(pr)
    if device.type=="cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()

    p = np.concatenate(probs, axis=0)
    return float((t1-t0)*1000.0), int(len(p)), float(p.mean())  # ms, n_patches, mean prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_dir", required=True, help="folder of PNG WSIs (class subfolders OK)")
    ap.add_argument("--ckpt", default="checkpoints_pvt_bin/pvt_bin_best.pt")
    ap.add_argument("--out_csv", default="results_binary_wsi/inference_test_bin.csv")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=112)
    ap.add_argument("--tissue_min_frac", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mean", nargs=3, type=float, default=[0.7488,0.6045,0.7521])
    ap.add_argument("--std",  nargs=3, type=float, default=[0.1571,0.1921,0.1504])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    # model
    ckpt = torch.load(args.ckpt, map_location=device)
    model = PVTBinary(pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)

    tfm = transforms.Compose([
        transforms.Resize(args.tile),
        transforms.CenterCrop(args.tile),
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std),
    ])

    pngs = [p for p in Path(args.wsi_dir).rglob("*.png") if p.is_file()]
    rows = []
    for p in pngs:
        ms, n, prob = infer_one_png(p, model, device, tfm, args.tile, args.stride, args.tissue_min_frac, args.batch_size)
        slide_id = p.stem
        rows.append({"slide_id": slide_id, "path": str(p), "n_patches": n,
                     "inference_ms": round(ms,2), "prob_malignant": prob})

        print(f"{slide_id}: n={n}  ms={ms:.1f}  p_mal={prob:.3f}")

    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"[done] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
