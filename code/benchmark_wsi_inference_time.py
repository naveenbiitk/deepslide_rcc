#!/usr/bin/env python3
import os, glob, time, csv, argparse, math
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# import your model (same as you used before)
from models.attn_resnet18 import AttnResNet18

# ---------- fast tissue mask (RGB PNGs) ----------
def tissue_ratio_rgb(rgb_u8: np.ndarray) -> float:
    # not-white & sufficiently saturated
    not_white = np.any(rgb_u8 < 230, axis=-1)
    arr = rgb_u8.astype(np.float32)/255.0
    mx = arr.max(axis=-1); mn = arr.min(axis=-1)
    sat = (mx - mn) / (mx + 1e-6)
    saturated = sat > 0.08
    return float((not_white & saturated).mean())

# ---------- slide tiling ----------
def make_grid(W, H, tile, stride):
    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))
    if xs[-1] != W - tile: xs.append(max(0, W - tile))
    if ys[-1] != H - tile: ys.append(max(0, H - tile))
    return xs, ys

# ---------- main inference for one slide ----------
@torch.no_grad()
def infer_one_slide_png(path, model, device, tfm, tile=224, stride=112,
                        tissue_min_frac=0.15, batch_size=128, class_idx=1):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    xs, ys = make_grid(W, H, tile, stride)

    patches = []
    for y0 in ys:
        for x0 in xs:
            crop = img.crop((x0, y0, x0+tile, y0+tile))
            if tissue_ratio_rgb(np.array(crop)) < tissue_min_frac:
                continue
            patches.append(crop)

    n_patches = len(patches)
    if n_patches == 0:
        return 0.0, 0, float("nan")  # no tissue; guard

    # stack in batches
    def gen_batches(seq, bs):
        for i in range(0, len(seq), bs):
            yield seq[i:i+bs]

    probs = []
    # warmup sync for fair timing on GPU
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for chunk in gen_batches(patches, batch_size):
        # transform to tensor batch
        tens = torch.stack([tfm(p) for p in chunk]).to(device, non_blocking=True)
        logits = model(tens)  # [B,2]
        probs_mal = torch.softmax(logits, dim=1)[:, class_idx]
        probs.append(probs_mal.detach().cpu())

    # sync & stop timer
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    probs = torch.cat(probs).numpy()
    slide_prob = float(probs.mean())  # simple mean aggregator
    elapsed_ms = (t1 - t0) * 1000.0
    return elapsed_ms, n_patches, slide_prob

def list_pngs(root: str, recursive=True, pattern="*.png"):
    root = Path(root)
    if root.is_file() and root.suffix.lower()==".png":
        return [str(root)]
    if recursive:
        return [str(p) for p in root.rglob(pattern)]
    else:
        return [str(p) for p in root.glob(pattern)]

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wsi_png", help="single PNG slide")
    src.add_argument("--wsi_dir", help="directory containing PNG slides (class subfolders ok)")

    ap.add_argument("--ckpt", default="checkpoints_attn_bin/attn_bin_best.pt")
    ap.add_argument("--out_csv", default="results_binary_wsi/inference_speed.csv")

    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=112, help="<= tile for overlap")
    ap.add_argument("--tissue_min_frac", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--class_idx", type=int, default=1, help="1 = Malignant")

    ap.add_argument("--mean", nargs=3, type=float, default=[0.7488, 0.6045, 0.7521])
    ap.add_argument("--std",  nargs=3, type=float, default=[0.1571, 0.1921, 0.1504])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # device & model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AttnResNet18(num_classes=2, pretrained=False).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)

    # transforms (resize->center crop to 224; same stats as training)
    tfm = transforms.Compose([
        transforms.Resize(args.tile),
        transforms.CenterCrop(args.tile),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ])

    # collect slides
    if args.wsi_png:
        slides = [args.wsi_png]
    else:
        slides = list_pngs(args.wsi_dir, recursive=True, pattern="*.png")
        slides = [s for s in slides if os.path.getsize(s) > 0]

    if len(slides) == 0:
        print("No PNG slides found.")
        return

    # optional warmup (first slide, not logged)
    _ = infer_one_slide_png(slides[0], model, device, tfm,
                            tile=args.tile, stride=args.stride,
                            tissue_min_frac=args.tissue_min_frac,
                            batch_size=args.batch_size, class_idx=args.class_idx)

    # run & log
    rows = []
    for p in slides:
        ms, n_patches, prob = infer_one_slide_png(
            p, model, device, tfm,
            tile=args.tile, stride=args.stride,
            tissue_min_frac=args.tissue_min_frac,
            batch_size=args.batch_size, class_idx=args.class_idx
        )
        slide_id = Path(p).stem
        mspp = (ms / n_patches) if n_patches > 0 else float("nan")
        rows.append({
            "slide_id": slide_id,
            "path": p,
            "n_patches": n_patches,
            "inference_ms": round(ms, 2),
            "ms_per_patch": round(mspp, 4),
            "mean_prob_malignant": round(prob, 6),
            "device": str(device),
            "batch_size": args.batch_size,
            "tile": args.tile,
            "stride": args.stride,
            "tissue_min_frac": args.tissue_min_frac,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"{slide_id}: {n_patches} patches, {ms:.1f} ms "
              f"({mspp:.3f} ms/patch), p_mal={prob:.3f}")

    # write CSV
    header = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows: w.writerow(r)

    # summary stats
    ms_all = np.array([r["inference_ms"] for r in rows], dtype=float)
    mspp_all = np.array([r["ms_per_patch"] for r in rows], dtype=float)
    print("\n=== Per-slide latency (ms) ===")
    print(f"mean {ms_all.mean():.0f} | p50 {np.percentile(ms_all,50):.0f} | "
          f"p90 {np.percentile(ms_all,90):.0f} | min {ms_all.min():.0f} | max {ms_all.max():.0f}")
    print("=== Per-patch latency (ms) ===")
    print(f"mean {mspp_all.mean():.3f} | p50 {np.percentile(mspp_all,50):.3f} | "
          f"p90 {np.percentile(mspp_all,90):.3f} | min {mspp_all.min():.3f} | max {mspp_all.max():.3f}")
    print(f"\nWrote {args.out_csv}")

if __name__ == "__main__":
    main()
