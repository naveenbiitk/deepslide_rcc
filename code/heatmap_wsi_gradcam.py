#!/usr/bin/env python3
import os, math, argparse
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from models.attn_resnet18 import AttnResNet18

# ---- Grad-CAM hooks on the backbone last conv block (layer4) ----
class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self.h1 = target_module.register_forward_hook(self._save_act)
        self.h2 = target_module.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        # out: (N, C=512, 7, 7)
        self.activations = out.detach()

    def _save_grad(self, module, grad_input, grad_output):
        # grad_output[0]: (N, 512, 7, 7)
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.h1.remove(); self.h2.remove()

    def cam(self, class_idx=1):
        # weights: channel-wise pooled grads
        # activations: (N,C,H,W); gradients: (N,C,H,W)
        assert self.activations is not None and self.gradients is not None
        B, C, H, W = self.activations.shape
        weights = self.gradients.mean(dim=(2,3), keepdim=True)         # (B,C,1,1)
        cam = torch.relu((weights * self.activations).sum(dim=1))      # (B,H,W)
        # normalize each sample to 0..1
        cam_min = cam.view(B,-1).min(dim=1)[0].view(B,1,1)
        cam_max = cam.view(B,-1).max(dim=1)[0].view(B,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam  # (B,H,W) in [0,1]

def make_heat_overlay(orig_rgb, heat_01, cmap='jet', alpha=0.40):
    """orig_rgb: (H,W,3) uint8, heat_01: (H,W) float [0,1]"""
    import matplotlib.cm as cm
    color = (cm.get_cmap(cmap)(heat_01)[...,:3] * 255).astype(np.uint8)    # RGB
    overlay = ( (1-alpha) * orig_rgb.astype(np.float32) + alpha * color.astype(np.float32) ).clip(0,255).astype(np.uint8)
    return color, overlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_png", required=True, help="Path to a single PNG WSI")
    ap.add_argument("--ckpt", default="checkpoints_attn_bin/attn_bin_best.pt")
    ap.add_argument("--outdir", default="heatmaps_bin")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224, help="112 for denser map; 224 for speed")
    ap.add_argument("--max_viz_dim", type=int, default=2048, help="downsample long side for visualization")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--class_idx", type=int, default=1, help="1=Malignant")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- load image & set viz scale ---
    img = Image.open(args.wsi_png).convert("RGB")
    W, H = img.size
    scale = min(1.0, args.max_viz_dim / max(W,H))
    vizW, vizH = int(W*scale), int(H*scale)
    img_viz = img.resize((vizW, vizH), Image.BILINEAR)
    img_viz_np = np.array(img_viz)  # (vizH, vizW, 3) uint8

    # --- model / target layer (backbone.layer4) ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AttnResNet18(num_classes=2, pretrained=False).to(device)
    sd = torch.load(args.ckpt, map_location=device)["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    # target module for Grad-CAM
    target_module = model.features[-1]  # resnet18.layer4
    gcam = GradCAM(model, target_module)

    # --- transforms (use your dataset stats) ---
    normalize = transforms.Normalize(mean=[0.7488,0.6045,0.7521],
                                     std=[0.1571,0.1921,0.1504])
    to_tensor = transforms.Compose([transforms.Resize(args.tile),
                                    transforms.CenterCrop(args.tile),
                                    transforms.ToTensor(), normalize])

    # --- accumulators on viz canvas ---
    heat = np.zeros((vizH, vizW), dtype=np.float32)
    count = np.zeros((vizH, vizW), dtype=np.float32)

    # --- slide over original-resolution grid, place into viz canvas ---
    tile = args.tile
    stride = args.stride
    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))
    if xs[-1] != W - tile: xs.append(max(0, W - tile))
    if ys[-1] != H - tile: ys.append(max(0, H - tile))

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            crop = img.crop((x, y, x+tile, y+tile))  # 224x224
            x_tensor = to_tensor(crop).unsqueeze(0).to(device)  # (1,3,224,224)

            # forward + backward for Grad-CAM on class_idx
            model.zero_grad(set_to_none=True)
            logits = model(x_tensor)                    # (1,2)
            score = logits[:, args.class_idx].sum()
            score.backward()

            cam7 = gcam.cam(class_idx=args.class_idx)[0]         # (7,7) float
            cam224 = F.interpolate(cam7.unsqueeze(0).unsqueeze(0), size=(tile,tile),
                                   mode="bilinear", align_corners=False)[0,0].cpu().numpy()

            # paste onto viz canvas at scaled coords
            vx, vy = int(x*scale), int(y*scale)
            vtile = (max(1,int(tile*scale)), max(1,int(tile*scale)))
            cam_v = np.array(Image.fromarray((cam224*255).astype(np.uint8)).resize(vtile, Image.BILINEAR)).astype(np.float32)/255.0

            heat[vy:vy+vtile[1], vx:vx+vtile[0]] += cam_v
            count[vy:vy+vtile[1], vx:vx+vtile[0]] += 1.0

    gcam.remove()

    # average over overlaps and normalize 0..1
    mask = count > 0
    heat[mask] = heat[mask] / count[mask]
    if mask.any():
        hmin, hmax = heat[mask].min(), heat[mask].max()
        heat[mask] = (heat[mask]-hmin) / (hmax-hmin + 1e-8)

    # --- colorize + overlay ---
    heat_color, overlay = make_heat_overlay(img_viz_np, heat, cmap='jet', alpha=0.40)

    # --- save outputs ---
    stem = os.path.splitext(os.path.basename(args.wsi_png))[0]
    p_orig   = os.path.join(args.outdir, f"{stem}_orig.png")
    p_heat   = os.path.join(args.outdir, f"{stem}_heat.png")
    p_overlay= os.path.join(args.outdir, f"{stem}_overlay.png")
    Image.fromarray(img_viz_np).save(p_orig)
    Image.fromarray(heat_color).save(p_heat)
    Image.fromarray(overlay).save(p_overlay)

    # combined 3-row panel
    fig = plt.figure(figsize=(6, 10))
    for i, (title, arr) in enumerate([("Original", img_viz_np),
                                      ("Grad-CAM (Malignant)", heat_color),
                                      ("Overlay", overlay)]):
        ax = fig.add_subplot(3,1,i+1)
        ax.imshow(arr); ax.axis("off"); ax.set_title(title, fontsize=12)
    fig.tight_layout()
    p_panel = os.path.join(args.outdir, f"{stem}_panel.png")
    fig.savefig(p_panel, dpi=180); plt.close(fig)

    print("Saved:")
    print(" ", p_orig)
    print(" ", p_heat)
    print(" ", p_overlay)
    print(" ", p_panel)

if __name__ == "__main__":
    main()
