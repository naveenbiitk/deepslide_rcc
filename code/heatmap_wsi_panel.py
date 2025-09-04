#!/usr/bin/env python3
import os, argparse, numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from models.attn_resnet18 import AttnResNet18

# ---------------- utils ----------------
def parse_roi(s):
    if not s: return None
    x,y,w,h = [int(v) for v in s.split(",")]
    return (x,y,w,h)

def tissue_ratio_rgb(rgb):
    """Skip near-white/low-saturation tiles (fast heuristic)."""
    arr = rgb.astype(np.uint8)
    not_white = np.any(arr < 230, axis=-1)
    r,g,b = arr[...,0].astype(np.float32)/255., arr[...,1]/255., arr[...,2]/255.
    mx = np.max(arr, axis=-1).astype(np.float32)/255.
    mn = np.min(arr, axis=-1).astype(np.float32)/255.
    sat = np.zeros_like(mx); nz = mx>0
    sat[nz] = (mx[nz]-mn[nz]) / (mx[nz] + 1e-6)
    saturated = sat > 0.08
    return (not_white & saturated).mean()

def box_blur_2d(x, k=9):
    if k <= 1: return x
    pad = k//2
    xpad = np.pad(x, ((0,0),(pad,pad)), mode='edge')
    ker = np.ones((k,), dtype=np.float32) / k
    h = np.apply_along_axis(lambda r: np.convolve(r, ker, mode='valid'), 1, xpad)
    xpad2 = np.pad(h, ((pad,pad),(0,0)), mode='edge')
    v = np.apply_along_axis(lambda c: np.convolve(c, ker, mode='valid'), 0, xpad2)
    return v

def colorize_heatmap(h01, cmap_name="jet"):
    c = cm.get_cmap(cmap_name)(h01)[...,:3]  # RGB [0..1]
    return (c*255).astype(np.uint8)

def overlay(orig_u8, heat_u8, alpha=0.40):
    return np.clip((1-alpha)*orig_u8.astype(np.float32) + alpha*heat_u8.astype(np.float32), 0, 255).astype(np.uint8)

# -------------- Grad-CAM hooks --------------
class GradCAM:
    def __init__(self, target_module):
        self.activations = None
        self.gradients = None
        self.h1 = target_module.register_forward_hook(self._save_act)
        self.h2 = target_module.register_full_backward_hook(self._save_grad)
    def _save_act(self, m, i, o): self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients = go[0].detach()
    def remove(self): self.h1.remove(); self.h2.remove()
    def cam(self):
        A, G = self.activations, self.gradients  # (B,C,H,W)
        B, C, H, W = A.shape
        w = G.mean(dim=(2,3), keepdim=True)           # (B,C,1,1)
        cam = torch.relu((w * A).sum(dim=1))          # (B,H,W)
        # normalize per-sample to [0,1]
        cam = (cam - cam.view(B,-1).min(1)[0].view(B,1,1)) / \
              (cam.view(B,-1).max(1)[0].view(B,1,1) - cam.view(B,-1).min(1)[0].view(B,1,1) + 1e-8)
        return cam

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_png", required=True)
    ap.add_argument("--ckpt", default="checkpoints_attn_bin/attn_bin_best.pt")
    ap.add_argument("--outdir", default="heatmaps_bin")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=112, help="<= tile for overlap; 112 reduces striping")
    ap.add_argument("--roi", type=str, default="", help="x,y,w,h crop rectangle on original WSI")
    ap.add_argument("--max_viz_dim", type=int, default=2048, help="downsample long side for viz")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--class_idx", type=int, default=1, help="1=Malignant")
    ap.add_argument("--tissue_min_frac", type=float, default=0.15, help="skip tiles with < this fraction tissue")
    ap.add_argument("--smooth_k", type=int, default=9, help="box blur kernel for smoothing Grad-CAM")
    ap.add_argument("--cmap", type=str, default="jet")
    ap.add_argument("--with_colorbar", action="store_true", help="add a colorbar next to Grad-CAM")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load image (optionally crop ROI)
    img_full = Image.open(args.wsi_png).convert("RGB")
    W0, H0 = img_full.size
    roi = parse_roi(args.roi)
    if roi:
        x,y,w,h = roi
        x = max(0, min(x, W0-1)); y = max(0, min(y, H0-1))
        w = max(1, min(w, W0-x)); h = max(1, min(h, H0-y))
        img = img_full.crop((x,y,x+w,y+h))
    else:
        img = img_full
    W, H = img.size

    # Viz scale
    scale = min(1.0, args.max_viz_dim / max(W,H))
    vizW, vizH = int(W*scale), int(H*scale)
    img_viz = img.resize((vizW, vizH), Image.BILINEAR)
    img_viz_u8 = np.array(img_viz)

    # Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AttnResNet18(num_classes=2, pretrained=False).to(device)
    sd = torch.load(args.ckpt, map_location=device)["model"]
    model.load_state_dict(sd, strict=True); model.eval()

    target_module = model.features[-1]  # resnet18.layer4
    gcam = GradCAM(target_module)

    normalize = transforms.Normalize(mean=[0.7488,0.6045,0.7521],
                                     std=[0.1571,0.1921,0.1504])
    tfm = transforms.Compose([transforms.Resize(args.tile),
                              transforms.CenterCrop(args.tile),
                              transforms.ToTensor(), normalize])

    # Grid with overlap
    tile, stride = args.tile, args.stride
    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))
    if xs[-1] != W - tile: xs.append(max(0, W - tile))
    if ys[-1] != H - tile: ys.append(max(0, H - tile))

    heat_cam = np.zeros((vizH, vizW), dtype=np.float32)
    count = np.zeros((vizH, vizW), dtype=np.float32)

    # Slide over tiles
    for y0 in ys:
        for x0 in xs:
            crop = img.crop((x0, y0, x0+tile, y0+tile))
            if tissue_ratio_rgb(np.array(crop)) < args.tissue_min_frac:
                continue
            xt = tfm(crop).unsqueeze(0).to(device)

            model.zero_grad(set_to_none=True)
            logits = model(xt)  # (1,2)

            # Grad-CAM wrt malignant logit
            logits[:, args.class_idx].sum().backward()
            cam7 = gcam.cam()[0]  # (7,7)
            cam224 = F.interpolate(cam7.unsqueeze(0).unsqueeze(0), size=(tile,tile),
                                   mode="bilinear", align_corners=False)[0,0].cpu().numpy()

            # paste into viz canvas
            vx, vy = int(x0*scale), int(y0*scale)
            vtw, vth = max(1,int(tile*scale)), max(1,int(tile*scale))
            cam_v = np.array(Image.fromarray((cam224*255).astype(np.uint8))
                             .resize((vtw, vth), Image.BILINEAR)).astype(np.float32)/255.0
            heat_cam[vy:vy+vth, vx:vx+vtw] += cam_v
            count[vy:vy+vth, vx:vx+vtw] += 1.0

    gcam.remove()

    # Average overlaps, smooth, normalize over tissue mask
    mask = count > 0
    if mask.any():
        heat_cam[mask] = heat_cam[mask] / count[mask]
        if args.smooth_k > 1:
            heat_cam = box_blur_2d(heat_cam, k=args.smooth_k)
        mn, mx = heat_cam[mask].min(), heat_cam[mask].max()
        heat_cam[mask] = (heat_cam[mask]-mn) / (mx-mn + 1e-8)

    cam_rgb   = colorize_heatmap(np.clip(heat_cam,0,1), cmap_name=args.cmap)
    overlay_u8 = overlay(img_viz_u8, cam_rgb, alpha=0.40)

    # Save individual images
    stem = os.path.splitext(os.path.basename(args.wsi_png))[0]
    tag = "_roi" if roi else ""
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    p_orig   = os.path.join(outdir, f"{stem}_orig{tag}.png")
    p_cam    = os.path.join(outdir, f"{stem}_gradcam{tag}.png")
    p_overlay= os.path.join(outdir, f"{stem}_overlay{tag}.png")
    Image.fromarray(img_viz_u8).save(p_orig)
    Image.fromarray(cam_rgb).save(p_cam)
    Image.fromarray(overlay_u8).save(p_overlay)

    # Panel: one row, three columns
    p_panel = os.path.join(outdir, f"{stem}{tag}_rowpanel.png")
    fig = plt.figure(figsize=(16, 5))
    axs = [fig.add_subplot(1,3,i+1) for i in range(3)]
    axs[0].imshow(img_viz_u8); axs[0].axis("off"); axs[0].set_title("Original", fontsize=12)
    axs[1].imshow(cam_rgb);    axs[1].axis("off"); axs[1].set_title("Grad-CAM (malignant evidence)", fontsize=12)
    axs[2].imshow(overlay_u8); axs[2].axis("off"); axs[2].set_title("Overlay", fontsize=12)

    # optional colorbar anchored to Grad-CAM axis
    if args.with_colorbar:
        norm01 = Normalize(vmin=0.0, vmax=1.0)
        sm = cm.ScalarMappable(norm=norm01, cmap=args.cmap)
        cax = inset_axes(axs[1], width="3%", height="80%", loc="center right", borderpad=1.0)
        cb = plt.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("Low  ⟶  High malignant evidence", fontsize=9)

    fig.tight_layout()
    fig.savefig(p_panel, dpi=180); plt.close(fig)

    print("Saved:")
    print(" ", p_orig)
    print(" ", p_cam)
    print(" ", p_overlay)
    print(" ", p_panel)

if __name__ == "__main__":
    main()
