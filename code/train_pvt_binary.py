#!/usr/bin/env python3
import os, argparse, time, json
from pathlib import Path
import numpy as np
from PIL import Image

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from models.pvt_binary import PVTBinary

BIN_MAP = {
    # 0 = Benign, 1 = Malignant
    "Benign": 0, "Oncocytoma": 0,
    "Clearcell": 1, "Papillary": 1, "Chromophobe": 1,
}

def list_images(root):
    root = Path(root)
    items = []
    for cls_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        cls = cls_dir.name
        y = BIN_MAP.get(cls, None)
        if y is None:  # ignore unexpected folders
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in [".jpg",".jpeg",".png",".tif",".tiff",".bmp"]:
                items.append((str(p), int(y)))
    return items

class PatchSet(Dataset):
    def __init__(self, items, img_size=224, train=False, mean=(0.7488,0.6045,0.7521), std=(0.1571,0.1921,0.1504)):
        self.items = items
        if train:
            self.tf = T.Compose([
                T.Resize((img_size,img_size)),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(0.2,0.2,0.2,0.0),
                T.ToTensor(),
                T.Normalize(mean,std),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((img_size,img_size)),
                T.ToTensor(),
                T.Normalize(mean,std),
            ])

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = Image.open(p).convert("RGB")
        x = self.tf(im)
        return x, torch.tensor([float(y)], dtype=torch.float32)

def set_seed(s=1337):
    import random
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False

def evaluate(model, dl, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            logit = model(x)
            p = torch.sigmoid(logit).squeeze(1).cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(y.squeeze(1).cpu().numpy().tolist())
    y = np.array(labels, dtype=int); p = np.array(probs, dtype=float)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    pred = (p >= 0.65).astype(int)  # match your τ=0.65 reporting
    acc = accuracy_score(y, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(auc=auc, acc=acc, prec=prec, rec=rec, f1=f1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="e.g., train_folder_cap20k/train")
    ap.add_argument("--val_dir",   required=True, help="e.g., train_folder_cap20k/val")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-5)  # small LR (we’re finetuning)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=0.85)  # ExponentialLR decay
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--out_dir", default="checkpoints_pvt_bin")
    ap.add_argument("--resume", default="", help="path to .pt to resume (optional)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    set_seed(1337)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_items = list_images(args.train_dir)
    val_items   = list_images(args.val_dir)

    print(f"Train patches: {len(train_items):,} | Val patches: {len(val_items):,}")
    # Class balance for pos_weight
    y_tr = np.array([y for _,y in train_items], int)
    pos = y_tr.sum(); neg = len(y_tr) - pos
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)

    tr_ds = PatchSet(train_items, img_size=args.img_size, train=True)
    va_ds = PatchSet(val_items,   img_size=args.img_size, train=False)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=8, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = PVTBinary(pretrained=True, dropout=args.dropout).to(device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=args.gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    os.makedirs(args.out_dir, exist_ok=True)
    best_auc = -1.0; start_ep = 1

    # Optional resume
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_ep = ckpt.get("epoch", 1) + 1
        best_auc = float(ckpt.get("best_auc", -1.0))
        print(f"[resume] from {args.resume} @ epoch {start_ep-1} (best_auc={best_auc:.3f})")

    for ep in range(start_ep, args.epochs + 1):
        model.train(); running = 0.0; n = 0
        t0 = time.time()
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logit = model(x)
                loss = crit(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0); n += x.size(0)
        tr_loss = running / max(1, n)
        sched.step()

        # Validation
        metrics = evaluate(model, va_dl, device)
        msg = (f"[ep {ep:03d}] train_loss={tr_loss:.4f} | "
               f"val_auc={metrics['auc']:.3f} acc={metrics['acc']:.3f} "
               f"prec={metrics['prec']:.3f} rec={metrics['rec']:.3f} f1={metrics['f1']:.3f} | "
               f"lr={opt.param_groups[0]['lr']:.2e} | dt={(time.time()-t0):.1f}s")
        print(msg)

        # Save best by AUC
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            ckpt = {
                "epoch": ep, "best_auc": best_auc,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, f"{args.out_dir}/pvt_bin_best.pt")
        # Save periodic checkpoints if you like
        if ep % 5 == 0:
            torch.save({"epoch": ep, "model": model.state_dict()}, f"{args.out_dir}/pvt_bin_ep{ep}.pt")

    print(f"[done] best val AUC = {best_auc:.3f} (saved to {args.out_dir}/pvt_bin_best.pt)")
    with open(f"{args.out_dir}/train_summary.json","w") as f:
        json.dump({"best_val_auc": best_auc}, f, indent=2)

if __name__ == "__main__":
    main()
