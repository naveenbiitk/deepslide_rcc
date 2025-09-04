#!/usr/bin/env python3
import os, time, argparse, csv, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.attn_resnet18 import AttnResNet18

def get_dls(root, bs, num_workers=8):
    norm = transforms.Normalize(mean=[0.7488,0.6045,0.7521],
                                std=[0.1571,0.1921,0.1504])
    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.5,0.5,0.5,0.2),
        transforms.ToTensor(), norm
    ])
    tfm_val = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), norm
    ])
    tr = datasets.ImageFolder(os.path.join(root,"train"), transform=tfm_train)
    va = datasets.ImageFolder(os.path.join(root,"val"),   transform=tfm_val)

    assert set(tr.classes) == {"Benign","Malignant"}, f"Train classes: {tr.classes}"
    assert set(va.classes) == {"Benign","Malignant"}, f"Val classes: {va.classes}"

    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=bs, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    return tr, va, dl_tr, dl_va

def build_opt_sched(model, args, total_epochs):
    # optimizer
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler with warmup via LambdaLR
    we = max(0, int(args.warmup_epochs))
    ws = float(args.warmup_start_factor)

    def exp_after(ep):
        # after warmup, exponential decay by gamma each epoch
        return args.gamma ** max(0, ep - we)

    def cosine_after(ep):
        # cosine from ep=we .. total_epochs
        if ep < we: return 1.0  # not used during warmup branch
        t = ep - we
        T = max(1, total_epochs - we)
        return 0.5 * (1.0 + math.cos(math.pi * t / T))

    def lr_lambda(ep):
        # ep is epoch index starting at 0
        if we > 0 and ep < we:
            # linear warmup from ws to 1.0 across warmup epochs
            return ws + (1.0 - ws) * ((ep + 1) / we)
        # post-warmup
        return exp_after(ep) if args.sched == "exp" else cosine_after(ep)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    return opt, sched

def save_ckpt(path, model, epoch, acc, opt=None, sched=None, best_acc=None):
    pkg = {"model": model.state_dict(), "epoch": epoch, "acc": acc}
    if opt is not None:   pkg["opt"] = opt.state_dict()
    if sched is not None: pkg["sched"] = sched.state_dict()
    if best_acc is not None: pkg["best_acc"] = best_acc
    torch.save(pkg, path)

def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tr, va, dl_tr, dl_va = get_dls(args.data_root, args.batch_size, args.num_workers)
    print("Classes:", tr.classes, "num_train_imgs:", len(tr.samples), "num_val_imgs:", len(va.samples))

    model = AttnResNet18(num_classes=2, pretrained=not args.no_imagenet).to(device)

    opt, sched = build_opt_sched(model, args, total_epochs=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    crit = nn.CrossEntropyLoss()

    start_epoch = 1
    best_acc = 0.0
    last_val_acc = 0.0

    # resume / finetune
    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        ck = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ck["model"], strict=True)
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_acc = float(ck.get("best_acc", ck.get("acc", 0.0)))
        if "opt" in ck and "sched" in ck and not args.reset_optimizer:
            try:
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                # align to continue smoothly
                if hasattr(sched, "last_epoch"):
                    sched.last_epoch = start_epoch - 1
                print(f"Loaded optimizer/scheduler; continuing at epoch {start_epoch}")
            except Exception as e:
                print("Could not load optimizer/scheduler state:", e)
                # advance schedule to match start_epoch
                for _ in range(start_epoch - 1):
                    sched.step()
        else:
            for _ in range(start_epoch - 1):
                sched.step()
            print(f"Resumed weights only; restarted optimizer. start_epoch={start_epoch}")
    elif args.finetune_from:
        print(f"Finetuning from {args.finetune_from}")
        ck = torch.load(args.finetune_from, map_location=device)
        model.load_state_dict(ck["model"], strict=True)
        # fresh opt/sched already built

    # I/O
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, args.log_name or time.strftime("attn_bin_%Y%m%d_%H%M%S.csv"))
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_acc","train_loss","val_acc","val_loss","lr"])

    print(f"Training: total epochs={args.epochs} (start at {start_epoch}) | "
          f"warmup_epochs={args.warmup_epochs} start_factor={args.warmup_start_factor} | "
          f"sched={args.sched} gamma={args.gamma}")
    print("Checkpoints ->", args.ckpt_dir, "  Log ->", log_path)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0; correct = 0; n = 0

        for x,y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if args.amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()

            loss_sum += loss.item() * x.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            n        += x.size(0)

        train_acc  = correct / max(1, n)
        train_loss = loss_sum / max(1, n)

        # val
        model.eval()
        vloss = 0.0; vcor = 0; vn = 0
        with torch.no_grad():
            for x,y in dl_va:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if args.amp:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                        loss = crit(logits, y)
                else:
                    logits = model(x); loss = crit(logits, y)
                vloss += loss.item() * x.size(0)
                vcor  += (logits.argmax(1) == y).sum().item()
                vn    += x.size(0)

        val_acc  = vcor / max(1, vn)
        val_loss = vloss / max(1, vn)
        last_val_acc = val_acc

        # step scheduler after each epoch
        sched.step()
        lr_now = sched.get_last_lr()[0]

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_acc:.6f}", f"{train_loss:.6f}",
                                    f"{val_acc:.6f}", f"{val_loss:.6f}", f"{lr_now:.8f}"])
        print(f"Epoch {epoch:03d} | train {train_acc:.3f}/{train_loss:.4f} | "
              f"val {val_acc:.3f}/{val_loss:.4f} | lr {lr_now:.6f}")

        # save
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt(os.path.join(args.ckpt_dir, "attn_bin_best.pt"),
                      model, epoch, best_acc, opt, sched, best_acc=best_acc)
        if (epoch % args.save_interval) == 0:
            save_ckpt(os.path.join(args.ckpt_dir, f"attn_bin_e{epoch}.pt"),
                      model, epoch, val_acc, opt, sched, best_acc=best_acc)

    print(f"Done. Best val acc = {best_acc:.4f}. Last val acc = {last_val_acc:.4f}.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",  default="/home/nbalaj4/data/code_ws/deepslide/train_folder_bin")
    ap.add_argument("--ckpt_dir",   default="checkpoints_attn_bin")
    ap.add_argument("--log_dir",    default="logs")
    ap.add_argument("--log_name",   default="")
    ap.add_argument("--epochs",     type=int, default=100, help="TOTAL epochs (not additional)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr",         type=float, default=1e-4)     # smaller LR by default
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers",type=int, default=8)
    ap.add_argument("--save_interval", type=int, default=5)
    ap.add_argument("--optimizer",  choices=["adam","adamw"], default="adam")
    ap.add_argument("--sched",      choices=["exp","cosine"], default="exp")
    ap.add_argument("--gamma",      type=float, default=0.95, help="exp decay factor")
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--warmup_start_factor", type=float, default=0.1)
    ap.add_argument("--resume_from",   type=str, default="")
    ap.add_argument("--finetune_from", type=str, default="")
    ap.add_argument("--reset_optimizer", action="store_true")
    ap.add_argument("--no_imagenet", action="store_true")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    if args.resume_from and args.finetune_from:
        raise SystemExit("Choose only one of --resume_from OR --finetune_from.")
    main(args)
