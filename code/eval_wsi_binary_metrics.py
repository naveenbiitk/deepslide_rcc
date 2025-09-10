#!/usr/bin/env python3
import argparse, os, csv, math, numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from typing import Optional

import os, glob, pandas as pd, numpy as np

MAL = {"Chromophobe","Clearcell","Papillary"}
BEN = {"Benign","Oncocytoma"}

def build_slide2class(wsi_root):
    s2c = {}
    for split in ["wsi_train","wsi_val","wsi_test"]:
        split_dir = os.path.join(wsi_root, split)
        if not os.path.isdir(split_dir): continue
        for cls in os.listdir(split_dir):
            cdir = os.path.join(split_dir, cls)
            if not os.path.isdir(cdir): continue
            for p in glob.glob(os.path.join(cdir, "*.png")):
                sid = os.path.splitext(os.path.basename(p))[0]  # "DHMC_0001"
                s2c[sid] = cls
    return s2c

# --- tolerant column handling ---
def _norm(s: str) -> str:
    # normalize header names: lowercase, remove spaces/underscores
    return "".join(ch for ch in s.lower().strip() if ch.isalnum())

def find_prob_column(df, user_prob_col: str = None) -> str:
    # if user specified and exists (case/space-insensitive), honor it
    actual_by_norm = {_norm(c): c for c in df.columns}
    if user_prob_col:
        key = _norm(user_prob_col)
        if key in actual_by_norm:
            return actual_by_norm[key]
        raise ValueError(f"--prob_col='{user_prob_col}' not found. Available: {list(df.columns)}")

    # try common aliases
    candidates = [
        "mean_prob_malignant", "prob_malignant", "p_malignant", "p_mal",
        "slide_prob", "mean_prob", "prob", "probability", "malignant_prob"
    ]
    for name in candidates:
        key = _norm(name)
        if key in actual_by_norm:
            return actual_by_norm[key]

    # last resort: any single float column at the end that looks like prob in [0,1]
    for c in reversed(df.columns):
        try:
            s = df[c].astype(float)
            if ((s >= 0) & (s <= 1)).mean() > 0.95:
                return c
        except Exception:
            continue
    raise ValueError(f"Could not find probability column. Available columns: {list(df.columns)}")

# def load_split(csv_path):
#     df = pd.read_csv(csv_path)
#     # expected columns from test_attention_binary.py:
#     # ['split_root','class','slide_id','n_patches','mean_prob_malignant']
#     y_true = (df['class'].astype(str) == 'Malignant').astype(int).to_numpy()
#     y_score = df['mean_prob_malignant'].astype(float).to_numpy()
#     slide_ids = df['slide_id'].astype(str).to_list()
#     return y_true, y_score, slide_ids, df
# def load_split(csv_path, wsi_root=None):
def load_split(csv_path: str, wsi_root: Optional[str], prob_col: Optional[str] = None):
    df = pd.read_csv(csv_path)
    # trim header whitespace
    df.columns = [c.strip() for c in df.columns]
    # find/standardize probability column
    pcol = find_prob_column(df, prob_col)
    if "mean_prob_malignant" not in df.columns:
        df["mean_prob_malignant"] = df[pcol].astype(float)

    # pick a label column if present
    label_col = None
    for col in ["binary_class", "orig_class", "class"]:
        if col in df.columns:
            label_col = col; break

    # If label col found but values look like slide IDs, treat as missing
    if label_col is not None:
        vals = set(map(str, df[label_col].dropna().unique().tolist()))
        if all(v.startswith("DHMC_") for v in vals):
            label_col = None  # clearly slide IDs, not classes

    # If missing, try to recover from slide_id + filesystem
    if label_col is None:
        if "slide_id" not in df.columns:
            raise ValueError(f"{csv_path}: no class/binary_class/orig_class AND no slide_id. Can't infer.")
        if not wsi_root:
            raise ValueError(f"{csv_path}: need --wsi_root to infer labels from folder structure.")
        s2c = build_slide2class(wsi_root)
        df["orig_class"] = df["slide_id"].astype(str).map(s2c)
        if df["orig_class"].isna().any():
            missing = df[df["orig_class"].isna()]["slide_id"].unique()
            raise ValueError(f"Could not find these slide_ids under {wsi_root}: {missing[:10]}")
        label_col = "orig_class"

    # Map to binary_class if needed
    if label_col != "binary_class":
        def map_bin(s):
            s = str(s)
            if s in MAL: return "Malignant"
            if s in BEN: return "Benign"
            raise ValueError(f"Unknown class value '{s}' in {label_col}")
        df["binary_class"] = df[label_col].apply(map_bin)
        label_col = "binary_class"

    if "mean_prob_malignant" not in df.columns:
        raise ValueError(f"{csv_path} missing mean_prob_malignant; need inference CSV with probs.")

    y_true  = (df[label_col] == "Malignant").astype(int).to_numpy()
    y_score = df["mean_prob_malignant"].astype(float).to_numpy()
    slide_ids = df["slide_id"].astype(str).to_list() if "slide_id" in df.columns else list(range(len(df)))
    n_pos = int((y_true==1).sum()); n_neg = int((y_true==0).sum())
    print(f"[{os.path.basename(csv_path)}] N={len(y_true)}  Malignant={n_pos}  Benign={n_neg}")
    return y_true, y_score, slide_ids, df



def pick_threshold_from_val(y_true, y_score, strategy="f1", fixed=None):
    if strategy == "fixed":
        assert fixed is not None, "Provide --fixed_thresh for strategy=fixed"
        return float(fixed)

    # sweep thresholds on [0.05..0.95]
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_stat = 0.5, -1
    for t in ts:
        y_pred = (y_score >= t).astype(int)
        if strategy == "f1":
            stat = f1_score(y_true, y_pred, zero_division=0)
        elif strategy == "youden":
            # sensitivity + specificity - 1
            tp = np.sum((y_true==1)&(y_pred==1))
            fn = np.sum((y_true==1)&(y_pred==0))
            tn = np.sum((y_true==0)&(y_pred==0))
            fp = np.sum((y_true==0)&(y_pred==1))
            sens = tp / max(1, tp+fn)
            spec = tn / max(1, tn+fp)
            stat = sens + spec - 1.0
        else:
            raise ValueError("strategy must be f1|youden|fixed")
        if stat > best_stat:
            best_stat, best_t = stat, float(t)
    return best_t

def compute_metrics(y_true, y_score, thresh):
    y_pred = (y_score >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # guard AUC for single-class edge cases
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])  # [[TN,FP],[FN,TP]]
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc, cm=cm)

def bootstrap_ci(y_true, y_score, thresh, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, precs, recs, f1s, aucs = [], [], [], [], []
    idx = np.arange(n)
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        yt, ys = y_true[b], y_score[b]
        ypred = (ys >= thresh).astype(int)
        accs.append(accuracy_score(yt, ypred))
        precs.append(precision_score(yt, ypred, zero_division=0))
        recs.append(recall_score(yt, ypred, zero_division=0))
        f1s.append(f1_score(yt, ypred, zero_division=0))
        try:
            aucs.append(roc_auc_score(yt, ys))
        except Exception:
            pass
    def ci(a): 
        a = np.array(a); 
        lo, hi = np.percentile(a, [2.5, 97.5])
        return float(lo), float(hi), float(np.mean(a))
    return {
        "acc_ci": ci(accs), "prec_ci": ci(precs),
        "rec_ci": ci(recs), "f1_ci": ci(f1s),
        "auc_ci": ci(aucs) if len(aucs)>0 else (float("nan"), float("nan"), float("nan"))
    }

def plot_confusion(cm, out_png, labels=("Benign","Malignant")):
    fig = plt.figure(figsize=(4.5,4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center")
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    fig.tight_layout(); fig.savefig(out_png, dpi=180, bbox_inches='tight'); plt.close(fig)

def plot_roc(y_true, y_score, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")
    fig = plt.figure(figsize=(4.5,4))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0,1], [0,1], linestyle='--')
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_png, dpi=180, bbox_inches='tight'); plt.close(fig)

def write_table2_csv(path, split_name, thresh, metrics, ci, n, pos, neg):
    row = {
        "split": split_name,
        "threshold": thresh,
        "N": n,
        "Positives (Malignant)": pos,
        "Negatives (Benign)": neg,
        "Accuracy": metrics["acc"],
        "Accuracy 95% CI": f"{ci['acc_ci'][0]:.3f}–{ci['acc_ci'][1]:.3f}",
        "Precision": metrics["prec"],
        "Precision 95% CI": f"{ci['prec_ci'][0]:.3f}–{ci['prec_ci'][1]:.3f}",
        "Recall": metrics["rec"],
        "Recall 95% CI": f"{ci['rec_ci'][0]:.3f}–{ci['rec_ci'][1]:.3f}",
        "F1": metrics["f1"],
        "F1 95% CI": f"{ci['f1_ci'][0]:.3f}–{ci['f1_ci'][1]:.3f}",
        "AUC": metrics["auc"],
        "AUC 95% CI": f"{ci['auc_ci'][0]:.3f}–{ci['auc_ci'][1]:.3f}",
    }
    hdr = list(row.keys())
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if write_header: w.writeheader()
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", default="inference_val_bin.csv")
    ap.add_argument("--test_csv", default="inference_test_bin.csv")
    ap.add_argument("--strategy", choices=["f1","youden","fixed"], default="f1")
    ap.add_argument("--fixed_thresh", type=float, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--outdir", default="results_binary_wsi")
    ap.add_argument("--wsi_root", default="/home/nbalaj4/data/DHMC_deepslide",
                help="Folder that contains wsi_train/wsi_val/wsi_test/<CLASS>/<SLIDE>.png")
    ap.add_argument("--prob_col", default=None,
                help="Name of probability column (optional). If omitted, auto-detects common aliases.")


    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # old:
    # yv, sv, _, _ = load_split(args.val_csv)
    # yt, st, _, _ = load_split(args.test_csv)

    # new:
    # yv, sv, _, _ = load_split(args.val_csv, args.wsi_root)
    # yt, st, _, _ = load_split(args.test_csv, args.wsi_root)

    yv, sv, _, dfv = load_split(args.val_csv, args.wsi_root, args.prob_col)
    yt, st, _, dft = load_split(args.test_csv, args.wsi_root, args.prob_col)

    # pick threshold on VAL
    tstar = pick_threshold_from_val(yv, sv, args.strategy, args.fixed_thresh)
    print(f"[VAL] threshold ({args.strategy}) = {tstar:.3f}")

    # compute metrics VAL
    m_val = compute_metrics(yv, sv, tstar)
    ci_val = bootstrap_ci(yv, sv, tstar, n_boot=args.n_boot, seed=args.seed)
    print(f"[VAL] acc={m_val['acc']:.3f} prec={m_val['prec']:.3f} rec={m_val['rec']:.3f} "
          f"f1={m_val['f1']:.3f} auc={m_val['auc']:.3f}")

    # compute metrics TEST
    m_test = compute_metrics(yt, st, tstar)
    ci_test = bootstrap_ci(yt, st, tstar, n_boot=args.n_boot, seed=args.seed)
    print(f"[TEST] acc={m_test['acc']:.3f} prec={m_test['prec']:.3f} rec={m_test['rec']:.3f} "
          f"f1={m_test['f1']:.3f} auc={m_test['auc']:.3f}")

    # counts
    n_val = len(yv); pos_val = int((yv==1).sum()); neg_val = n_val - pos_val
    n_test = len(yt); pos_test = int((yt==1).sum()); neg_test = n_test - pos_test

    # write Table 2 style CSV
    table2 = os.path.join(args.outdir, "Table2_WSI_Binary_Performance.csv")
    write_table2_csv(table2, "Validation", tstar, m_val, ci_val, n_val, pos_val, neg_val)
    write_table2_csv(table2, "Test",       tstar, m_test, ci_test, n_test, pos_test, neg_test)
    print("Wrote", table2)

    # plots
    plot_confusion(m_test["cm"], os.path.join(args.outdir, "confusion_matrix_test.png"))
    plot_roc(yt, st, os.path.join(args.outdir, "roc_curve_test.png"))
    print("Saved confusion_matrix_test.png and roc_curve_test.png")

if __name__ == "__main__":
    main()
