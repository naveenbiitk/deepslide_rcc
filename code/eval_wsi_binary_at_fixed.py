#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
from pathlib import Path
import re
from typing import Tuple, Optional
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

# ----- helpers -----
# BIN_MAP = {"Benign":0, "Oncocytoma":0, "Clearcell":1, "Papillary":1, "Chromophobe":1}
# binary mapping (supports 5-class layout and 2-class binary layout)
BIN_MAP = {
    "benign": 0, "oncocytoma": 0,
    "clearcell": 1, "chromophobe": 1, "papillary": 1,
    "malignant": 1,  # support binary folder layout
}
ALIASES = {
    "clear_cell": "clearcell",
    "clear-cell": "clearcell",
    "ccrcc": "clearcell",   # common shorthand
}

def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower().strip() if ch.isalnum())

def find_prob_column(df: pd.DataFrame, user_prob_col: str = None) -> str:
    actual = {_norm(c): c for c in df.columns}
    if user_prob_col:
        k = _norm(user_prob_col)
        if k in actual: return actual[k]
        raise ValueError(f"--prob_col='{user_prob_col}' not found. Columns: {list(df.columns)}")
    aliases = ["mean_prob_malignant","prob_malignant","pmalignant",
               "pmal","slideprob","meanprob","prob","probability","malignantprob"]
    for name in aliases:
        if _norm(name) in actual: return actual[_norm(name)]
    # last resort: float col in [0,1]
    for c in df.columns[::-1]:
        try:
            s = df[c].astype(float)
            if ((s >= 0) & (s <= 1)).mean() > 0.95: return c
        except Exception:
            pass
    raise ValueError("Could not detect probability column.")



def infer_label_from_path(wsi_root: str, p: str, meta_map=None) -> int:
    root = Path(wsi_root)          # DO NOT resolve symlinks
    pp   = Path(p)                 # keep symlink path

    # Try relative to wsi_root first (works for split symlink layout)
    try:
        rel_parts = list(pp.relative_to(root).parts)
    except Exception:
        rel_parts = list(pp.parts)  # fallback to absolute parts

    # 1) check normalized path segments
    tokens = []
    for seg in rel_parts:
        s = seg.strip().lower()
        s = ALIASES.get(s, s)
        tokens.append(s)
        if s in BIN_MAP:
            return BIN_MAP[s]

    # 2) fallback: regex word-boundary search on the relative path string
    rel_str = "/".join(rel_parts).lower()
    for k in BIN_MAP.keys():
        k2 = re.escape(k)
        if re.search(rf"(?<!\w){k2}(?!\w)", rel_str):
            return BIN_MAP[k]

    raise ValueError(f"Cannot infer class from path: {p} (scanned tokens={tokens})")

def load_split(csv_path: str, wsi_root: str = None, prob_col: str = None,
               time_col: str = "inference_ms") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    pcol = find_prob_column(df, prob_col)
    if "mean_prob_malignant" not in df.columns:
        df["mean_prob_malignant"] = df[pcol].astype(float)

    # labels: prefer explicit column if present
    y = None
    for cand in ["label","y","target"]:
        if cand in df.columns:
            y = df[cand].astype(int).to_numpy(); break
    if y is None:
        if not wsi_root:
            raise ValueError(f"{csv_path}: need --wsi_root to infer labels from folder structure.")
        y = np.array([infer_label_from_path(wsi_root, p) for p in df["path"].tolist()], dtype=int)

    p = df["mean_prob_malignant"].astype(float).to_numpy()
    # times (optional)
    t = df[time_col].astype(float).to_numpy() if time_col in df.columns else None
    return y, p, t, df

def time_summary(ms: Optional[np.ndarray]) -> dict:
    if ms is None or len(ms)==0: return {}
    ms = ms.astype(float)
    return dict(
        ms_mean=float(ms.mean()),
        ms_p50=float(np.percentile(ms,50)),
        ms_p90=float(np.percentile(ms,90)),
        ms_min=float(ms.min()),
        ms_max=float(ms.max()),
    )

def compute_at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    acc  = accuracy_score(y, yhat)
    prec = precision_score(y, yhat, zero_division=0)
    rec  = recall_score(y, yhat, zero_division=0)               # sensitivity
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0                     # specificity
    f1   = f1_score(y, yhat, zero_division=0)
    auc  = roc_auc_score(y, p) if len(np.unique(y))==2 else float("nan")
    return dict(
        threshold=float(t),
        acc=float(acc), precision=float(prec),
        recall=float(rec), specificity=float(spec),
        f1=float(f1), auc=float(auc),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        N=int(len(y)), N_pos=int((y==1).sum()), N_neg=int((y==0).sum()),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", help="inference_val_bin.csv")
    ap.add_argument("--test_csv", help="inference_test_bin.csv")
    ap.add_argument("--wsi_root", required=True, help="root of wsi_val/wsi_test folders")
    ap.add_argument("--prob_col", default=None, help="override probability column name")
    ap.add_argument("--time_col", default="inference_ms", help="per-slide inference time column")
    ap.add_argument("--fixed_thresh", type=float, default=0.65)
    ap.add_argument("--out_csv", default="results_binary_wsi/Table2_WSI_Binary_Performance.csv")
    args = ap.parse_args()

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)
    rows = []

    def run_split(name, path):
        y, p, t, df = load_split(path, args.wsi_root, args.prob_col, args.time_col)
        met = compute_at_threshold(y, p, args.fixed_thresh)
        ts  = time_summary(t)
        print(f"[{name}] τ={args.fixed_thresh:.3f} "
              f"acc={met['acc']:.3f} prec={met['precision']:.3f} "
              f"sens={met['recall']:.3f} spec={met['specificity']:.3f} "
              f"f1={met['f1']:.3f} auc={met['auc']:.3f} "
              f"CM=[TN={met['TN']}, FP={met['FP']}, FN={met['FN']}, TP={met['TP']}] N={met['N']}")
        if ts:
            print(f"[{name}] latency per WSI (ms): mean {ts['ms_mean']:.0f} | "
                  f"p50 {ts['ms_p50']:.0f} | p90 {ts['ms_p90']:.0f} | "
                  f"min {ts['ms_min']:.0f} | max {ts['ms_max']:.0f}")
        row = {"split": name, **met, **ts}
        rows.append(row)

    if args.val_csv:  run_split("VAL",  args.val_csv)
    if args.test_csv: run_split("TEST", args.test_csv)

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[done] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
