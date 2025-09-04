#!/usr/bin/env python3
import argparse, os, glob, numpy as np, pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

MAL = {"Chromophobe","Clearcell","Papillary"}
BEN = {"Benign","Oncocytoma"}

def build_slide2class(wsi_root: str) -> Dict[str,str]:
    s2c = {}
    for split in ["wsi_train","wsi_val","wsi_test"]:
        d = os.path.join(wsi_root, split)
        if not os.path.isdir(d): continue
        for cls in os.listdir(d):
            cdir = os.path.join(d, cls)
            if not os.path.isdir(cdir): continue
            for p in glob.glob(os.path.join(cdir, "*.png")):
                sid = os.path.splitext(os.path.basename(p))[0]
                s2c[sid] = cls
    return s2c

def map_binary_from_any(df: pd.DataFrame, wsi_root: Optional[str]) -> pd.DataFrame:
    # Prefer an existing binary column; else map from orig_class/class; else infer from slide_id and filesystem
    label_col = None
    for c in ["binary_class", "orig_class", "class"]:
        if c in df.columns:
            label_col = c; break

    if label_col is not None:
        # guard against slide IDs mistakenly living in 'class'
        vals = set(map(str, df[label_col].dropna().unique().tolist()))
        if all(v.startswith("DHMC_") for v in vals):
            label_col = None

    if label_col is None:
        if "slide_id" not in df.columns:
            raise ValueError("CSV needs either binary_class/orig_class/class or slide_id to infer labels.")
        if not wsi_root:
            raise ValueError("Missing --wsi_root to infer labels from folder structure.")
        s2c = build_slide2class(wsi_root)
        df["orig_class"] = df["slide_id"].astype(str).map(s2c)
        if df["orig_class"].isna().any():
            missing = df[df["orig_class"].isna()]["slide_id"].unique()
            raise ValueError(f"Could not map slide_ids from filesystem: {missing[:10]}")
        label_col = "orig_class"

    if label_col != "binary_class":
        def bin_map(s: str) -> str:
            if s in MAL: return "Malignant"
            if s in BEN: return "Benign"
            raise ValueError(f"Unknown class: {s}")
        df["binary_class"] = df[label_col].astype(str).apply(bin_map)
    return df

def load_split(csv_path: str, wsi_root: Optional[str]) -> Tuple[np.ndarray,np.ndarray,List[str],pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "mean_prob_malignant" not in df.columns:
        raise ValueError(f"{csv_path} missing mean_prob_malignant; need inference CSV with probabilities.")
    df = map_binary_from_any(df, wsi_root)
    y_true = (df["binary_class"] == "Malignant").astype(int).to_numpy()
    y_score = df["mean_prob_malignant"].astype(float).to_numpy()
    slide_ids = df["slide_id"].astype(str).tolist() if "slide_id" in df.columns else [f"s{i}" for i in range(len(df))]
    return y_true, y_score, slide_ids, df

def pick_threshold(y: np.ndarray, p: np.ndarray, strategy: str, fixed_thresh: float) -> float:
    if strategy == "fixed":
        return float(fixed_thresh)
    best_t, best_val = 0.5, -1.0
    # sweep unique scores + edges for stability
    candidates = np.unique(np.concatenate([p, np.array([0.0, 1.0])]))
    if len(candidates) > 4000:  # cap cost
        candidates = np.linspace(0,1,2001)
    for t in candidates:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
        sens = tp / (tp+fn) if (tp+fn)>0 else 0.0      # recall for malignant
        spec = tn / (tn+fp) if (tn+fp)>0 else 0.0      # specificity for benign
        if strategy == "f1":
            val = f1_score(y, yhat, zero_division=0)
        elif strategy == "youden":
            val = sens + spec - 1.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        if val > best_val:
            best_val, best_t = val, float(t)
    return best_t

def summarize_times(df: pd.DataFrame, mode: str, time_col: str, ms_per_patch: float) -> Dict[str, float]:
    times = None
    if mode == "csv":
        if time_col not in df.columns:
            raise ValueError(f"time_col='{time_col}' not in CSV; available: {list(df.columns)}")
        times = df[time_col].astype(float).to_numpy()
    elif mode == "estimate":
        if "n_patches" not in df.columns:
            raise ValueError("estimate mode needs 'n_patches' column in CSV.")
        times = df["n_patches"].astype(float).to_numpy() * float(ms_per_patch)
    elif mode == "none":
        return {}
    else:
        raise ValueError(f"Unknown time mode: {mode}")

    if times is None or len(times)==0:
        return {}
    return {
        "inf_ms_mean": float(np.mean(times)),
        "inf_ms_p50":  float(np.percentile(times,50)),
        "inf_ms_p90":  float(np.percentile(times,90)),
        "inf_ms_min":  float(np.min(times)),
        "inf_ms_max":  float(np.max(times)),
    }

def compute_at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> Dict[str, float]:
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    acc = accuracy_score(y, yhat)
    prec = precision_score(y, yhat, zero_division=0)
    rec = recall_score(y, yhat, zero_division=0)          # Sensitivity (malignant)
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0               # Specificity (benign)
    f1 = f1_score(y, yhat, zero_division=0)
    auc = roc_auc_score(y, p) if len(np.unique(y))==2 else float("nan")
    return {
        "threshold": float(t),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),          # sensitivity
        "specificity": float(spec),
        "f1": float(f1),
        "auc": float(auc),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "N": int(len(y)), "N_pos": int((y==1).sum()), "N_neg": int((y==0).sum()),
    }

def save_table_row(path: str, split_name: str, row: Dict[str, float], time_stats: Dict[str,float]):
    out = {"split": split_name, **row, **time_stats}
    df = pd.DataFrame([out])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--wsi_root", required=False, default=None,
                    help="Needed if CSV lacks class labels; used to infer from folder structure.")
    ap.add_argument("--strategy", choices=["f1","youden","fixed"], default="f1")
    ap.add_argument("--fixed_thresh", type=float, default=0.50)
    ap.add_argument("--time_mode", choices=["none","csv","estimate"], default="none",
                    help="'csv' uses a time column; 'estimate' uses n_patches*ms_per_patch.")
    ap.add_argument("--time_col", default="inference_ms")
    ap.add_argument("--ms_per_patch", type=float, default=3.0, help="for estimate mode")
    ap.add_argument("--out_dir", default="results_binary_wsi")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "Table2_WSI_Binary_Performance_Extras.csv")

    # VAL
    yv, pv, sids_v, dfv = load_split(args.val_csv, args.wsi_root)
    t_star = pick_threshold(yv, pv, args.strategy, args.fixed_thresh)
    row_v = compute_at_threshold(yv, pv, t_star)
    time_v = summarize_times(dfv, args.time_mode, args.time_col, args.ms_per_patch)
    print(f"[VAL] threshold ({args.strategy}) = {t_star:.3f}")
    print(f"[VAL] acc={row_v['acc']:.3f} prec={row_v['precision']:.3f} "
          f"rec={row_v['recall']:.3f} spec={row_v['specificity']:.3f} "
          f"f1={row_v['f1']:.3f} auc={row_v['auc']:.3f}")
    if time_v:
        print(f"[VAL] inference ms (mean/p50/p90/min/max)= "
              f"{time_v['inf_ms_mean']:.0f}/{time_v['inf_ms_p50']:.0f}/"
              f"{time_v['inf_ms_p90']:.0f}/{time_v['inf_ms_min']:.0f}/{time_v['inf_ms_max']:.0f}")
    save_table_row(out_csv, "VAL", row_v, time_v)

    # TEST
    yt, pt, sids_t, dft = load_split(args.test_csv, args.wsi_root)
    row_t = compute_at_threshold(yt, pt, t_star)
    time_t = summarize_times(dft, args.time_mode, args.time_col, args.ms_per_patch)
    print(f"[TEST] acc={row_t['acc']:.3f} prec={row_t['precision']:.3f} "
          f"rec={row_t['recall']:.3f} spec={row_t['specificity']:.3f} "
          f"f1={row_t['f1']:.3f} auc={row_t['auc']:.3f}")
    print(f"[TEST] CM = [TN={row_t['TN']}, FP={row_t['FP']}, FN={row_t['FN']}, TP={row_t['TP']}]  "
          f"N={row_t['N']} (pos={row_t['N_pos']}, neg={row_t['N_neg']})")
    if time_t:
        print(f"[TEST] inference ms (mean/p50/p90/min/max)= "
              f"{time_t['inf_ms_mean']:.0f}/{time_t['inf_ms_p50']:.0f}/"
              f"{time_t['inf_ms_p90']:.0f}/{time_t['inf_ms_min']:.0f}/{time_t['inf_ms_max']:.0f}")
    save_table_row(out_csv, "TEST", row_t, time_t)

    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
