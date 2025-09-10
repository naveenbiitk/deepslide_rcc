#!/usr/bin/env python3
import os, shutil
from pathlib import Path
import random

SRC = Path("/home/nbalaj4/data/code_ws/deepslide")
SRC_TRAIN = SRC/"train_folder_cap20k"/"train"
SRC_VAL   = SRC/"train_folder"/"val"    # keep your original val patches

DST = SRC/"train_folder_bin"
DST_TRAIN = DST/"train"
DST_VAL   = DST/"val"

M_BENIGN = {"Benign","Oncocytoma"}
M_MALIGN = {"Clearcell","Papillary","Chromophobe"}

def link_all(srcs, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for s in srcs:
        for p in s.glob("*.jpg"):
            t = dst/p.name
            if t.exists(): continue
            try:
                os.link(p, t)  # hardlink
            except OSError:
                shutil.copy2(p, t)

def build_split(src_root, dst_root):
    (dst_root/"Benign").mkdir(parents=True, exist_ok=True)
    (dst_root/"Malignant").mkdir(parents=True, exist_ok=True)
    # Benign
    link_all([src_root/c for c in M_BENIGN if (src_root/c).exists()], dst_root/"Benign")
    # Malignant
    link_all([src_root/c for c in M_MALIGN if (src_root/c).exists()], dst_root/"Malignant")

if __name__ == "__main__":
    build_split(SRC_TRAIN, DST_TRAIN)
    build_split(SRC_VAL,   DST_VAL)
    print("✅ Binary patch root:", DST)
