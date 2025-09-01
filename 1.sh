python - <<'PY'
import glob
root="/home/nbalaj4/data/DHMC_deepslide/wsi_train"
print("train pngs (two-level):", len(glob.glob(root+"/*/*.png")))
print("sample:", glob.glob(root+"/*/*.png")[:5])
PY

