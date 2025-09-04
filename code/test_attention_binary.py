#!/usr/bin/env python3
import os, glob, csv, torch
from PIL import Image
from torchvision import transforms
from models.attn_resnet18 import AttnResNet18

CKPT = "checkpoints_attn_bin/attn_bin_best.pt"
VAL_ROOT = "patches_eval_val"
TEST_ROOT = "patches_eval_test"
OUT_VAL = "inference_val_bin.csv"
OUT_TEST = "inference_test_bin.csv"

norm = transforms.Normalize(mean=[0.7488,0.6045,0.7521], std=[0.1571,0.1921,0.1504])
tfm = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), norm])

def slide_folders(root):
    # expects structure: root/<Class>/<SlideID>/*.jpg
    return sorted(glob.glob(os.path.join(root,"*","*")))

def infer_split(root, out_csv, device):
    model = AttnResNet18(num_classes=2, pretrained=False).to(device)
    sd = torch.load(CKPT, map_location=device)["model"]; model.load_state_dict(sd); model.eval()
    rows = [["split_root","class","slide_id","n_patches","mean_prob_malignant"]]
    with torch.no_grad():
        for slide_dir in slide_folders(root):
            cls = os.path.basename(os.path.dirname(slide_dir))
            sid = os.path.basename(slide_dir)
            probs = []
            for jp in glob.glob(os.path.join(slide_dir,"*.jpg")):
                im = Image.open(jp).convert("RGB")
                x = tfm(im).unsqueeze(0).to(device)
                logits = model(x)
                p_mal = torch.softmax(logits, dim=1)[0,1].item()
                probs.append(p_mal)
            rows.append([root, cls, sid, len(probs), sum(probs)/max(1,len(probs))])
    with open(out_csv,"w",newline="") as f:
        csv.writer(f).writerows(rows)
    print("Wrote", out_csv)

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if os.path.isdir(VAL_ROOT):  infer_split(VAL_ROOT, OUT_VAL, device)
    if os.path.isdir(TEST_ROOT): infer_split(TEST_ROOT, OUT_TEST, device)
