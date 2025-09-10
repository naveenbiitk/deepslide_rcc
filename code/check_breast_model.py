
import torch, timm, torch.nn as nn
from pathlib import Path

ckpt_path = "/home/nbalaj4/data/code_ws/deepslide/checkpoints_pvt_bin/breast_pvt.pth"

class PVTBinary(nn.Module):
    def __init__(self, name="pvt_v2_b1"):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 1)
    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)

def _normalize(sd):
    sd = {k.replace("module.","",1): v for k,v in sd.items()}
    out={}
    for k,v in sd.items():
        if k.startswith("0."): out["backbone."+k[2:]] = v
        elif k.startswith("1."): out["head."+k[2:]] = v
        else: out[k]=v
    return out

ck = torch.load(ckpt_path, map_location="cpu")
sd = ck.get("model_state") or ck.get("state_dict") or ck
sd = _normalize(sd)

m = PVTBinary("pvt_v2_b1")
missing, unexpected = m.load_state_dict(sd, strict=False)
print("Loaded with strict=False")
print("missing:", missing)
print("unexpected:", unexpected)
print("head weight shape:", m.head.weight.shape)

