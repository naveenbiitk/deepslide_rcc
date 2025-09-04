import torch, torch.nn as nn
import timm

class PVTBinary(nn.Module):
    """
    PVTv2-B1 backbone (timm) + 1-logit head for binary (malignant vs benign).
    """
    def __init__(self, name: str = "pvt_v2_b1", pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        logit = self.head(feats)  # shape [B,1]
        return logit
