import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models

def he_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Glorot
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

class AttnResNet18(nn.Module):
    """
    Patch-level spatial attention:
      - backbone: ResNet-18 up to conv5 (C=512, H=W=7 for 224x224 input)
      - attention: 64 Conv3d filters of size (D=512, 3, 3) with padding (0,1,1)
      - dropout 0.5 after concatenation
      - classifier: FC(64) -> 2 logits (Benign/Malignant) with CE loss
    """
    def __init__(self, num_classes=2, pretrained=True, p_dropout=0.5):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        # feature extractor to conv5_x output (512x7x7); strip avgpool & fc
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        # attention conv3d: in=(N,1,512,7,7) -> (N,64,1,7,7)
        self.attn = nn.Conv3d(in_channels=1, out_channels=64,
                              kernel_size=(512,3,3), padding=(0,1,1), bias=True)
        self.bn_attn = nn.BatchNorm3d(64)
        self.drop = nn.Dropout(p_dropout)
        self.classifier = nn.Linear(64, num_classes)

        # init
        self.apply(he_init)

        # set BN affine to unit weight/zero bias explicitly (already done in he_init)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1.0); m.bias.data.zero_()

    def forward(self, x):
        # x: (N,3,224,224)
        f = self.features(x)                  # (N,512,7,7)
        f3 = f.unsqueeze(1)                   # (N,1,512,7,7)
        a = self.attn(f3)                     # (N,64,1,7,7)
        a = self.bn_attn(a)
        a = F.relu(a, inplace=True)
        a = a.squeeze(2)                      # (N,64,7,7)

        # soft attention per filter over spatial grid
        a_flat = a.view(a.size(0), a.size(1), -1)      # (N,64,49)
        w = F.softmax(a_flat, dim=-1)                  # normalize per filter
        # weight the backbone features per filter
        f_flat = f.view(f.size(0), f.size(1), -1)      # (N,512,49)

        # For each filter, we take the weighted sum of spatial locations using the same weights for all channels,
        # then compress channels by average to get a 64-d vector.
        # (simple, robust; alternatives: 1x1 conv after weighted pooling).
        pooled = torch.einsum('nkw,ncw->nkc', w, f_flat)  # (N,64,512)
        pooled = pooled.mean(dim=-1)                      # (N,64)

        z = self.drop(pooled)
        logits = self.classifier(z)                       # (N,2)
        return logits
