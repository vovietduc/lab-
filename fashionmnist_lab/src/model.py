"""Mô hình: Backbone -> Feature Vector -> Header (MLP).

- FashionMNISTClassifier: Backbone ResNet-18 + MLP Task Head.
- MLPBaseline: MLP thuần để làm baseline so sánh.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FashionMNISTClassifier(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True, hidden_size=256,
                 dropout=0.3, resize_to=64, pretrained=True):
        super().__init__()
        # 1. BACKBONE: ResNet-18 (bỏ lớp FC cuối)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        in_features = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. TASK HEAD: MLP
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.resize_to = resize_to

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:      # backbone đóng băng: giữ BatchNorm ở chế độ eval
            self.backbone.eval()
        return self

    def forward(self, x):
        if self.resize_to and x.shape[-1] != self.resize_to:
            x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:           # ảnh xám 1 kênh -> 3 kênh
            x = x.repeat(1, 3, 1, 1)
        return self.head(self.backbone(x))


class MLPBaseline(nn.Module):
    def __init__(self, in_dim=784, hidden_sizes=(256, 128), num_classes=10, dropout=0.2):
        super().__init__()
        layers = [nn.Flatten()]
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(cfg: dict, pretrained=None) -> nn.Module:
    """Tạo model từ cấu hình YAML. pretrained=None: lấy theo cfg."""
    m = cfg["model"]
    if m["name"] == "resnet18":
        use_pre = m.get("pretrained", True) if pretrained is None else pretrained
        return FashionMNISTClassifier(
            num_classes=m.get("num_classes", 10), freeze_backbone=m.get("freeze_backbone", True),
            hidden_size=m.get("hidden_size", 256), dropout=m.get("dropout", 0.3),
            resize_to=m.get("resize_to", 64), pretrained=use_pre)
    if m["name"] == "mlp":
        return MLPBaseline(hidden_sizes=tuple(m.get("hidden_sizes", [256, 128])),
                           num_classes=m.get("num_classes", 10), dropout=m.get("dropout", 0.2))
    raise ValueError(f"Model không hỗ trợ: {m['name']}")


if __name__ == "__main__":  # Smoke Run: truyền Tensor giả qua model
    dummy_input = torch.randn(4, 1, 28, 28)  # lô 4 ảnh 28x28, 1 kênh
    for name, mdl in [("ResNet18 + MLP head", FashionMNISTClassifier(pretrained=False)),
                      ("MLP baseline", MLPBaseline())]:
        out = mdl(dummy_input)
        print(f"Smoke Run Success! [{name}] Input {tuple(dummy_input.shape)} -> Output {tuple(out.shape)}")
