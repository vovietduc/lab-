"""Hai kiến trúc học sâu cho FashionMNIST: MLP và CNN."""
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """MLP: làm phẳng ảnh 28x28 thành vector 784 chiều."""
    def __init__(self, hidden_sizes=(256, 128), num_classes=10, dropout=0.2):
        super().__init__()
        layers = [nn.Flatten()]
        in_features = 784
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout)])
            in_features = hidden
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CNNClassifier(nn.Module):
    """CNN giữ cấu trúc ảnh 2D và học đặc trưng bằng convolution."""
    def __init__(self, num_classes=10, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 3 * 3, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    name = model_cfg["name"].lower()
    if name == "mlp":
        return MLPClassifier(
            hidden_sizes=tuple(model_cfg.get("hidden_sizes", [256, 128])),
            num_classes=model_cfg.get("num_classes", 10),
            dropout=model_cfg.get("dropout", 0.2),
        )
    if name == "cnn":
        return CNNClassifier(
            num_classes=model_cfg.get("num_classes", 10),
            dropout=model_cfg.get("dropout", 0.25),
        )
    raise ValueError(f"Model không hỗ trợ: {name}. Chọn mlp hoặc cnn.")


if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28)
    for name, net in [("MLP", MLPClassifier()), ("CNN", CNNClassifier())]:
        print(name, tuple(x.shape), "->", tuple(net(x).shape))
