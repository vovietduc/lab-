"""Data Pipeline: tải FashionMNIST -> chia train/val/test -> chuẩn hóa -> DataLoader.

Chuẩn hóa dùng mean/std CHỈ của tập train (tránh data leakage).
Ảnh được giữ ở 28x28; việc phóng to / nhân kênh do model xử lý.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_raw(root: str = "./data"):
    """Trả về (x_train uint8 [N,28,28], y_train, x_test, y_test). Tự tải nếu chưa có."""
    train = datasets.FashionMNIST(root=root, train=True, download=True)
    test = datasets.FashionMNIST(root=root, train=False, download=True)
    return train.data, train.targets.long(), test.data, test.targets.long()


def split_indices(n: int, val_size: int, seed: int = 42):
    """Chia ngẫu nhiên (cố định theo seed) thành chỉ số train và val, không giao nhau."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return perm[val_size:], perm[:val_size]


def compute_stats(x_uint8: torch.Tensor, idx: torch.Tensor):
    x = x_uint8[idx].float() / 255.0
    return x.mean().item(), x.std().item()


def normalize(x_uint8: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """uint8 [N,28,28] -> float32 [N,1,28,28], z-score."""
    return ((x_uint8.float() / 255.0 - mean) / std).unsqueeze(1)


def get_dataloaders(root="./data", val_size=10000, batch_size=128, num_workers=0,
                    seed=42, dry_run=False, pin_memory=False):
    """Trả về (loaders, info). loaders: dict train/val/test."""
    xtr, ytr, xte, yte = load_raw(root)
    tr_idx, va_idx = split_indices(len(xtr), val_size, seed)
    mean, std = compute_stats(xtr, tr_idx)

    x_train, y_train = normalize(xtr[tr_idx], mean, std), ytr[tr_idx]
    x_val, y_val = normalize(xtr[va_idx], mean, std), ytr[va_idx]
    x_test, y_test = normalize(xte, mean, std), yte

    if dry_run:  # Dry Run: tập mẫu rất nhỏ để kiểm tra pipeline
        x_train, y_train = x_train[:50], y_train[:50]
        x_val, y_val = x_val[:20], y_val[:20]
        x_test, y_test = x_test[:20], y_test[:20]

    kw = dict(num_workers=num_workers, pin_memory=pin_memory)
    loaders = {
        "train": DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, **kw),
        "val": DataLoader(TensorDataset(x_val, y_val), batch_size=512, shuffle=False, **kw),
        "test": DataLoader(TensorDataset(x_test, y_test), batch_size=512, shuffle=False, **kw),
    }
    info = {"mean": mean, "std": std, "class_names": CLASS_NAMES,
            "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test)}
    return loaders, info


if __name__ == "__main__":  # Smoke Run cho dữ liệu
    loaders, info = get_dataloaders(dry_run=True)
    xb, yb = next(iter(loaders["train"]))
    print(f"Smoke Run Success! batch ảnh {tuple(xb.shape)}, nhãn {tuple(yb.shape)}, "
          f"mean={info['mean']:.4f}, std={info['std']:.4f}")
