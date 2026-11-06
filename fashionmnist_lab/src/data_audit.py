"""Lightweight FashionMNIST data audit.

Run from the fashionmnist_lab directory:
    python -m src.data_audit

Downloads FashionMNIST if it is not already present in ./data.
"""
import torch
from torchvision import datasets

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def run_audit(root="./data", val_size=10_000, seed=42):
    train = datasets.FashionMNIST(root=root, train=True, download=True)
    test = datasets.FashionMNIST(root=root, train=False, download=True)

    x_train, y_train = train.data, train.targets.long()
    x_test, y_test = test.data, test.targets.long()

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(x_train), generator=generator)
    val_idx, train_idx = permutation[:val_size], permutation[val_size:]

    checks = {
        "train_count_60000": len(x_train) == 60_000,
        "test_count_10000": len(x_test) == 10_000,
        "image_shape_28x28": tuple(x_train.shape[1:]) == (28, 28) and tuple(x_test.shape[1:]) == (28, 28),
        "grayscale_uint8": x_train.dtype == torch.uint8 and x_test.dtype == torch.uint8,
        "pixel_range_0_255": int(x_train.min()) >= 0 and int(x_train.max()) <= 255 and int(x_test.min()) >= 0 and int(x_test.max()) <= 255,
        "labels_in_0_9": int(y_train.min()) == 0 and int(y_train.max()) == 9 and int(y_test.min()) == 0 and int(y_test.max()) == 9,
        "train_val_disjoint": len(set(train_idx.tolist()).intersection(set(val_idx.tolist()))) == 0,
        "split_sizes_50000_10000": len(train_idx) == 50_000 and len(val_idx) == 10_000,
    }

    train_counts = torch.bincount(y_train[train_idx], minlength=10)
    val_counts = torch.bincount(y_train[val_idx], minlength=10)
    test_counts = torch.bincount(y_test, minlength=10)
    checks["all_classes_present_each_split"] = bool(
        (train_counts > 0).all() and (val_counts > 0).all() and (test_counts > 0).all()
    )

    train_float = x_train[train_idx].float() / 255.0
    mean, std = train_float.mean().item(), train_float.std().item()
    checks["normalization_stats_finite"] = bool(
        torch.isfinite(torch.tensor([mean, std])).all() and std > 0
    )

    print("\\nFASHIONMNIST DATA AUDIT")
    print("-" * 62)
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\\nClass distribution")
    print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8}")
    for index, name in enumerate(CLASS_NAMES):
        print(f"{name:<15} {int(train_counts[index]):>8} {int(val_counts[index]):>8} {int(test_counts[index]):>8}")

    print(f"\\nTrain-only normalization: mean={mean:.6f}, std={std:.6f}")
    print(f"\\nSummary: {sum(checks.values())}/{len(checks)} checks passed")
    return checks


if __name__ == "__main__":
    run_audit()
