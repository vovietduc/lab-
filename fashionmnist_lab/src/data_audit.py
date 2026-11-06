
"""FashionMNIST data audit.

Run from the project root:
    python -m src.data_audit

Optional:
    python -m src.data_audit --data-root ./data --val-size 10000 --seed 42
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torchvision import datasets


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Audit FashionMNIST dataset.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--val-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./reports/data_audit")
    return parser.parse_args()


def save_class_distribution(output_dir, train_counts, val_counts, test_counts):
    csv_path = output_dir / "class_distribution.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Train", "Validation", "Test"])

        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([
                name,
                int(train_counts[i]),
                int(val_counts[i]),
                int(test_counts[i]),
            ])

    try:
        import matplotlib.pyplot as plt

        x = range(len(CLASS_NAMES))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            [i - width for i in x],
            train_counts.tolist(),
            width,
            label="Train",
        )
        ax.bar(
            x,
            val_counts.tolist(),
            width,
            label="Validation",
        )
        ax.bar(
            [i + width for i in x],
            test_counts.tolist(),
            width,
            label="Test",
        )

        ax.set_title("FashionMNIST Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of images")
        ax.set_xticks(list(x))
        ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(output_dir / "class_distribution.png", dpi=200)
        plt.close(fig)

        return True

    except ImportError:
        return False


def run_audit(root="./data", val_size=10_000, seed=42,
              output_dir="./reports/data_audit"):

    if not 0 < val_size < 60_000:
        raise ValueError("val_size must be between 1 and 59999.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nFASHIONMNIST DATA AUDIT")
    print("=" * 62)

    train_dataset = datasets.FashionMNIST(
        root=root, train=True, download=True
    )
    test_dataset = datasets.FashionMNIST(
        root=root, train=False, download=True
    )

    x_train = train_dataset.data
    y_train = train_dataset.targets.long()
    x_test = test_dataset.data
    y_test = test_dataset.targets.long()

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(x_train), generator=generator)

    val_idx = permutation[:val_size]
    train_idx = permutation[val_size:]

    train_images = x_train[train_idx]
    val_images = x_train[val_idx]

    train_labels = y_train[train_idx]
    val_labels = y_train[val_idx]

    train_counts = torch.bincount(train_labels, minlength=10)
    val_counts = torch.bincount(val_labels, minlength=10)
    test_counts = torch.bincount(y_test, minlength=10)

    train_float = train_images.float() / 255.0
    mean = train_float.mean().item()
    std = train_float.std().item()

    train_idx_set = set(train_idx.tolist())
    val_idx_set = set(val_idx.tolist())

    checks = {
        "train_count_60000": len(x_train) == 60_000,
        "test_count_10000": len(x_test) == 10_000,
        "image_shape_28x28": (
            tuple(x_train.shape[1:]) == (28, 28)
            and tuple(x_test.shape[1:]) == (28, 28)
        ),
        "grayscale_uint8": (
            x_train.dtype == torch.uint8
            and x_test.dtype == torch.uint8
        ),
        "pixel_range_0_255": (
            int(x_train.min()) >= 0
            and int(x_train.max()) <= 255
            and int(x_test.min()) >= 0
            and int(x_test.max()) <= 255
        ),
        "labels_in_0_9": (
            bool(((y_train >= 0) & (y_train <= 9)).all())
            and bool(((y_test >= 0) & (y_test <= 9)).all())
        ),
        "all_classes_present": (
            set(y_train.unique().tolist()) == set(range(10))
            and set(y_test.unique().tolist()) == set(range(10))
        ),
        "train_val_disjoint": len(train_idx_set & val_idx_set) == 0,
        "split_sizes_correct": (
            len(train_idx) == len(x_train) - val_size
            and len(val_idx) == val_size
        ),
        "split_covers_original_train": (
            len(train_idx_set | val_idx_set) == len(x_train)
        ),
        "all_classes_present_each_split": (
            bool((train_counts > 0).all())
            and bool((val_counts > 0).all())
            and bool((test_counts > 0).all())
        ),
        "normalization_stats_finite": (
            torch.isfinite(torch.tensor([mean, std])).all().item()
            and std > 0
        ),
    }

    print("-" * 62)

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")

    print("\nCLASS DISTRIBUTION")
    print("-" * 62)
    print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8}")

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:<15}"
            f"{int(train_counts[i]):>8}"
            f"{int(val_counts[i]):>8}"
            f"{int(test_counts[i]):>8}"
        )

    print(f"\nTrain-only normalization: mean={mean:.6f}, std={std:.6f}")

    passed_count = sum(checks.values())
    total_count = len(checks)

    print(f"\nSUMMARY: {passed_count}/{total_count} checks passed")

    summary = {
        "dataset": "FashionMNIST",
        "seed": seed,
        "validation_size": val_size,
        "train_size": len(train_idx),
        "test_size": len(x_test),
        "image_shape": list(x_train.shape[1:]),
        "normalization": {
            "mean": mean,
            "std": std,
            "computed_from": "training split only",
        },
        "class_distribution": {
            name: {
                "train": int(train_counts[i]),
                "validation": int(val_counts[i]),
                "test": int(test_counts[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "checks": checks,
        "passed": passed_count,
        "total": total_count,
        "all_passed": all(checks.values()),
    }

    with (output_dir / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    log_lines = [
        "FASHIONMNIST DATA AUDIT",
        "=" * 62,
    ]

    for name, passed in checks.items():
        log_lines.append(f"[{'PASS' if passed else 'FAIL'}] {name}")

    log_lines.extend([
        "",
        f"Mean: {mean:.6f}",
        f"Std: {std:.6f}",
        f"Summary: {passed_count}/{total_count} checks passed",
    ])

    (output_dir / "audit_log.txt").write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )

    save_class_distribution(
        output_dir,
        train_counts,
        val_counts,
        test_counts,
    )

    print(f"\nSaved audit files to: {output_dir.resolve()}")

    return checks


if __name__ == "__main__":
    args = parse_args()

    run_audit(
        root=args.data_root,
        val_size=args.val_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
