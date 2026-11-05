"""Hàm tiện ích dùng chung: seed, thiết bị, đo lường, vẽ biểu đồ."""
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # lưu ảnh ra file, không cần cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def peak_vram_mb(device: torch.device):
    """VRAM cao nhất đã dùng (MB); None nếu chạy CPU."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


@torch.no_grad()
def measure_latency_ms(model, device, input_shape=(1, 1, 28, 28), warmup: int = 10, runs: int = 50) -> float:
    """Thời gian suy luận trung bình trên 1 ảnh (ms)."""
    model.eval()
    x = torch.randn(*input_shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def plot_history(history: dict, path, title: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ep = range(1, len(history["train_loss"]) + 1)
    best = int(np.argmax(history["val_acc"])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, history["train_loss"], marker="o", label="train")
    axes[0].plot(ep, history["val_loss"], marker="o", label="val")
    axes[0].set_title(f"Loss {title}"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(ep, history["train_acc"], marker="o", label="train")
    axes[1].plot(ep, history["val_acc"], marker="o", label="val")
    axes[1].axvline(best, color="gray", ls="--", label=f"tốt nhất (epoch {best})")
    axes[1].set_title(f"Accuracy {title}"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion_matrix(cm, class_names, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title("Confusion matrix (tập test)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im); plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def show_predictions(x, y_true, y_pred, class_names, mean, std, path, indices, title: str = "") -> None:
    """Lưới ảnh: dòng trên = dự đoán, trong ngoặc = thực tế; xanh đúng, đỏ sai."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = 8
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.2))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), indices):
        img = x[i].squeeze().cpu().numpy() * std + mean  # đảo chuẩn hóa để hiển thị
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ok = y_true[i] == y_pred[i]
        ax.set_title(f"{class_names[y_pred[i]]}\n({class_names[y_true[i]]})", fontsize=7,
                     color="green" if ok else "red")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
