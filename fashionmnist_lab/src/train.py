"""Vòng lặp huấn luyện: forward -> loss -> backward -> optimizer, có validation, early stopping, AMP.

Chạy:
  python -m src.train --dry-run                  # Dry Run: 1 epoch trên 50 mẫu
  python -m src.train                            # Normal Run (configs/default.yaml)
  python -m src.train --config configs/mlp.yaml  # baseline MLP
"""
import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

import src.dataset
from src.model import build_model
from src.utils import (count_params, get_device, peak_vram_mb, plot_history, save_json, set_seed)


def make_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def get_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Optimizer không hỗ trợ: {name}")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    """Một epoch. optimizer=None -> chế độ validation (eval + no_grad)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)          # 1. xóa gradient batch trước
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)                                 # 2. forward
                loss = criterion(out, y)                       # 3. loss
            if training:
                scaler.scale(loss).backward()                  # 4. backward
                scaler.step(optimizer)                         # 5. cập nhật trọng số
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, device, epochs=15, lr=1e-3, optimizer_name="adam",
        weight_decay=0.0, patience=4, use_amp=False, verbose=True):
    """Huấn luyện; trả về (history, best_val_acc). Model được nạp lại trọng số tốt nhất."""
    use_amp = bool(use_amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(optimizer_name, params, lr, weight_decay)
    scaler = make_scaler(use_amp)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, bad = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        vl, va = run_epoch(model, val_loader, criterion, device, None, None, use_amp)
        for k, v in zip(history, (tl, vl, ta, va)):
            history[k].append(v)
        if va > best_acc:
            best_acc, bad = va, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose:
            print(f"Epoch {epoch:02d}/{epochs} | train loss {tl:.4f} acc {ta:.4f} | "
                  f"val loss {vl:.4f} acc {va:.4f} | {time.time() - t0:.1f}s")
        if bad >= patience:
            if verbose:
                print(f"Early stopping ở epoch {epoch} (val acc không cải thiện {patience} epoch liên tiếp)")
            break
    model.load_state_dict(best_state)
    return history, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cnn.yaml")
    ap.add_argument("--dry-run", action="store_true", help="1 epoch trên tập mẫu nhỏ (20-50 ảnh)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tc = cfg["train"]
    epochs = 1 if args.dry_run else (args.epochs or tc["epochs"])
    reports_dir = Path(cfg["output"]["reports_dir"])
    model_path = Path(cfg["output"]["model_path"])
    if args.dry_run:  # Dry Run không ghi đè kết quả thật
        reports_dir = Path("reports/dry_run")
        model_path = reports_dir / "dry_run_model.pth"

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    d = cfg["data"]
    loaders, info = src.dataset.get_dataloaders(d["root"], d["val_size"], d["batch_size"], d["num_workers"],
                                    cfg["seed"], args.dry_run, pin_memory=device.type == "cuda")
    print(f"Train {info['n_train']} | Val {info['n_val']} | Test {info['n_test']} | "
          f"mean={info['mean']:.4f} std={info['std']:.4f}")

    model = build_model(cfg).to(device)
    print(f"Model: {cfg['model']['name']} | tham số: {count_params(model):,} "
          f"(huấn luyện được: {count_params(model, True):,})")

    t0 = time.time()
    history, best_acc = fit(model, loaders["train"], loaders["val"], device, epochs, tc["lr"],
                            tc["optimizer"], tc["weight_decay"], tc["patience"], tc.get("amp", False))
    train_time = time.time() - t0

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "mean": info["mean"], "std": info["std"], "classes": info["class_names"]}, model_path)
    save_json(history, reports_dir / "history.json")
    plot_history(history, reports_dir / "training_curves.png", f"({cfg['model']['name']})")
    save_json({"best_val_acc": best_acc, "train_time_s": train_time, "epochs_run": len(history["val_acc"]),
               "peak_vram_mb_train": peak_vram_mb(device)}, reports_dir / "train_summary.json")
    print(f"\nBest val acc: {best_acc:.4f} | thời gian: {train_time:.0f}s | đã lưu: {model_path}")
    vram = peak_vram_mb(device)
    if vram is not None:
        print(f"Peak VRAM khi train: {vram:.0f} MB")


if __name__ == "__main__":
    main()
