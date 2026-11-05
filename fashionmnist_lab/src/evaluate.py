"""Đánh giá mô hình trên tập TEST: Accuracy, F1, Loss, Latency, VRAM, confusion matrix, ảnh dự đoán.

Chạy: python -m src.evaluate [--config configs/default.yaml] [--checkpoint đường_dẫn.pth]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import (count_params, get_device, measure_latency_ms, peak_vram_mb, plot_confusion_matrix,
                       save_json, set_seed, show_predictions)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, total_loss, n = [], [], 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item()
        n += x.size(0)
        ps.append(out.argmax(1).cpu()); ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), total_loss / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ckpt_path = args.checkpoint or cfg["output"]["model_path"]
    reports_dir = Path(cfg["output"]["reports_dir"])

    set_seed(cfg["seed"])
    device = get_device()
    d = cfg["data"]
    loaders, info = get_dataloaders(d["root"], d["val_size"], d["batch_size"], d["num_workers"],
                                    cfg["seed"], False, pin_memory=device.type == "cuda")
    ckpt = torch.load(ckpt_path, map_location=device)
    assert abs(ckpt["mean"] - info["mean"]) < 1e-6, "mean/std lệch với lúc huấn luyện: kiểm tra seed và val_size"

    model = build_model(cfg, pretrained=False).to(device)   # trọng số lấy từ checkpoint
    model.load_state_dict(ckpt["state_dict"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    y_true, y_pred, test_loss = predict(model, loaders["test"], device)
    vram = peak_vram_mb(device)
    latency = measure_latency_ms(model, device)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    names = info["class_names"]
    report = classification_report(y_true, y_pred, target_names=names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {"model": cfg["model"]["name"], "test_accuracy": acc, "test_macro_f1": macro_f1,
               "test_loss": test_loss, "latency_ms_per_image": latency, "peak_vram_mb_eval": vram,
               "params_total": count_params(model), "params_trainable": count_params(model, True)}
    save_json(metrics, reports_dir / "test_metrics.json")
    (reports_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(cm, names, reports_dir / "confusion_matrix.png")

    x_test = loaders["test"].dataset.tensors[0]
    rng = np.random.default_rng(cfg["seed"])
    show_predictions(x_test, y_true, y_pred, names, info["mean"], info["std"],
                     reports_dir / "predictions_random.png", rng.choice(len(y_true), 16, replace=False),
                     "16 ảnh test ngẫu nhiên (dòng trên: dự đoán | trong ngoặc: thực tế)")
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) > 0:
        show_predictions(x_test, y_true, y_pred, names, info["mean"], info["std"],
                         reports_dir / "predictions_wrong.png", rng.choice(wrong, min(16, len(wrong)), replace=False),
                         "Các ảnh bị dự đoán sai (dòng trên: dự đoán | trong ngoặc: thực tế)")

    print(report)
    print(f"Test accuracy: {acc:.4f} | macro-F1: {macro_f1:.4f} | loss: {test_loss:.4f}")
    print(f"Latency: {latency:.2f} ms/ảnh | Peak VRAM (eval): {vram if vram is None else round(vram)} MB")
    print(f"Đã lưu kết quả vào: {reports_dir}")


if __name__ == "__main__":
    main()
