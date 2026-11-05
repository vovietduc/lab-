"""Đánh giá mô hình trên tập TEST và lưu metrics/biểu đồ."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import (count_params, get_device, measure_latency_ms, peak_vram_mb,
                       plot_confusion_matrix, save_json, set_seed, show_predictions)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, total_loss, n = [], [], 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        n += x.size(0)
        ps.append(logits.argmax(1).cpu())
        ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    checkpoint_path = args.checkpoint or cfg["output"]["model_path"]
    reports_dir = Path(cfg["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = get_device()
    data_cfg = cfg["data"]
    loaders, info = get_dataloaders(
        data_cfg["root"], data_cfg["val_size"], data_cfg["batch_size"],
        data_cfg["num_workers"], cfg["seed"], False,
        pin_memory=device.type == "cuda"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if abs(checkpoint["mean"] - info["mean"]) >= 1e-6 or abs(checkpoint["std"] - info["std"]) >= 1e-6:
        raise ValueError("Mean/std khác checkpoint. Hãy dùng đúng seed và cấu hình dữ liệu khi train.")

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    y_true, y_pred, test_loss = predict(model, loaders["test"], device)
    vram = peak_vram_mb(device)
    latency = measure_latency_ms(model, device)

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    class_names = info["class_names"]
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "model": cfg["model"]["name"],
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_loss": test_loss,
        "latency_ms_per_image": latency,
        "peak_vram_mb_eval": vram,
        "params_total": count_params(model),
        "params_trainable": count_params(model, True),
    }
    save_json(metrics, reports_dir / "test_metrics.json")
    (reports_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    plot_confusion_matrix(matrix, class_names, reports_dir / "confusion_matrix.png")

    x_test = loaders["test"].dataset.tensors[0]
    rng = np.random.default_rng(cfg["seed"])
    show_predictions(
        x_test, y_true, y_pred, class_names, info["mean"], info["std"],
        reports_dir / "predictions_random.png",
        rng.choice(len(y_true), 16, replace=False),
        "Ảnh test ngẫu nhiên: dự đoán (thực tế)"
    )
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong):
        show_predictions(
            x_test, y_true, y_pred, class_names, info["mean"], info["std"],
            reports_dir / "predictions_wrong.png",
            rng.choice(wrong, min(16, len(wrong)), replace=False),
            "Các ảnh dự đoán sai: dự đoán (thực tế)"
        )

    print(report_text)
    print(f"Test accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f} | Loss: {test_loss:.4f}")
    print(f"Latency: {latency:.3f} ms/ảnh | Peak VRAM: {vram} MB")
    print(f"Kết quả đã lưu tại: {reports_dir}")


if __name__ == "__main__":
    main()
