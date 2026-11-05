"""Suy luận trên MỘT ảnh và in kết quả dạng JSON.

Chạy: python -m src.predict --image duong_dan.png [--checkpoint fashion_mnist_model.pth]
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from src.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="fashion_mnist_model.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt["config"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    img = Image.open(args.image).convert("L").resize((28, 28))          # xám, 28x28
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    x = ((x - ckpt["mean"]) / ckpt["std"]).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,28,28)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    result = {"label": int(probs.argmax()), "class": ckpt["classes"][int(probs.argmax())],
              "confidence": round(float(probs.max()), 4),
              "probabilities": {c: round(float(p), 4) for c, p in zip(ckpt["classes"], probs)}}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
