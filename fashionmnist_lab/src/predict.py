"""Dự đoán một ảnh bằng checkpoint đã huấn luyện."""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="reports/cnn/fashion_mnist_cnn.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image = Image.open(args.image).convert("L").resize((28, 28))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)
    tensor = ((tensor - checkpoint["mean"]) / checkpoint["std"])
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    predicted = int(probabilities.argmax())
    result = {
        "label": predicted,
        "class": checkpoint["classes"][predicted],
        "confidence": round(float(probabilities[predicted]), 4),
        "probabilities": {
            name: round(float(prob), 4)
            for name, prob in zip(checkpoint["classes"], probabilities)
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
