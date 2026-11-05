"""Smoke test training loop với MLP."""
import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import MLPClassifier
from src.train import fit


def test_fit_one_epoch_on_fake_data():
    torch.manual_seed(0)
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    model = MLPClassifier()
    history, best_accuracy = fit(
        model, loader, loader, torch.device("cpu"), epochs=1, verbose=False
    )
    assert len(history["train_loss"]) == 1
    assert math.isfinite(history["train_loss"][0])
    assert 0.0 <= best_accuracy <= 1.0
