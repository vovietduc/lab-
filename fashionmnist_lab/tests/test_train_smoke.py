"""Dry Run tự động: 1 epoch trên dữ liệu giả, loss phải hữu hạn."""
import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import MLPBaseline
from src.train import fit


def test_fit_one_epoch_on_fake_data():
    torch.manual_seed(0)
    x, y = torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    model = MLPBaseline()
    history, best = fit(model, loader, loader, torch.device("cpu"), epochs=1, verbose=False)
    assert len(history["train_loss"]) == 1
    assert math.isfinite(history["train_loss"][0]) and 0.0 <= best <= 1.0
