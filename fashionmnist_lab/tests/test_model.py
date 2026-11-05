import pytest
import torch
from src.model import MLPClassifier, CNNClassifier


@pytest.mark.parametrize("model", [MLPClassifier(), CNNClassifier()])
def test_output_shape(model):
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)


@pytest.mark.parametrize("model", [MLPClassifier(), CNNClassifier()])
def test_forward_batch_size_one(model):
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 28, 28))
    assert output.shape == (1, 10)
