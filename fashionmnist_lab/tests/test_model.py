import pytest
import torch

from src.model import FashionMNISTClassifier, MLPBaseline


@pytest.mark.parametrize("model", [FashionMNISTClassifier(pretrained=False), MLPBaseline()])
def test_output_shape(model):
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_backbone_frozen_head_trainable():
    m = FashionMNISTClassifier(pretrained=False, freeze_backbone=True)
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    m.train()
    assert not m.backbone.training  # BatchNorm của backbone luôn ở eval khi đóng băng
