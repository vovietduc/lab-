import torch

from src.dataset import compute_stats, normalize, split_indices


def test_split_disjoint_and_sizes():
    tr, va = split_indices(1000, 100, seed=1)
    assert len(tr) == 900 and len(va) == 100
    assert len(set(tr.tolist()) & set(va.tolist())) == 0


def test_split_reproducible():
    a = split_indices(500, 50, seed=7)
    b = split_indices(500, 50, seed=7)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_normalize_shape_and_stats():
    x = torch.randint(0, 256, (200, 28, 28), dtype=torch.uint8)
    mean, std = compute_stats(x, torch.arange(200))
    z = normalize(x, mean, std)
    assert z.shape == (200, 1, 28, 28) and z.dtype == torch.float32
    assert abs(z.mean().item()) < 1e-3 and abs(z.std().item() - 1) < 1e-2
