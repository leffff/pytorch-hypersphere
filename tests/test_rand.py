import torch

from rand import euclidean_randn_spherical, nsphere_randn_spherical


def test_nsphere_randn_spherical():
    shape = (4, 16)
    stretch_coefficient = 2.0
    x = nsphere_randn_spherical(shape, stretch_coefficient)
    assert x.shape == shape
    assert torch.allclose(torch.norm(x, dim=1), torch.ones(shape[0]) * stretch_coefficient, rtol=1e-6)


def test_euclidean_randn_spherical():
    shape = (4, 16)
    stretch_coefficient = 2.0
    x = euclidean_randn_spherical(shape, stretch_coefficient)
    assert x.shape == shape + (3,)
    assert torch.all(x[..., 1:].cos() >= -1.0) and torch.all(x[..., 1:].cos() <= 1.0)
