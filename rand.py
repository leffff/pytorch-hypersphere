import torch


def randn_spherical(shape: tuple) -> torch.Tensor:
    x = torch.randn(shape)
    x_spherical = x / torch.norm(x, dim=1)

    return x_spherical
