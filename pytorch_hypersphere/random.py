import torch


def nsphere_randn_spherical(shape: tuple, stretch_coefficient: float = 1.0) -> torch.Tensor:
    x = torch.randn(shape)
    x_spherical = x / torch.norm(x, dim=1) * stretch_coefficient

    return x_spherical


def euclidean_randn_spherical(shape: tuple, stretch_coefficient: float = 1.0) -> torch.Tensor:
    r = torch.randn(shape) * stretch_coefficient
    x_spherical = torch.FloatTensor(*shape).uniform_(0.0, torch.pi)
    x_spherical[:, 0] = r

    return x_spherical
