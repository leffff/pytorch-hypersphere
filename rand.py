import torch


def nsphere_randn_spherical(shape: tuple, stretch_coefficient: float = 1.0) -> torch.Tensor:
    """
    Generate a random tensor in N-spherical space with the given shape and stretch coefficient.

    :param shape: The shape of the tensor to generate.
    :param stretch_coefficient: The stretch coefficient to apply to the tensor.

    :return: A tensor with the given shape and stretch coefficient in N-spherical space.
    """
    x = torch.randn(shape)
    x_norm = torch.norm(x, dim=1, keepdim=True)
    x_spherical = x / x_norm * stretch_coefficient

    return x_spherical


def euclidean_randn_spherical(shape: tuple, stretch_coefficient: float = 1.0) -> torch.Tensor:
    """
    Generate a random tensor in Euclidean space with the given shape and stretch coefficient,
    and convert it to hyper-spherical space.

    :param shape: The shape of the tensor to generate.
    :param stretch_coefficient: The stretch coefficient to apply to the tensor.

    :return: A tensor with the given shape and stretch coefficient in hyper-spherical space.
    """
    r = torch.randn(shape) * stretch_coefficient
    theta = torch.FloatTensor(*shape).uniform_(0.0)
    x_spherical = torch.stack([r, theta.cos(), theta.sin()], dim=-1)

    return x_spherical
