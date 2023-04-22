import torch


def spherical_distance(x1: torch.Tensor, x2: torch.Tensor, sphere_radius) -> float:
    """
    :param x1: torch.Tensor
    :param x2: torch.Tensor
    :param sphere_radius: float
    :return: float

    Solution was taken from here: https://math.stackexchange.com/questions/1304169/distance-between-two-points-on-a-sphere
    """
    d = (x1 * x2).sum(dim=-1) / (sphere_radius ** 2)
    return d
