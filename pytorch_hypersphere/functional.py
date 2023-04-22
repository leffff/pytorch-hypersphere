import torch
from torch import arccos, cos, sin


def to_hypersphere(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: torch.Tensor, tensor in euclidean coordinates, shape [batch_size, embedding_dim]
    :return: torch.Tensor, tensor in N-sphere coordinates, shape [batch_size, embedding_dim]
    """
    input_dim = x.shape[1]

    r = torch.norm(x, dim=1).reshape(-1, 1)

    phis = []
    for i in range(input_dim - 2):
        norm_i = torch.norm(x[:, i:], dim=1)
        phi_i = arccos(x[:, i] / norm_i).reshape(-1, 1)
        phis.append(phi_i)

    phi_last = arccos(x[:, input_dim - 2] / torch.norm(x[:, input_dim - 2 :], dim=1))
    neg_mask = x[:, input_dim - 1] < 0
    phi_last[neg_mask] = 2 * torch.pi - phi_last[neg_mask]
    phi_last = phi_last.reshape(-1, 1)

    hyperspherical = torch.cat([r, torch.cat(phis, dim=1), phi_last], dim=1)

    return hyperspherical


def to_euclidean(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: torch.Tensor, tensor in N-sphere coordinates, shape [batch_size, embedding_dim]
    :return: torch.Tensor, tensor in euclidean coordinates, shape [batch_size, embedding_dim]
    """
    input_dim = x.shape[1]

    r = x[:, 0]

    xs = []

    cur_base = r
    x_0 = cur_base * cos(x[:, 1])
    xs.append(x_0.reshape(-1, 1))
    for i in range(2, input_dim):
        cur_base = cur_base * sin(x[:, i - 1])
        x_i = cur_base * cos(x[:, i])
        xs.append(x_i.reshape(-1, 1))

    x_last = cur_base * sin(x[:, input_dim - 1])
    xs.append(x_last.reshape(-1, 1))

    euclidean = torch.cat(xs, dim=1)

    return euclidean


def project_to_sphere(sphere_center: torch.Tensor, sphere_radius: float, x: torch.Tensor):
    """
    :param sphere_center: torch.Tensor
    :param sphere_radius: torch.Tensor
    :param x: torch.Tensor
    :return: torch.Tensor

    Solution was taken from here: https://stackoverflow.com/questions/9604132/how-to-project-a-point-on-to-a-sphere
    """

    delta = sphere_radius / x.norm(dim=1) * x
    x_proj = delta + sphere_center

    return x_proj