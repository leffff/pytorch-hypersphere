import torch


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
        phi_i = torch.arccos(x[:, i] / norm_i).reshape(-1, 1)
        phis.append(phi_i)

    phi_last = torch.arccos(x[:, input_dim - 2] / torch.norm(x[:, input_dim - 2:], dim=1))
    neg_mask = x[:, input_dim - 1] < 0
    phi_last[neg_mask] = 2 * torch.pi - phi_last[neg_mask]
    phi_last = phi_last.reshape(-1, 1)

    hyper_spherical = torch.cat([r, torch.cat(phis, dim=1), phi_last], dim=1)

    return hyper_spherical


def to_euclidean(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: torch.Tensor, tensor in N-sphere coordinates, shape [batch_size, embedding_dim]
    :return: torch.Tensor, tensor in euclidean coordinates, shape [batch_size, embedding_dim]
    """
    input_dim = x.shape[1]

    r = x[:, 0]

    xs = []

    cur_base = r
    x_0 = cur_base * torch.cos(x[:, 1])
    xs.append(x_0.reshape(-1, 1))
    for i in range(2, input_dim):
        cur_base = cur_base * torch.sin(x[:, i - 1])
        x_i = cur_base * torch.cos(x[:, i])
        xs.append(x_i.reshape(-1, 1))

    x_last = cur_base * torch.sin(x[:, input_dim - 1])
    xs.append(x_last.reshape(-1, 1))

    euclidean = torch.cat(xs, dim=1)

    return euclidean
