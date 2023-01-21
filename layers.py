import torch
from torch import arccos, cos, nn, sin


class ToHyperSphere(nn.Module):
    def __init__(self, input_dim):
        """
        :param input_dim: embedding dimensions
        """
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor, tensor in euclidean coordinates, shape [batch_size, embedding_dim]
        :return: torch.Tensor, tensor in N-sphere coordinates, shape [batch_size, embedding_dim]
        """
        if x.shape[1] != self.input_dim:
            raise Exception

        r = torch.norm(x, dim=1).reshape(-1, 1)

        phis = []
        for i in range(self.input_dim - 2):
            norm_i = torch.norm(x[:, i:], dim=1)
            phi_i = arccos(x[:, i] / norm_i).reshape(-1, 1)
            phis.append(phi_i)

        phi_last = arccos(
            x[:, self.input_dim - 2] / torch.norm(x[:, self.input_dim - 2 :], dim=1)
        )
        neg_mask = x[:, self.input_dim - 1] < 0
        phi_last[neg_mask] = 2 * torch.pi - phi_last[neg_mask]
        phi_last = phi_last.reshape(-1, 1)

        hyperspherical = torch.cat([r, torch.cat(phis, dim=1), phi_last], dim=1)

        return hyperspherical

    def __repr__(self) -> str:
        return f"ToHyperSphere({self.input_dim})"


class ToEuclidean(nn.Module):
    def __init__(self, input_dim):
        """
        :param input_dim: embedding dimensions
        """
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor, tensor in N-sphere coordinates: shape [batch_size, embedding_dim]
        :return: torch.Tensor, tensor in euclidean coordinates: shape [batch_size, embedding_dim]
        """
        if x.shape[1] != self.input_dim:
            raise Exception

        r = x[:, 0]

        xs = []

        cur_base = r
        x_0 = cur_base * cos(x[:, 1])
        xs.append(x_0.reshape(-1, 1))
        for i in range(2, self.input_dim):
            cur_base = cur_base * sin(x[:, i - 1])
            x_i = cur_base * cos(x[:, i])
            xs.append(x_i.reshape(-1, 1))

        x_last = cur_base * sin(x[:, self.input_dim - 1])
        xs.append(x_last.reshape(-1, 1))

        euclidean = torch.cat(xs, dim=1)

        return euclidean

    def __repr__(self) -> str:
        return f"ToEuclidean({self.input_dim})"
