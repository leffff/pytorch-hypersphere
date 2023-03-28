import torch
from torch import nn


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
            raise ValueError(f"Expected {self.input_dim} dimensions, nut found {x.shape[1]}")

        r = torch.norm(x, dim=1).reshape(-1, 1)

        phis = []
        for i in range(self.input_dim - 2):
            norm_i = torch.norm(x[:, i:], dim=1)
            phi_i = torch.arccos(x[:, i] / norm_i).reshape(-1, 1)
            phis.append(phi_i)

        phi_last = torch.arccos(
            x[:, self.input_dim - 2] / torch.norm(x[:, self.input_dim - 2:], dim=1)
        )
        neg_mask = x[:, self.input_dim - 1] < 0
        phi_last[neg_mask] = 2 * torch.pi - phi_last[neg_mask]
        phi_last = phi_last.reshape(-1, 1)

        hyper_spherical = torch.cat([r, torch.cat(phis, dim=1), phi_last], dim=1)

        return hyper_spherical

    def __repr__(self) -> str:
        return f"ToHyperSphere({self.input_dim})"
