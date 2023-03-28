import torch
from torch import nn


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
            raise ValueError(f"Expected {self.input_dim} dimensions, nut found {x.shape[1]}")

        r = x[:, 0]

        xs = []

        cur_base = r
        x_0 = cur_base * torch.cos(x[:, 1])
        xs.append(x_0.reshape(-1, 1))
        for i in range(2, self.input_dim):
            cur_base = cur_base * torch.sin(x[:, i - 1])
            x_i = cur_base * torch.cos(x[:, i])
            xs.append(x_i.reshape(-1, 1))

        x_last = cur_base * torch.sin(x[:, self.input_dim - 1])
        xs.append(x_last.reshape(-1, 1))

        euclidean = torch.cat(xs, dim=1)

        return euclidean

    def __repr__(self) -> str:
        return f"ToEuclidean({self.input_dim})"
