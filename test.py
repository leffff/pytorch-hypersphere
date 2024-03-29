import torch

from pytorch_hypersphere.layers import ToEuclidean, ToHyperSphere

ths = ToHyperSphere(16)
te = ToEuclidean(16)

REQUIRED_PRECISION = 1e-6

for i in range(10):
    x_eucl = torch.randn((4, 16))
    x_sphere = ths(x_eucl)
    x_eucl_2 = te(x_sphere)

    error = torch.abs(x_eucl - x_eucl_2).mean().item()
    print("PASSED") if error < REQUIRED_PRECISION else print("FAILED")
