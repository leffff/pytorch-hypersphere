# pytorch-hypersphere

A simple yet efficient implementation of transformation from Euclidian to Hyperspherical (N-spherical) space

The library uses [this Wikipedia article](https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates) as a basis

Package listing:

* layers

```python

import torch


from layers import ToHyperSphere, ToEuclidean


ths = ToHyperSphere(16) # initialize transformation layer

te = ToEuclidean(16) # initialize transformation layer


x_eucl = torch.randn((4, 16)) # random floats in euclidian space

x_sphere = ths(x_eucl) # transformation to hyperspherical

x_eucl_2 = te(x_sphere) # transformation back to euclidean

```


* functional

```python

import torch

from functional import to_hypersphere, to_euclidean


x_eucl = torch.randn((4, 16)) # random floats in euclidian space

x_sphere = to_hypersphere(x_eucl) # transformation to hyperspherical

x_eucl_2 = to_euclidean(x_sphere) # transformation back to euclidean

```



* rand

```python

from rand import randn_spherical


random_hyperspherical_points = randn_spherical(shape=(4, 16)) # generate points randomly distributed on a sphere

```


