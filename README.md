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

from rand import euclidean_randn_spherical, nsphere_randn_spherical


random_points_on_sphere_in_euclidean = euclidean_randn_spherical(shape=(4, 16), stretch_coefficient=2) # generate points randomly distributed on a sphere, in euclidean coordinates, with radius of 2
random_points_on_sphere_in_nsphere = nsphere_randn_spherical(shape=(4, 16), stretch_coefficient=1) # generate points randomly distributed on a sphere, in spherical coordinates, with radius of 1
```


