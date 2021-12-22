"""Compute site-kernels. Used for computing Heisenberg exchange.
Specifically, one maps DFT calculations onto a Heisenberg lattice model,
where the site-kernels define the lattice sites and magnetic moments."""

import numpy as np

def sinc(x):
    """np.sinc(x) = sin(pi*x) / (pi*x), hence the division by pi"""
    return np.sinc(x / np.pi)
