import numpy as np


def integrate_lebedev(f_nx):
    """Integrate the function f(r) on the angular Lebedev quadrature.

    Here, n is the quadrature index for the angular dependence of the function
    defined on a spherical grid, while x are some arbitrary extra dimensions.
    """
    from gpaw.sphere.lebedev import weight_n
    return 4. * np.pi * np.tensordot(weight_n, f_nx, axes=([0], [0]))
