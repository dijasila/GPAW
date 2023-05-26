import numpy as np


def integrate_lebedev(f_nx):
    """Integrate the function f(r) on the angular Lebedev quadrature.

    Here, n is the quadrature index for the angular dependence of the function
    defined on a spherical grid, while x are some arbitrary extra dimensions.
    """
    from gpaw.sphere.lebedev import weight_n
    return 4. * np.pi * np.tensordot(weight_n, f_nx, axes=([0], [0]))


class RealSphericalHarmonicsExpansion:
    """
    Some documentation here! XXX
    """

    def __init__(self, f_gM, Y_nL, L_M=None):
        """
        Some documentation here! XXX
        """
        self.f_gM = f_gM
        self.Y_nL = Y_nL

        if L_M is None:
            # Assume that all the composite indices L=(l,m) are represented
            assert f_gM.shape[1] == self.nL
            L_M = np.arange(self.nL)
        self.L_M = L_M

    @property
    def nL(self):
        return self.Y_nL.shape[1]


def calculate_rshe(f_ng, Y_nL) -> RealSphericalHarmonicsExpansion:
    r"""Expand the function f(r) in real spherical harmonics.

            / ^    ^     ^
    f (r) = |dr Y (r) f(rr)
     lm     /    lm

    Note that the Lebedev quadrature, which is used to perform the angular
    integral above, is exact up to polynomial order l=11. This implies that
    expansion coefficients up to l=5 are exact.

    Parameters
    ----------
    Documentation here! XXX
    """
    lmax = min(int(np.sqrt(Y_nL.shape[1])) - 1, 36)
    nL = (lmax + 1)**2
    L_L = np.arange(nL)

    # Perform the real spherical harmonics expansion
    f_ngL = np.repeat(f_ng, nL, axis=1).reshape((*f_ng.shape, nL))
    Y_ngL = np.repeat(Y_nL[:, L_L], f_ng.shape[1],
                      axis=0).reshape((*f_ng.shape, nL))
    f_gL = integrate_lebedev(Y_ngL * f_ngL)

    return RealSphericalHarmonicsExpansion(f_gL, Y_nL)
