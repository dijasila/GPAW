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

    @property
    def lmax(self):
        flmax = np.sqrt(self.nL)
        lmax = int(flmax)
        assert abs(flmax - lmax) < 1e-8
        return lmax

    @property
    def l_L(self):
        l_L = []
        for l in range(self.lmax + 1):
            l_L += [l] * (2 * l + 1)
        return l_L

    @property
    def l_M(self):
        return [self.l_L[L] for L in self.L_M]

    def evaluate_on_quadrature(self):
        Y_nM = self.Y_nL[:, self.L_M]
        return Y_nM @ self.f_gM.T

    def reduce_expansion(self, fns_g, lmax=-1, wmin=None):
        """Reduce the composite index L=(l,m) to M, which indexes coefficients
        contributing with a weight larger than rshewmin to the surface norm
        square on average.
        Remember to adjust documentation XXX

        Parameters
        ----------
        Some documentation here! XXX
        """
        # Redo me XXX
        # We do not expand beyond l=5
        if lmax == -1:
            lmax = 5
        assert lmax in range(6)
        wmin = wmin if wmin is not None else 0.
        assert isinstance(wmin, float) and wmin >= 0.

        # We assume to start with a full expansion
        assert self.f_gM.shape[1] == self.nL
        f_gL = self.f_gM

        # Filter away (l,m)-coefficients based on their average weight in
        # completing the surface norm square of df
        fw_gL = self._calculate_ns_weights(self.nL, f_gL, fns_g)
        rshew_L = np.average(fw_gL, axis=0)  # Average over the radial grid

        # Take rshe coefficients up to l <= lmax (<= 5) which contribute with
        # at least wmin to the surface norm square on average
        nL = min(self.nL, (lmax + 1)**2)
        L_L = np.arange(nL)
        L_M = np.where(rshew_L[L_L] > wmin)[0]
        f_gM = f_gL[:, L_M]
        reduced_rshe = RealSphericalHarmonicsExpansion(
            f_gM, self.Y_nL, L_M=L_M)

        # Construct info string about the reduced expansion
        info_string = self.get_reduction_info_string(nL, wmin, fw_gL, rshew_L)

        return reduced_rshe, info_string

    @staticmethod
    def _calculate_ns_weights(nL, df_gL, dfSns_g):
        """Calculate the weighted contribution of each rsh coefficient to the
        surface norm square of df as a function of radial grid index g."""
        # This function is in dire need of clean-up! XXX
        nallL = df_gL.shape[1]
        dfSns_gL = np.repeat(dfSns_g, nallL).reshape(dfSns_g.shape[0], nallL)
        dfSw_gL = df_gL ** 2 / dfSns_gL

        return dfSw_gL

    def get_reduction_info_string(self, nL, wmin, fw_gL, rshew_L):
        """
        Some documentation here! XXX
        """
        info_string = '{0:6}  {1:10}  {2:10}  {3:8}'.format('(l,m)',
                                                            'max weight',
                                                            'avg weight',
                                                            'included')
        for L, (fw_g, rshew) in enumerate(zip(fw_gL.T, rshew_L)):
            info_string += '\n' + self.get_rshe_coefficient_info_string(
                L, nL, rshew, wmin, fw_g)

        tot_avg_cov = np.average(np.sum(fw_gL, axis=1))
        avg_cov = np.average(
            np.sum(fw_gL[:, :nL][:, rshew_L[:nL] > wmin], axis=1))
        info_string += f'\nIn total: {avg_cov} of the surface norm square is '\
            'covered on average'
        info_string += f'\nIn total: {tot_avg_cov} of the surface norm '\
            'square could be covered on average'

        return info_string

    @staticmethod
    def get_rshe_coefficient_info_string(L, nL, rshew, wmin, fw_g):
        """
        Some documentation here! XXX
        """
        l = int(np.sqrt(L))
        m = L - l * (l + 1)
        included = 'yes' if (rshew > wmin and L < nL) else 'no'
        info_string = '{0:6}  {1:1.8f}  {2:1.8f}  {3:8}'.format(f'({l},{m})',
                                                                np.max(fw_g),
                                                                rshew,
                                                                included)
        return info_string


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
    # Redo this function XXX
    lmax = min(int(np.sqrt(Y_nL.shape[1])) - 1, 36)
    nL = (lmax + 1)**2
    L_L = np.arange(nL)

    # Perform the real spherical harmonics expansion
    f_ngL = np.repeat(f_ng, nL, axis=1).reshape((*f_ng.shape, nL))
    Y_ngL = np.repeat(Y_nL[:, L_L], f_ng.shape[1],
                      axis=0).reshape((*f_ng.shape, nL))
    f_gL = integrate_lebedev(Y_ngL * f_ngL)

    return RealSphericalHarmonicsExpansion(f_gL, Y_nL)


def calculate_reduced_rshe(f_ng, Y_nL, lmax=-1, wmin=None):
    """
    Some documentation here! XXX
    """
    rshe = calculate_rshe(f_ng, Y_nL)
    dfns_g = integrate_lebedev(f_ng ** 2)
    return rshe.reduce_expansion(dfns_g, lmax=lmax, wmin=wmin)
