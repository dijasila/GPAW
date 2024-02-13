import numpy as np

from gpaw import debug


def get_nmG(kpt1, kpt2, mypawcorr, n, qpd, I_G, pair_calc, timer=None):
    if timer:
        timer.start('utcc and pawcorr multiply')
    ut1cc_R = kpt1.ut_nR[n].conj()
    C1_aGi = mypawcorr.multiply(kpt1.P_ani, band=n)
    if timer:
        timer.stop('utcc and pawcorr multiply')
    n_mG = pair_calc.calculate_pair_density(
        ut1cc_R, C1_aGi, kpt2, qpd, I_G)
    return n_mG


def compare_dicts(dict1, dict2, rel_tol=1e-14, abs_tol=1e-14):
    """
    Compare each key-value pair within dictionaries that contain nested data
    structures of arbitrary depth. If a kvp contains floats, you may specify
    the tolerance (abs or rel) to which the floats are compared. Individual
    elements within lists are not compared to floating point precision.

    :params dict1: Dictionary containing kvp to compare with other dictionary.
    :params dict2: Second dictionary.
    :params rel_tol: Maximum difference for being considered "close",
    relative to the magnitude of the input values as defined by math.isclose().
    :params abs_tol: Maximum difference for being considered "close",
    regardless of the magnitude of the input values as defined by
    math.isclose().

    :returns: bool indicating kvp's don't match (False) or do match (True)
    """
    from math import isclose
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1.keys():
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, dict) and isinstance(val2, dict):
            # recursive func call to ensure nested structures are also compared
            if not compare_dicts(val1, val2, rel_tol, abs_tol):
                return False
        elif isinstance(val1, float) and isinstance(val2, float):
            if not isclose(val1, val2, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
        else:
            if val1 != val2:
                return False

    return True


class Sigma:
    def __init__(self, iq, q_c, fxc, esknshape, **inputs):
        """Inputs are used for cache invalidation, and are stored for each
           file.
        """
        self.iq = iq
        self.q_c = q_c
        self.fxc = fxc
        self._buf = np.zeros((2, *esknshape))
        # self-energies and derivatives:
        self.sigma_eskn, self.dsigma_eskn = self._buf

        self.inputs = inputs

    def sum(self, comm):
        comm.sum(self._buf)

    def __iadd__(self, other):
        self.validate_inputs(other.inputs)
        self._buf += other._buf
        return self

    def validate_inputs(self, inputs):
        equals = compare_dicts(inputs, self.inputs, rel_tol=1e-14,
                               abs_tol=1e-14)
        if not equals:
            raise RuntimeError('There exists a cache with mismatching input '
                               f'parameters: {inputs} != {self.inputs}.')

    @classmethod
    def fromdict(cls, dct):
        instance = cls(dct['iq'], dct['q_c'], dct['fxc'],
                       dct['sigma_eskn'].shape, **dct['inputs'])
        instance.sigma_eskn[:] = dct['sigma_eskn']
        instance.dsigma_eskn[:] = dct['dsigma_eskn']
        return instance

    def todict(self):
        return {'iq': self.iq,
                'q_c': self.q_c,
                'fxc': self.fxc,
                'sigma_eskn': self.sigma_eskn,
                'dsigma_eskn': self.dsigma_eskn,
                'inputs': self.inputs}


class SigmaCache:

class SigmaIntegrator:
    def integrate_sigma(self, ie, k, kpt1, kpt2, qpd, Wdict,
                        *, symop, sigmas, blocks1d, pawcorr, pair_calc,
                        bands, fxc_modes):
        """Calculates the contribution to the self-energy and its derivative
        for a given set of k-points, kpt1 and kpt2."""
        mypawcorr, I_G = symop.apply_symop_q(qpd, pawcorr, kpt1, kpt2)

        for n in range(kpt1.n2 - kpt1.n1):
            eps1 = kpt1.eps_n[n]
            n_mG = get_nmG(kpt1, kpt2, mypawcorr,
                           n, qpd, I_G, pair_calc)

            if symop.sign == 1:
                n_mG = n_mG.conj()

            f_m = kpt2.f_n
            deps_m = eps1 - kpt2.eps_n

            nn = kpt1.n1 + n - bands[0]

            assert set(Wdict) == set(sigmas)
            for fxc_mode in fxc_modes:
                sigma = sigmas[fxc_mode]
                Wmodel = Wdict[fxc_mode]
                # m is band index of all (both unoccupied and occupied) wave
                # functions in G
                for m, (deps, f, n_G) in enumerate(zip(deps_m, f_m, n_mG)):
                    # 2 * f - 1 will be used to select the branch of Hilbert
                    # transform, see get_HW of screened_interaction.py
                    # at FullFrequencyHWModel class.
                    S_GG, dSdw_GG = Wmodel.get_HW(deps, 2 * f - 1)
                    if S_GG is None:
                        continue

                    nc_G = n_G.conj()

                    # ie: ecut index for extrapolation
                    # kpt1.s: spin index of *
                    # k: k-point index of *
                    # nn: band index of *
                    # * wave function, where the sigma expectation value is
                    # evaluated
                    slot = ie, kpt1.s, k, nn
                    myn_G = n_G[blocks1d.myslice]
                    sigma.sigma_eskn[slot] += (myn_G @ S_GG @ nc_G).real
                    sigma.dsigma_eskn[slot] += (myn_G @ dSdw_GG @ nc_G).real



