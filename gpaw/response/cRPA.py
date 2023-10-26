import numpy as np
from gpaw.mpi import world
from gpaw.response.chi0 import find_maximum_frequency, Chi0Calculator
from gpaw.response.pair import get_gs_and_context
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pair import KPointPairFactory
from ase.units import Ha


class cRPA:
    def __init__(self, extra_weights_nk, bandrange, gs, context):
        # Probability that state nk is in wannier subspace
        self.extra_weights_nk = extra_weights_nk
        self.bandrange = bandrange
        self.gs = gs
        self.context = context

    @classmethod
    def from_wannier_matrix(cls, Uwan_wnk, bandrange, calc,
                            txt='chi0.txt', timer=None, world=world):
        """Initialize cRPA_weight from Wannier transformation matrix
        Uwan_wnk: cmplx or str
                  if cmplx: Wannier transfirmation matrix:
                  w = wannier index, n band index, k k-index
                  if str: name of wannier90 output file with transformation
                  matrix
        bandrange: int
                   Range of bands that Wannier functions were constructed
                   from
        calc:      str
                   gpw-file
        txt:       str
                   output file
        """

        from gpaw.wannier90 import read_uwan
        gs, context = get_gs_and_context(calc, txt, world, timer)
        nbands = gs.bd.nbands
        kd = gs.kd
        bandrange = np.arange(bandrange[0], bandrange[1])
        # if Uwan is string try to read wannier90 matrix
        if isinstance(Uwan_wnk, str):
            seed = Uwan_wnk
            assert kd is not None
            Uwan_wnk, nk, nw1, nw2 = read_uwan(seed, kd)
            assert nw2 == len(bandrange)
            assert nk == kd.nbzkpts
        extra_weights_nk = np.zeros((nbands, nk))
        extra_weights_nk[bandrange, :] = np.sum(np.abs(Uwan_wnk)**2, axis=0)
        assert np.allclose(np.sum(extra_weights_nk, axis=0), nw1)
        return cls(extra_weights_nk, bandrange, gs, context)

    @classmethod
    def from_band_indexes(cls, bands, calc,
                          txt='chi0.txt', timer=None, world=world):
        """Initialize cRPA_weight from Wannier transformation matrix
        bands:     list(int)
                   list with bandindexes that the screening should be
                   removed from
        calc:      str
                   gpw-file
        txt:       str
                   output file
        """
        gs, context = get_gs_and_context(calc, txt, world, timer)
        nbands = gs.bd.nbands
        nk = gs.kd.nbzkpts
        extra_weights_nk = np.zeros((nbands, nk))
        extra_weights_nk[bands, :] = 1
        assert np.allclose(np.sum(extra_weights_nk, axis=0), len(bands))
        return cls(extra_weights_nk, bands, gs, context)

    """
    Note: One can add more classmethods to initialize extra weight
    from bandindex or from energywindow
    """

    def get_constrained_chi0_calculator(self,
                                        frequencies={'type': 'nonlinear'},
                                        ecut=50,
                                        nblocks=1,
                                        nbands=None,
                                        **kwargs):

        nbands = nbands or self.gs.bd.nbands

        if (isinstance(frequencies, dict) and
            frequencies.get('omegamax') is None):
            omegamax = find_maximum_frequency(self.gs.kpt_u, self.context,
                                              nbands=nbands)
            frequencies['omegamax'] = omegamax * Ha

        wd = FrequencyDescriptor.from_array_or_dict(frequencies)
        kptpair_factory = KPointPairFactory(self.gs,
                                            self.context,
                                            nblocks=nblocks)

        return Chi0Calculator(wd=wd, kptpair_factory=kptpair_factory,
                              nbands=nbands, ecut=ecut,
                              crpa_weight=self, **kwargs)

    def get_weight_nm(self, n_n, m_m, k1_c, k2_c):
        """ weight_nm = 1. - P_n*Pm where
        P_n is probability that state n is in model
        subspace.
        """
        ikn = self.gs.kd.where_is_q(k1_c, self.gs.kd.bzk_kc)
        ikm = self.gs.kd.where_is_q(k2_c, self.gs.kd.bzk_kc)
        weights_n = self.extra_weights_nk[n_n, ikn]
        weights_m = self.extra_weights_nk[m_m, ikm]
        weights_nm = - np.outer(weights_n, weights_m)
        weights_nm += 1.0
        return weights_nm

    def get_drude_weight_n(self, n_n, k_c):
        """ weight_nm = 1. - P_n*Pm where
        P_n is probability that state n is in model
        subspace.
        """
        ikn = self.gs.kd.where_is_q(k_c, self.gs.kd.bzk_kc)
        weights_n = 1. - self.extra_weights_nk[n_n, ikn]**2
        return weights_n
