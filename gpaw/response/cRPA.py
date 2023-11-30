import numpy as np
from gpaw.mpi import world
from gpaw.response.chi0 import find_maximum_frequency, Chi0Calculator
from gpaw.response.pair import get_gs_and_context
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pair import KPointPairFactory
from ase.units import Ha


class cRPA:
    def __init__(self, extra_weights_nk, bandrange, gs, context, factor = 1.):
        # Probability that state nk is in wannier subspace
        self.extra_weights_nk = extra_weights_nk
        self.bandrange = bandrange
        self.gs = gs
        self.context = context
        self.factor = factor # weight = (1 - p*p)**factor. spex has factor = 2, but I think it is wrong
    @classmethod
    def from_wannier_matrix(cls, Uwan_mnk, bandrange, calc,
                            wannier_range=None, round_weights = False,
                            factor = 1.,
                            txt='chi0.txt',
                            timer=None, world=world):
        """Initialize cRPA_weight from Wannier transformation matrix
        Uwan_mnk: cmplx or str
                  if cmplx: Wannier transfirmation matrix:
                  w = wannier index, n band index, k k-index
                  if str: name of wannier90 output file with transformation
                  matrix
        bandrange: int
                   Range of bands that Wannier functions were constructed
                   from
        calc:      str
                   gpw-file
        wannier_range: list(int) or None
                   If not None only remove screening from selected
                   wannier functions in list
        txt:       str
                   output file
        """

        from gpaw.wannier90 import read_uwan
        gs, context = get_gs_and_context(calc, txt, world, timer)
        context.print("Initializing cRPA from wannier matrix")
        nbands = gs.bd.nbands
        kd = gs.kd
        bandrange = np.arange(bandrange[0], bandrange[-1])
        # if Uwan is string try to read wannier90 matrix
        if isinstance(Uwan_mnk, str):
            seed = Uwan_mnk
            assert kd is not None
            Uwan_mnk, nk, nw1, nw2 = read_uwan(seed, kd, dis=False)
            if nw2 < len(bandrange):
                Uwan_mnk, nk, nw1, nw2 = read_uwan(seed, kd,
                                                   dis=True)
            assert nw2 == len(bandrange)
            assert nk == kd.nbzkpts
            nwan = len(Uwan_mnk)
            assert nw1 == nwan
        else:
            nwan = len(Uwan_mnk)

        if wannier_range is not None:
            assert(len(Uwan_mnk) > max(wannier_range))
        else:
            wannier_range = range(nwan)
        context.print("Removing screening from Wannier functions:")
        context.print(wannier_range)
        extra_weights_nk = np.zeros((nbands, nk))
        extra_weights_nk[bandrange, :] = \
            np.sum(np.abs(Uwan_mnk[wannier_range])**2,
                                                axis=0)
        if round_weights:
            extra_weights_nk = np.rint(extra_weights_nk)
        assert np.allclose(np.sum(extra_weights_nk, axis=0),
                           len(wannier_range))
        return cls(extra_weights_nk, bandrange, gs, context, factor=factor)

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
        context.print("Initializing cRPA from band indexes")
        context.print("Removing screening from bands:")
        context.print(bands)
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

    def get_weight_nm(self, n_n, m_m, ikn, ikm):
        """ weight_nm = 1. - P_n*Pm where
        P_n is probability that state n is in model
        subspace.
        """
        weights_n = self.extra_weights_nk[n_n, ikn]
        weights_m = self.extra_weights_nk[m_m, ikm]
        weights_nm = - np.outer(weights_n, weights_m)
        weights_nm += 1.0
        weights_nm[weights_nm <= 1e-20] = 0.0
        return weights_nm ** self.factor

    def get_drude_weight_n(self, n_n, ikn):
        """ weight_nm = 1. - P_n*Pm where
        P_n is probability that state n is in model
        subspace.
        """
        weights_n = 1. - self.extra_weights_nk[n_n, ikn]**2
        weights_n[weights_n <= 1e-20] = 0.0
        return weights_n ** self.factor
