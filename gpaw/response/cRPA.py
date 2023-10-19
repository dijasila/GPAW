import numpy as np
from gpaw.response.chi0 import Chi0Calculator
from gpaw import mpi
from typing import Union
from gpaw.typing import Array1D

class cRPA_weight:
    def __init__(self, extra_weights_nk, bandrange, nk):
        #probability that state nk is in wannier subspace
        self.extra_weights_nk = extra_weights_nk
        self.bandrange = bandrange
        self.nk = nk

    @classmethod
    def from_wannier_matrix(cls, Uwan_wnk, bandrange, nbands, kd=None):
        from gpaw.wannier90 import read_uwan

        # if Uwan is string try to read wannier90 matrix
        if isinstance(Uwan_wnk, str):
            seed = Uwan_wnk
            assert kd is not None
            Uwan_wnk, nk, nw1, nw2 = read_uwan(seed, kd)
            assert nw2 == len(bandrange)
        extra_weights_nk = np.zeros((nbands, nk))
        extra_weights_nk[bandrange,:] = np.sum(np.abs(Uwan_wnk)**2, axis=0)
        assert np.allclose(np.sum(extra_weights_nk, axis=0), nw1)
        return cls(extra_weights_nk, bandrange, nk)

    """
    Note: One can add more classmethods to initialize extra weight
    from bandindex or from energywindow
    """


        
class cRPA(Chi0Calculator):
    """Class for calculating constrained non-interacting response functions
    """

    def __init__(self,
                 calc,
                 cRPA_weight,
                 *,
                 frequencies: Union[dict, Array1D] = None,
                 ecut=50,
                 threshold=1,
                 world=mpi.world, txt='-', timer=None,
                 nblocks=1,
                 nbands=None,
                 **kwargs):
        """Construct Chi0 object.

        Parameters
        ----------
        cRPA_weight: cRPA_weight object
        Remaining parameters: See Chi0

        """
        from gpaw.response.pair import get_gs_and_context
        gs, context = get_gs_and_context(calc, txt, world, timer)
        nbands = nbands or gs.bd.nbands

        wd = new_frequency_descriptor(
            gs, context, nbands, frequencies,
            domega0=domega0,
            omega2=omega2, omegamax=omegamax)

        pair = PairDensityCalculator(
            gs, context,
            threshold=threshold,
            nblocks=nblocks)
        assert len(extra_weights) == calc.numbands
        self.cRPA_weight = cRPA_weight
        super().__init__(wd=wd, pair=pair, nbands=nbands, ecut=ecut, **kwargs)

        
    def get_integrand(self, ...):
        return cRPAIntegrand(...)

class cRPAIntegrand(Chi0Integrand):

    def __init__(...):
        ...

    
    @timer('Get matrix element')
    def matrix_element(self, k_v, s):

        """Return pair density matrix element for integration.
        
        Overrides function in Chi0Integrand to add aditional
        weight in cRPA approach.
        """

        if self.optical:
            target_method = self.pair.get_optical_pair_density
            out_ngmax = self.qpd.ngmax + 2
        else:
            target_method = self.pair.get_pair_density
            out_ngmax = self.qpd.ngmax

        n_mnG = self._get_any_matrix_element(
            k_v, s, block=not self.optical,
            target_method=target_method,
        )
        n_nmG *= self.extra_pair_weights_nm[..., np.newaxis]**0.5

        
        return self.n_mnG.reshape(-1, out_ngmax)

