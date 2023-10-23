import numpy as np
from gpaw.response import ResponseGroundStateAdapter


class cRPA_weight:
    def __init__(self, extra_weights_nk, bandrange, nk):
        # Probability that state nk is in wannier subspace
        self.extra_weights_nk = extra_weights_nk
        self.bandrange = bandrange
        self.nk = nk

    @classmethod
    def from_wannier_matrix(cls, Uwan_wnk, bandrange, gs):
        """Initialize cRPA_weight from Wannier transformation matrix
        Uwan_wnk: cmplx or str
                  if cmplx: Wannier transfirmation matrix:
                  w = wannier index, n band index, k k-index
                  if str: name of wannier90 output file with transformation
                  matrix
        bandrange: int
                   Range of bands that Wannier functions were constructed
                   from
        nbands:    int
                   total number of bands in the calculation
        gs: ResponseGroundStateAdapter or str (path to gpw file)
        """

        from gpaw.wannier90 import read_uwan
        context = ResponseContext(txt=self.filename + '.txt',
                                  comm=world, timer=None)
        if isinstance(gs, str):
            gs = ResponseGroundStateAdapter.from_gpw_file(gs,
                                                          context=context)
        nbands = gs.nbands
        kd = gs.kd

        # if Uwan is string try to read wannier90 matrix
        if isinstance(Uwan_wnk, str):
            seed = Uwan_wnk
            assert kd is not None
            Uwan_wnk, nk, nw1, nw2 = read_uwan(seed, kd)
            assert nw2 == len(bandrange)
        extra_weights_nk = np.zeros((nbands, nk))
        extra_weights_nk[bandrange, :] = np.sum(np.abs(Uwan_wnk)**2, axis=0)
        assert np.allclose(np.sum(extra_weights_nk, axis=0), nw1)
        return cls(extra_weights_nk, bandrange, nk)

    """
    Note: One can add more classmethods to initialize extra weight
    from bandindex or from energywindow
    """

    def get_weight_nm(self, n_n, m_m, ikn, ikm):
        """ weight_nm = 1. - P_n*Pm where
        P_n is probability that state n is in model
        subspace.
        """
        nmax = max(n_n[-1], self.bandrange[-1])
        mmax = max(m_m[-1], self.bandrange[-1])
        weight_n = np.zeros(nmax)
        weight_n[self.bandrange] = self.extra_weight_nk[self.bandrange, ikn]
        weight_m = np.zeros(mmax)
        weight_m[self.bandrange] = self.extra_weight_nk[self.bandrange, ikm]
        weight_nm = - np.outer(weight_n[n_n], weight_m[m_m])
        weight_nm += 1.0
        return weight_nm
