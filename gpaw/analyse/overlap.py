import numpy as np


class Overlap:
    """Wave funcion overlap of two GPAW objects"""
    def __init__(self, calc):
        self.calc = calc
        self.n = calc.get_number_of_bands() * calc.density.nspins
        self.gd = self.calc.wfs.gd

    def pseudo(self, other, normalize=True):
        """Overlap with pseudo wave functions only

        Parameter
        ---------
        other: gpaw
            gpaw-object containing pseudo wave functions
        normalize: bool
            normalize pseudo wave functions in the overlap integral

        Returns
        -------
        out: array
            u_ij =  \int dx mypsitilde_i^*(x) otherpsitilde_j(x)
        """
        no = other.get_number_of_bands() * other.density.nspins
        assert(self.calc.density.nspins == 1)
        assert(other.density.nspins == 1)

        overlap_nn = np.zeros((self.n, no), dtype=self.calc.wfs.dtype)
        psit_nG = self.calc.wfs.kpt_u[0].psit_nG
        norm_n = self.gd.integrate(psit_nG.conj() * psit_nG)
        psito_nG = other.wfs.kpt_u[0].psit_nG
        normo_n = other.wfs.gd.integrate(psito_nG.conj() * psito_nG)
        for i in range(self.n):
            p_nG = np.repeat(psit_nG[i].conj()[np.newaxis], no, axis=0)
            overlap_nn[i] = self.gd.integrate(p_nG * psito_nG)
            if normalize:
                overlap_nn[i] /= np.sqrt(np.repeat(norm_n[i], no) * normo_n)
        return overlap_nn

