import numpy as np

from gpaw.utilities import packed_index

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
        assert(len(self.calc.wfs.kpt_u) == 1)
        assert(len(other.wfs.kpt_u) == 1)

        overlap_nn = np.zeros((self.n, no), dtype=self.calc.wfs.dtype)
        mkpt = self.calc.wfs.kpt_u[0]
        okpt = other.wfs.kpt_u[0]
        psit_nG = mkpt.psit_nG
        norm_n = self.gd.integrate(psit_nG.conj() * psit_nG)
        psito_nG = okpt.psit_nG
        normo_n = other.wfs.gd.integrate(psito_nG.conj() * psito_nG)
        for i in range(self.n):
            p_nG = np.repeat(psit_nG[i].conj()[np.newaxis], no, axis=0)
            overlap_nn[i] = self.gd.integrate(p_nG * psito_nG)
            if normalize:
                overlap_nn[i] /= np.sqrt(np.repeat(norm_n[i], no) * normo_n)
        return overlap_nn

    def full(self, other):
        """Overlap of Kohn-Sham states including local terms.

        Parameter
        ---------
        other: gpaw
            gpaw-object containing wave functions
 
        Returns
        -------
        out: array
            u_ij =  \int dx mypsi_i^*(x) otherpsi_j(x)
        """
        ov_nn = self.pseudo(other, normalize=False)
        assert(len(self.calc.wfs.kpt_u) == 1)
        mkpt = self.calc.wfs.kpt_u[0]
        assert(len(other.wfs.kpt_u) == 1)
        okpt = other.wfs.kpt_u[0]

        for a, mP_ni in mkpt.P_ani.items():
            oP_ni = okpt.P_ani[a]
            Delta_p = (np.sqrt(4 * np.pi) *
                       self.calc.wfs.setups[a].Delta_pL[:,0])
            for n0, mP_i in enumerate(mP_ni):
                for n1, oP_i in enumerate(oP_ni):
                    ni = len(mP_i)
                    assert(len(oP_i) == ni)
                    for i, mP in enumerate(mP_i):
                        for j, oP in enumerate(oP_i):
                            ij = packed_index(i, j, ni)
                            ov_nn[n0, n1] += Delta_p[ij] * mP.conj() * oP
        return ov_nn
