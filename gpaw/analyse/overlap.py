import numpy as np


class Overlap:
    """Wave funcion overlap of two GPAW objects"""
    def __init__(self, calc):
        self.calc = calc
        self.n = calc.get_number_of_bands()
        self.gd = self.calc.wfs.gd

    def pseudo(self, other):
        """Overlap with pseudo wave functions only"""
        no = other.get_number_of_bands()
        #    spin1 = calc1.get_number_of_spins() == 2
        #    spin2 = calc2.get_number_of_spins() == 2
        overlap_nn = np.zeros((self.n, no), dtype=self.calc.wfs.dtype)
        psit_nG = self.calc.wfs.kpt_u[0].psit_nG
        norm_n = self.gd.integrate(psit_nG.conj() * psit_nG)
        psito_nG = other.wfs.kpt_u[0].psit_nG
        normo_n = other.wfs.gd.integrate(psito_nG.conj() * psito_nG)
        for i in range(self.n):
            p_nG = np.repeat(psit_nG[i].conj()[np.newaxis], no, axis=0)
            overlap_nn[i] = (self.gd.integrate(p_nG * psito_nG) /
                np.sqrt(np.repeat(norm_n[i], no) * normo_n))
        return overlap_nn

