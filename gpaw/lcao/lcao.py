from gpaw.kpoint import KPoint
from gpaw.utilities.blas import axpy

class LCAOKPoint(KPoint):
    """Special KPoint class for LCAO calculation"""
    
    def __init__(self, nuclei, gd, weight, s, k, u, k_c, dtype):
        KPoint.__init__(self, gd, weight, s, k, u, k_c, dtype)
        self.nuclei = nuclei
        
    def add_to_density(self, nt_G):
        """Add contribution to pseudo electron-density."""
        psit_G = self.gd.zeros()
        for n in range(self.nbands):
            psit_G[:] = 0.0
            m1 = 0
            for nucleus in self.nuclei:
                niao = nucleus.get_number_of_atomic_orbitals()
                m2 = m1 + niao
                nucleus.phit_i.add(psit_G, self.C_nm[n, m1:m2].real.copy())
                m1 = m2
            axpy(self.f_n[n], psit_G**2, nt_G)  # nt_G += f * psit_G**2



        
