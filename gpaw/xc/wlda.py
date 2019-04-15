from gpaw.xc.functional import XCFunctional
import numpy as np
from gpaw.xc.lda import PurePythonLDAKernel, lda_c, lda_x, lda_constants

class WLDA(XCFunctional):
    def __init__(self, kernel=None):
        XCFunctional.__init__(self, 'WLDA','LDA')
        if kernel is None:
            kernel = PurePythonLDAKernel()
        self.kernel = kernel



    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        #Fermi wavelength is given by k_f^3 = 3*pi^2 n

        from gpaw.xc.lda import lda_c
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        if len(n_sg) == 2:
            na = 2. * n_sg[0]
            na[na < 1e-20] = 1e-40
            nb = 2. * n_sg[1]
            nb[nb < 1e-20] = 1e-40
            n = 0.5 * (na + nb)
            zeta = 0.5 * (na - nb) / n
            lda_x(1, e_g, na, v_sg[0])
            lda_x(1, e_g, nb, v_sg[1])
            lda_c(1, e_g, n, v_sg, zeta)
        else:
            n = n_sg[0]
            n[n < 1e-20] = 1e-40
            rs = (C0I/n)**(1/3)
            ex = C1/rs
            dexdrs = -ex/rs
            e_g[:] = n*ex
            v_sg[0] += ex - rs*dexdrs/3
            
            #e_g[:] = prefactor*(np.abs(n_sg[0])**(4/3))
            zeta = 0
            #lda_x(0, e_g, n, v_sg[0])
            lda_c(0, e_g, n, v_sg[0], zeta)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        from gpaw.xc.lda import calculate_paw_correction
        from gpaw.xc.lda import LDARadialCalculator, LDARadialExpansion
        collinear = True
        rcalc = LDARadialCalculator(self.kernel)
        expansion = LDARadialExpansion(rcalc, collinear)
        corr = calculate_paw_correction(expansion,
                                        setup, D_sp, dEdD_sp,
                                        True, a)
        return corr

        

                      
