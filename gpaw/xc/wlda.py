from gpaw.xc.functional import XCFunctional
import numpy as np


class WLDA(XCFunctional):
    def __init__(self, kernel=None):
        XCFunctional.__init__(self, 'WLDA','LDA')
        self.kernel = kernel



    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        from gpaw.xc.lda import lda_c
        prefactor = -0.5*3*(3/(4*np.pi))**(1/3)
        for s, n_g in enumerate(n_sg):
            v_sg[s, :] = prefactor*4/3*np.abs(n_g)**(1/3)*0.5
            #lda_c(s, e_g, n_g, v_sg[s], 0)

        if len(n_sg) == 2:
            e_g[:] = prefactor*(n_sg[0]**(4/3) + n_sg[1]**(4/3))
        else:
            e_g[:] = prefactor*(np.abs(n_sg[0])**(4/3))

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

        

                      
