from gpaw.xc.functional import XCFunctional
import numpy as np
from gpaw.xc.lda import PurePythonLDAKernel, lda_c, lda_x, lda_constants

class WLDA(XCFunctional):
    def __init__(self, kernel=None):
        XCFunctional.__init__(self, 'WLDA','LDA')
        if kernel is None:
            kernel = PurePythonLDAKernel()
        self.kernel = kernel
        self.weight = self._theta_filter


    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        #Fermi wavelength is given by k_f^3 = 3*pi^2 n
        self.apply_weighting(gd, n_sg)


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

    def apply_weighting(self, gd, n_sg):
        if n_sg.shape[0] > 1:

            raise NotImplementedError
        K_G = self._get_K_G(gd, n_sg)

        fn_G = np.fft.fftn(n_sg[0])


        #assert (Nx, Ny, Nz) == fn_G.shape This is not true
        k_F = K_G[5,5,5]
        
        self.weight(k_F, n_sg, K_G, fn_G)
        
        q = np.fft.ifftn(fn_G)
        b = np.allclose(q, q.real)
        #if not b:
        #    print(q.imag)
        #assert b
        n_sg[0,:] = q.real




    def _get_K_G(self, gd, n_sg):
        dKx, dKy, dKz = 2*np.pi/np.diag(gd.cell_cv)
        dx, dy, dz = gd.coords(0)[1] - gd.coords(0)[0], gd.coords(1)[1] - gd.coords(1)[0], gd.coords(2)[1] - gd.coords(2)[0]
        Nx, Ny, Nz = n_sg[0].shape #len(gd.coords(0)), len(gd.coords(1)), len(gd.coords(2)) doesnt work, not same shape as density
        ##But then dx,dy,dz is inconsistent with Nx,Ny,Nz
        
        reciprocal_vecs = gd.icell_cv.T #get dKs from this
        
        K_G = np.indices((Nx, Ny, Nz)).T.dot(2*np.pi/np.array([dx*Nx, dy*Ny, dz*Nz])).T

        return K_G

    def _theta_filter(self, k_F, K_G, n_G):
        
        Theta_G = (K_G**2 < 2*k_F**2).astype(np.complex128)

        return n_G*Theta_G


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

        

                      
