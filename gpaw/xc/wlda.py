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
            
            
            zeta = 0
            
            lda_c(0, e_g, n, v_sg[0], zeta)

    def apply_weighting(self, gd, n_sg):
        if n_sg.shape[0] > 1:
            raise NotImplementedError
        self.tabulate_weights(n_sg[0])

        n_g = n_sg[0]
        wn_g = np.zeros_like(n_g)

        
        n_gi = self._get_ni_weights(n_g)
        wtable_gi = self.weight_table

        wn_g = np.einsum("ijkl, ijkl -> ijk", n_gi, wtable_gi)

        n_sg[0, :] = wn_g
                    
                    



    def get_ni_weights(self, n_g):
        nis = self.nis
        bign_g = np.zeros(n_g.shape + nis.shape)
        for i in range(len(nis)):
            bign_g[..., i] = n_g
        lesser_g = bign_g <= nis
        nlesser_g = lesser_g.astype(int).sum(axis=-1)

        greater_g = bign_g > nis
        ngreater_g = greater_g.sum(axis=-1)

        vallesser_g = nis.take(nlesser_g-1)
        valgreater_g = nis.take(-ngreater_g)

        print("")
        print(nis)
        print("################################")
        print(lesser_g)
        print("################################")
        print(nlesser_g)
        print("################################")
        print(n_g)
        print("################################")
        print(vallesser_g)

        #partlesser_g = 

        n_gi = np.zeros_like(bign_g)
        
        
        return n_gi


    def _get_ni_weights(self, n_g):
        flatn_g = n_g.reshape(-1)

        n_gi = np.array([self._get_ni_vector(n) for n in n_g.reshape(-1)]).reshape(n_g.shape + (len(self.nis),))


        return n_gi
            
        
                    
    def _get_ni_vector(self, n):
        #Find best superposition of nis to interpolate to n
        nis = self.nis
        ni_vector = np.zeros_like(nis)
        nlesser = (nis <= n).sum()
        ngreater = (nis > n).sum()
        if ngreater == 0:
            d = n - nis[-1]
            m1 = 1 + d/(nis[-1]-nis[-2])
            m2 = -d/(nis[-1]-nis[-2])
            ni_vector[-1] = m1
            ni_vector[-2] = m2
            return ni_vector
        
        if ngreater == len(nis):
            d = n - nis[0]
            m1 = 1 - d/(nis[1]-nis[0])
            m2 = d/(nis[1]-nis[0])
            ni_vector[0] = m1
            ni_vector[1] = m2
            return ni_vector

        vallesser = nis[nlesser - 1]
        valgreater = nis[-ngreater]
        
        partlesser = (valgreater-n)/(valgreater-vallesser)
        partgreater = (n - vallesser)/(valgreater-vallesser)


        ni_vector[nlesser-1] = partlesser
        ni_vector[-ngreater] = partgreater

        return ni_vector

        


    def _get_K_G(self, shape):
        Nx, Ny, Nz = shape
        kxs = np.array([2*np.pi/Nx*i if i < Nx/2 else 2*np.pi/Nx*(i-Nx) for i in range(Nx)])
        kys = np.array([2*np.pi/Ny*i if i < Ny/2 else 2*np.pi/Ny*(i-Ny) for i in range(Ny)])
        kzs = np.array([2*np.pi/Nz*i if i < Nz/2 else 2*np.pi/Nz*(i-Nz) for i in range(Nz)])
        K_G = np.array([[[np.linalg.norm([kx, ky, kz]) for kz in kzs] for ky in kys] for kx in kxs])

        return K_G

    def _theta_filter(self, k_F, K_G, n_G):
        
        Theta_G = (K_G**2 <= 4*k_F**2).astype(np.complex128)

        return n_G*Theta_G

    def tabulate_weights(self, n_g):
        nis = np.arange(0, max(np.max(n_g), 5), 0.1)
        K_G = self._get_K_G(n_g.shape)
        self.nis = nis
        self.weight_table = np.zeros(n_g.shape+(len(nis),))
        n_G = np.fft.fftn(n_g)

        for i, ni in enumerate(nis):
            k_F = (3*np.pi**2*ni)**(1/3)
            fil_n_G = self._theta_filter(k_F, K_G, n_G)
            x = np.fft.ifftn(fil_n_G)
            assert np.allclose(x, x.real)
            self.weight_table[:,:,:,i] = x.real

        
       

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
        

