from gpaw.xc.functional import XCFunctional
import numpy as np
from gpaw.xc.lda import PurePythonLDAKernel, lda_c, lda_x, lda_constants
from gpaw.utilities.tools import construct_reciprocal
import gpaw.mpi as mpi

class WLDA(XCFunctional):
    def __init__(self, kernel=None, mode=""):
        XCFunctional.__init__(self, 'WLDA','LDA')
        if kernel is None:
            kernel = PurePythonLDAKernel()
        self.kernel = kernel
        self.stepsize = 0.05
        self.nalpha = 20
        self.alpha_n = None
        self.mode = mode


        self.gd1 = None

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        if self.gd1 is None:
            self.gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        self.alpha_n = None #Reset alpha grid for each calc
        n1_sg = gd.collect(n_sg)
        if gd.comm.rank == 0:
            if self.mode == "":                
                self.apply_weighting(self.gd1, n1_sg)

            elif self.mode.lower() == "altcenter":
                n1_g = self.calculate_nstar(n1_sg[0], self.gd1)
                n1_sg[0, :] = n1_g
        
            elif self.mode.lower() == "renorm":
                norm_s = gd.integrate(n_sg)
                self.apply_weighting(self.gd1, n1_sg)
                newnorm_s = gd.integrate(n1_sg)
                n1_sg[0, :] = n1_sg[0,:]*norm_s[0]/newnorm_s[0]
            else:
                raise ValueError("WLDA mode not recognized")
        gd.distribute(n1_sg, n_sg)
        #n_sg[1, :] = n_sg[1,:]*norm_s[1]/newnorm_s[1]

        

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
        print("1")
        self.tabulate_weights(n_sg[0], gd)
        print("2")
        n_g = n_sg[0]
        wn_g = np.zeros_like(n_g)

        print("3")
        n_gi = self.get_ni_weights(n_g)
        print("4")
        wtable_gi = self.weight_table

        wn_g = np.einsum("ijkl, ijkl -> ijk", n_gi, wtable_gi)
        print("5")
        n_sg[0, :] = wn_g
                    
         
    def apply_other_weighting(self, gd, n_sg):
        if n_sg.shape[0] > 1:
            raise NotImplementedError

        n_g = n_sg[0]
        
        weight_function_gg = self.get_other_weightfunction(gd, n_g)

        n_g = gd.integrate(weight_function_gg, n_g)
        n_sg[0, :] = n_g


    def get_weightslice(self, gd, n_g, pos, num_cells=5):
        grid_vg = gd.get_grid_point_coordinates()

        _, Nx, Ny, Nz = grid_vg.shape

        cell_cv = gd.cell_cv
        
        w_g = np.zeros((Nx, Ny, Nz))
        for ix in range(Nx):
            for iy in range(Ny):
                w_g[ix, iy, :] = np.array([self.get_weightslice_value(grid_vg[:, ix, iy, iz]-pos, (3*np.pi**2*n_g[ix, iy, iz])**(1/3), gd, num_cells) for iz in range(Nz)])
                # for iz in range(Nz):
                #     kF = (3*np.pi**2*n_g[ix, iy, iz])**(1/3)



                #     r_v = grid_vg[:, ix, iy, iz] - pos                            
                #     w_g[ix, iy, iz] = self.get_weightslice_value(r_v, kF, gd, num_cells)
        

                #     for n in range(1, num_cells):
                #         for c, R_v in enumerate(cell_cv):
                #             R = n*R_v
                #             r_v = grid_vg[:, ix, iy, iz] + R - pos
                            
                #             w_g[ix, iy, iz] += self.get_weightslice_value(r_v, kF, gd, num_cells)
        
        return w_g


    def get_weightslice_value(self, r_v, kF, gd, num_cells):
        val = 0
        r = np.linalg.norm(r_v)
        if np.allclose(r, 0):
            val = 0
        else:
            val = (1/(2*np.pi**2))*(np.sin(2*kF*r)-2*kF*r*np.cos(2*kF*r))/(r**3)

        for n in range(1, num_cells):
            for c, R_v in enumerate(gd.cell_cv):
                R = n*R_v
                r_v = r_v + R
                r = np.linalg.norm(r_v)
                val += (1/(2*np.pi**2))*(np.sin(2*kF*r)-2*kF*r*np.cos(2*kF*r))/(r**3)
                    
        return val


    def get_other_weightfunction(self, gd, n_g):
        #grid = gd.get_grid_point_coordinates()
        #_, Nx, Ny, Nz = grid.shape
        Nx, Ny, Nz = n_g.shape
        #assert n_g.shape == grid_gv[:,:,:,0].shape
        grid = self._get_grid(gd)


        dists = np.array([np.linalg.norm(grid[:,ix,iy,iz]-grid[:,ix2,iy2,iz2]) for ix in range(Nx) for iy in range(Ny) for iz in range(Nz) for ix2 in range(Nx) for iy2 in range(Ny) for iz2 in range(Nz)])
        dists = [d for d in dists if not np.allclose(d, 0)]
        cutoff = min(dists)
        weight_function_gg = np.zeros((Nx, Ny, Nz, Nx, Ny, Nz))
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    for ix2 in range(Nx):
                        for iy2 in range(Ny):
                            for iz2 in range(Nz):
                                diffVec = grid[:, ix, iy, iz] - grid[:, ix2, iy2, iz2]
                                dist = np.linalg.norm(diffVec)
                                #dist = max(dist, cutoff)
                                if np.allclose(dist, 0):
                                    weight_function_gg[ix, iy, iz, ix2, iy2, iz2] = 0
                                else:
                                    dens = 2*(3*np.pi**2*n_g[ix2, iy2, iz2])**(1/3)
                                    weight_function_gg[ix, iy, iz, ix2, iy2, iz2] = (1/(2*np.pi**2))*(np.sin(dist*dens)-dist*dens*np.cos(dist*dens))/dist**3

        return weight_function_gg

    def _get_grid(self, gd):
        xs, ys, zs = gd.coords(0), gd.coords(1), gd.coords(2)
        Nx, Ny, Nz = len(xs), len(ys), len(zs)

        grid = np.zeros((3, Nx, Ny, Nz))

        na = np.newaxis
        grid[0, :, :, :] = xs[:, na, na]
        grid[1, :, :, :] = ys[na, :, na]
        grid[2, :, :, :] = zs[na, na, :]

        return grid
        
        


    def get_ni_weights(self, n_g):
        n_g = np.abs(n_g)
        nis = self.nis
        nlesser_g = (n_g[:, :, :, np.newaxis] >= nis).sum(axis=-1)
        ngreater_g = (n_g[:, :, :, np.newaxis] < nis).sum(axis=-1)
        assert (nlesser_g >= 1).all()
        assert (ngreater_g >= 0).all()

        vallesser_g = nis.take(nlesser_g-1)
        valgreater_g = nis.take(-ngreater_g)
        
        # vallesser_g2 = np.zeros_like(n_g)
        # valgreater_g2 = np.zeros_like(n_g)

        # for ix, n_yz in enumerate(nlesser_g):
        #     for iy, n_z in enumerate(n_yz):
        #         for iz, n in enumerate(n_z):
        #             vallesser_g2[ix, iy, iz] = nis[n-1]
        #             ng = ngreater_g[ix, iy, iz]
        #             valgreater_g2[ix, iy, iz] = nis[-ng]
                    
        # assert np.allclose(vallesser_g, vallesser_g2) ##This is true
        # assert np.allclose(valgreater_g, valgreater_g2) ##This is also true


        partlesser_g = (valgreater_g-n_g)/(valgreater_g-vallesser_g)
        partgreater_g = (n_g - vallesser_g)/(valgreater_g-vallesser_g)

        n_gi = np.zeros(n_g.shape + nis.shape)
        for ix, p_yz in enumerate(partlesser_g):
            for iy, p_z in enumerate(p_yz):
                for iz, p in enumerate(p_z):
                    nl = nlesser_g[ix, iy, iz]
                    n_gi[ix, iy, iz, nl - 1] = p
                    ng = ngreater_g[ix, iy, iz]
                    pg = partgreater_g[ix, iy, iz]
                    n_gi[ix, iy, iz, -ng] = pg

                
        
        return n_gi


    def _get_ni_weights(self, n_g):
        flatn_g = n_g.reshape(-1)

        n_gi = np.array([self._get_ni_vector(n) for n in n_g.reshape(-1)]).reshape(n_g.shape + (len(self.nis),))

#np.transpose(np.array([self._get_ni_vector(n) for n in n_g.reshape(-1)]).reshape((len(self.nis),) + n_g.shape), (1, 2, 3, 0))

#.reshape(n_g.shape + (len(self.nis),))

#.reshape((len(self.nis),) + n_g.shape), (1, 2, 3, 0))
        


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

        


    def _get_K_G(self, shape, gd):
        assert gd.comm.size == 1 #Construct_reciprocal doesnt work in parallel
        k2_Q, _ = construct_reciprocal(gd)
        k2_Q[0,0,0] = 0
        return k2_Q**(1/2)
        


    def _theta_filter(self, k_F, K_G, n_G):
        
        Theta_G = (K_G**2 <= 4*k_F**2).astype(np.complex128)

        return n_G*Theta_G

    def get_nis(self, n_g):
        return np.arange(0, max(np.max(n_g)+2*self.stepsize, 5), self.stepsize)

    def tabulate_weights(self, n_g, gd):
        '''
        Calculate \int  w(r-r', ni)n(r') dr'
        for various values of ni.

        This is later used to calculate nstar(r) = \int w(r-r', n(r))n(r')
        by interpolating the values at the ni to the value at n(r).

        '''
        n_g = np.abs(n_g)
        nis = self.get_nis(n_g)
        K_G = self._get_K_G(n_g.shape, gd)
        self.nis = nis
        self.weight_table = np.zeros(n_g.shape+(len(nis),))
        n_G = np.fft.fftn(n_g)

        for i, ni in enumerate(nis):
            k_F = (3*np.pi**2*ni)**(1/3)
            fil_n_G = self._theta_filter(k_F, K_G, n_G)
            x = np.fft.ifftn(fil_n_G)
            if (n_g >= 0).all():
                assert np.allclose(x, x.real)

            self.weight_table[:,:,:,i] = x.real

    
    def tabulate_weights2(self, n_g, gd):
        n_g = np.abs(n_g)
        nis = self.get_nis(n_g)
        K_G = self._get_K_G(n_g.shape, gd)
        self.nis = nis
        #self.weight_table = np.zeros(n_g.shape + (len(nis),))
        n_G = np.fft.fftn(n_g)
        k_Fi = (3*np.pi**2*nis)**(1/3)
        theta_Gi = ((K_G**2)[:, :, :, np.newaxis] <= 4*k_Fi**2).astype(np.complex128)
        filn_Gi = n_G[:,:, :, np.newaxis]*theta_Gi
        filn_gi = np.fft.ifftn(filn_Gi, axes=[0,1,2])
        self.weight_table = filn_gi.real



    def get_alpha_grid(self, n_g):
        if self.alpha_n is not None:
            return self.alpha_n
        
        alpha_n = np.linspace(0, np.max(n_g), self.nalpha)

        self.alpha_n = alpha_n
        return alpha_n


    
    def construct_cubic_splines(self, n_g):
        '''
        This construct cubic spline interpolations for the indicator functions f_alpha.
        See http://en.wikipedia.org/wiki/Spline_(mathematics) (subsection Algorithm for computing natural cubic splines)

        We here calculate the splines for all the indicator functions simultaneously, so our arrays have an extra index. The row-index enumerates the different splines making up and indicator, while the column index enumerates the different indicators

        f_alpha(x) should be 1 in x = alpha and zero otherwise. But this is too strict so we to interpolation instead.
        '''
        alpha_n = self.get_alpha_grid(n_g)
        nalpha = self.nalpha #~n+1 in algo on wikipedia

        y = np.eye(nalpha) #Target values for the indicator functions
        a = y 
        h = alpha_n[1:] - alpha_n[:-1]
        
        beta_k = 3*(a[2:] - a[1:-1])/h[1:, np.newaxis] - 3*(a[1:-1] - a[:-2])/h[:-1, np.newaxis]

        l = np.ones((nalpha, nalpha))
        mu = np.zeros((nalpha, nalpha))
        z = np.zeros((nalpha, nalpha))
        
        for i in range(1, nalpha-1):
            l[i] = 2*(alpha_n[i+1] - alpha_n[i-1]) - h[i-1]*mu[i-1]
            mu[i] = h[i]/l[i]
            z[i] = (beta_k[i-1] - h[i-1]*z[i-1])/l[i]
        
        b = np.zeros((nalpha, nalpha))
        c = np.zeros((nalpha, nalpha))
        d = np.zeros((nalpha, nalpha))
        
        for i in range(nalpha-2, -1, -1):
            c[i] = z[i] - mu[i]*c[i+1]
            b[i] = (a[i+1]-a[i])/h[i] - (h[i]*(c[i+1] + 2*c[i]))/3
            d[i] = (c[i+1] - c[i])/(3*h[i])

            
        #Now we switch the order, so the first index is the indicator, the second is the spline, the third is the coefficient
        #We also add an extra spline for every indicator. This is just set to be constant
        self.C_nsc = np.zeros((nalpha, nalpha, 4))
        self.C_nsc[:, :-1, 0] = a[:-1].T
        self.C_nsc[:, :-1, 1] = b[:-1].T
        self.C_nsc[:, :-1, 2] = c[:-1].T
        self.C_nsc[:, :-1, 3] = d[:-1].T
        self.C_nsc[-1, -1, 0] = 1.0 #The last indicator should be one at the last alpha-pt, therefore the extra spline should have value 1

        return self.C_nsc
                                         
        
    

    def get_spline_values(self, C_nsc, n_g, gd):
        grid_vg = gd.get_grid_point_coordinates()
        
        grid_vx = grid_vg.reshape(3, -1)

        n_x = n_g.reshape(-1)
        gridshape = n_g.shape
        # C_ng = np.array([[self.expand_spline(n, C_sc) for n in n_x] for C_sc in C_nsc]).reshape(len(C_nsc), *gridshape)
        # C_ng = np.array([self.expand_spline2(n_x, C_sc) for C_sc in C_nsc]).reshape(len(C_nsc), *gridshape)
        C_ng = self.expand_spline3(n_x, C_nsc).reshape(len(C_nsc), *gridshape)

        return C_ng


    def expand_spline3(self, n_x, C_nsc):
        n_x = np.abs(n_x)
        alpha_i = self.alpha_n
        nlesser_x = (n_x[:, np.newaxis] >= alpha_i).sum(axis=1)
        assert (nlesser_x >= 1).all()
        coeff_nsc = C_nsc[:, nlesser_x - 1]
        a_xn, b_xn, c_xn, d_xn = coeff_nsc.T
        alpha_x = alpha_i[nlesser_x - 1]
        a_nx, b_nx, c_nx, d_nx = a_xn.T, b_xn.T, c_xn.T, d_xn.T

        C_nx = a_nx + b_nx*(n_x-alpha_x) + c_nx*(n_x-alpha_x)**2 + d_nx*(n_x-alpha_x)**3

        return C_nx

    def expand_spline2(self, x, C_sc):
        x = np.abs(x)
        alpha_n = self.alpha_n
        nlessers = np.array([(alpha_n <= xval).astype(int).sum() for xval in x])
        assert (nlessers >= 1).all()
        a, b, c, d = C_sc[nlessers - 1].T
        alphas = alpha_n[nlessers - 1]

        return a + b*(x-alphas) + c*(x-alphas)**2 + d*(x-alphas)**3



    def expand_spline(self, x, C_sc):
        x = np.abs(x)
        alpha_n = self.alpha_n

        val = 0
        nlesser = (alpha_n <= x).astype(int).sum()
        assert nlesser >= 1
        a, b, c, d = C_sc[nlesser - 1]
        alpha = alpha_n[nlesser - 1]

        return a + b*(x-alpha) + c*(x-alpha)**2 + d*(x-alpha)**3



    def get_weight_alphaG(self, gd):
        alpha_n = self.get_alpha_grid(None)
        _, nx, ny, nz = gd.get_grid_point_coordinates().shape
        K_G = self._get_K_G((nx, ny, nz), gd)
        w_alphaG = np.zeros((len(alpha_n), *K_G.shape), dtype=np.complex128)

        for ia, alpha in enumerate(alpha_n):
            k_F = self.get_kF(alpha)
            step_fct_G = (K_G**2 <= 4*k_F**2).astype(np.complex128)
            w_alphaG[ia, :, :, :] = step_fct_G
        
        # norm_alpha = np.array([gd.integrate(np.fft.ifftn(w_G)) for w_G in w_alphaG])
        # w_alphaG = np.array([w_G/norm_alpha[a] for a, w_G in enumerate(w_alphaG)])
        #norm = gd.integrate(np.fft.ifftn(w_alphaG[0]))
        w_alphaG /= gd.dv
        return w_alphaG
    

    def calculate_nstar(self, n_g, gd):
        C_nsc = self.construct_cubic_splines(n_g)
        C_alphag = self.get_spline_values(C_nsc, n_g, gd)
        product_alphag = np.array([C_g*n_g for C_g in C_alphag])
        product_alphaG = np.array([np.fft.fftn(product_g) for product_g in product_alphag])
        w_alphaG = self.get_weight_alphaG(gd)
        integrand_alphaG = np.array([w_G*product_alphaG[ia] for ia, w_G in enumerate(w_alphaG)])
        integrand_G = np.sum(integrand_alphaG, axis=0)
        nstar_g = np.fft.ifftn(integrand_G)


        return nstar_g
       

    def get_kF(self, density):
        return (3*np.pi**2*density)**(1/3)


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
        

