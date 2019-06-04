from gpaw.xc.functional import XCFunctional
import numpy as np
from gpaw.xc.lda import PurePythonLDAKernel, lda_c, lda_x, lda_constants
from gpaw.utilities.tools import construct_reciprocal
import gpaw.mpi as mpi

class WLDA(XCFunctional):
    def __init__(self, kernel=None, mode="", filter_kernel=""):
        XCFunctional.__init__(self, 'WLDA','LDA')
        if kernel is None:
            kernel = PurePythonLDAKernel()
        self.kernel = kernel
        self.stepsize = 0.1
        self.stepfactor = 1.1
        self.nalpha = 20
        self.alpha_n = None
        self.mode = mode
        self.lmax = 1
        self.n_gi = None

        self.gd1 = None

        if filter_kernel.lower() == "fermikinetic":
            self.filter_kernel = self._fermi_kinetic
        elif filter_kernel.lower() == "fermicoulomb":
            self.filter_kernel = self._fermi_coulomb
        else:
            filter_kernel = "step function filter"
            self.filter_kernel = self._theta_filter

        print("Using kernel: {}".format(filter_kernel))


        self.rcut = 0.0625 # In Bohr
        
    def initialize(self, density, hamiltonian, wfs, occupations):
        self.density = density #.D_asp
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.occupations = occupations

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        assert len(n_sg) == 1
        if self.gd1 is None:
            self.gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        self.alpha_n = None #Reset alpha grid for each calc

        n1_sg = gd.collect(n_sg)
        v1_sg = gd.collect(v_sg)
        e1_g = gd.collect(e_g)
        if gd.comm.rank == 0:
            self.calc_corrected_density(n1_sg)
            if self.mode == "" or self.mode == "normal":
                self.apply_weighting(self.gd1, n1_sg)
            
            elif self.mode.lower() == "altcenter":
                prenorm = self.gd1.integrate(n1_sg[0])
                n1_g = self.calculate_nstar(n1_sg[0], self.gd1)
                postnorm = self.gd1.integrate(n1_g)
                assert np.allclose(prenorm, postnorm)
                n1_sg[0, :] = n1_g
        
            elif self.mode.lower() == "renorm":
                norm_s = gd.integrate(n_sg)
                self.apply_weighting(self.gd1, n1_sg)
                # unnormed_n_sg = n1_sg.copy()
                newnorm_s = gd.integrate(n1_sg)
                # Dont renorm because we would have to save another copy of the unnormed array for the corrections.
                # Instead manually multiply by the renorm factor where needed
                # n1_sg[0, :] = n1_sg[0,:]
            elif self.mode.lower() == "constant":
                self.apply_const_weighting(self.gd1, n1_sg)
            elif self.mode.lower() == "function":
                norm_s = self.gd1.integrate(n1_sg)
                fct = np.exp
                fn_sg = fct(n1_sg)
                self.apply_weighting(self.gd1, fn_sg)
                newnorm_s = self.gd1.integrate(fn_sg)
                n1_sg[0, :] = fn_sg[0, :] * norm_s[0]/newnorm_s[0]
            else:
                raise ValueError("WLDA mode not recognized")

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
                if self.mode.lower() == "renorm":
                    
                    n = n1_sg[0]
                    n[n < 1e-20] = 1e-40
                    rs = (C0I/n)**(1/3) * (newnorm_s[0] / norm_s[0])**(1/3)
                    ex = C1/rs
                    dexdrs = -ex/rs
                    e1_g[:] = n * ex * norm_s[0] / newnorm_s[0]
                    v1_sg[0] += ex - rs*dexdrs/3
            
            
                    zeta = 0
            
                    lda_c(0, e_g, n * norm_s[0] / newnorm_s[0], v1_sg, zeta)
                else:
                    n = n1_sg[0]
                    n[n < 1e-20] = 1e-40
                    rs = (C0I/n)**(1/3)
                    ex = C1/rs
                    dexdrs = -ex/rs
                    e1_g[:] = n*ex
                    v1_sg[0] += ex - rs*dexdrs/3
            
            
                    zeta = 0
            
                    lda_c(0, e1_g, n, v1_sg, zeta)

                if self.mode.lower() == "":
                    self.potential_correction(v1_sg, self.gd1, n1_sg)
                    
                elif self.mode.lower() == "renorm":
                    self.renorm_potential_correction(v1_sg, self.gd1, n1_sg, norm_s[0], newnorm_s[0])

        gd.distribute(v1_sg, v_sg)
        #gd.distribute(n1_sg, n_sg)        
        gd.distribute(e1_g, e_g)

    def apply_weighting(self, gd, n_sg):
        if n_sg.shape[0] > 1:
            raise NotImplementedError

        self.tabulate_weights(n_sg[0], gd)
        n_g = n_sg[0]

        take_g, weight_g = self.get_take_weight_array(n_g)
        wtable_gi = self.weight_table
        wn_g = wtable_gi.take(take_g) * weight_g + wtable_gi.take(take_g + 1) * (1 - weight_g)

        n_sg[0, :] = wn_g



    def apply_const_weighting(self, gd, n_sg):
        if n_sg.shape[0] > 1:
            raise NotImplementedError
        n_g = np.abs(n_sg[0])
        prenorm = gd.integrate(n_g)
        avg_dens = np.mean(n_g)
        effective_kF = (3*np.pi**2*avg_dens)**(1/3)
        K_G = self._get_K_G(gd)
        n_G = np.fft.fftn(n_g)
        filn_G = self.filter_kernel(effective_kF, K_G, n_G)
        filn_g = np.fft.ifftn(filn_G)
        assert np.allclose(filn_g, filn_g.real)
        postnorm = gd.integrate(filn_g)
        n_sg[0, :] = filn_g*prenorm/postnorm
         
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


    def get_take_weight_array(self, n_g):
        n_g = np.abs(n_g)
        nis = self.get_nis(n_g)
        nx, ny, nz = n_g.shape
        take_g = np.array([j * len(nis) for j in range(nx * ny * nz)]).reshape(nx, ny, nz) + (n_g[:, :, :, np.newaxis] >= nis).sum(axis=-1).astype(int) - 1

        nlesser_g = (n_g[:, :, :, np.newaxis] >= nis).sum(axis=-1)
        ngreater_g = (n_g[:, :, :, np.newaxis] < nis).sum(axis=-1)
        vallesser_g = nis.take(nlesser_g-1)
        valgreater_g = nis.take(-ngreater_g)
        weight_g = (valgreater_g-n_g)/(valgreater_g-vallesser_g)

        return take_g.astype(int), weight_g
        

    def get_ni_weights(self, n_g):
        n_g = np.abs(n_g)
        nis = self.get_nis(n_g)
        nlesser_g = (n_g[:, :, :, np.newaxis] >= nis).sum(axis=-1)
        ngreater_g = (n_g[:, :, :, np.newaxis] < nis).sum(axis=-1)
        assert (nlesser_g >= 1).all()
        assert (ngreater_g >= 0).all()

        vallesser_g = nis.take(nlesser_g-1)
        valgreater_g = nis.take(-ngreater_g)

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



    def _get_ni_weights(self, n_g):
        flatn_g = n_g.reshape(-1)

        n_gi = np.array([self._get_ni_vector(n) for n in n_g.reshape(-1)]).reshape(n_g.shape + (len(self.nis),))

#np.transpose(np.array([self._get_ni_vector(n) for n in n_g.reshape(-1)]).reshape((len(self.nis),) + n_g.shape), (1, 2, 3, 0))

#.reshape(n_g.shape + (len(self.nis),))

#.reshape((len(self.nis),) + n_g.shape), (1, 2, 3, 0))
        


        return n_gi

    def get_ni_index_weight(self, n, nis):
        n = np.abs(n)
        assert np.min(nis) <= n
        assert np.max(nis) > n
        index = (nis <= n).sum()
        assert index >= 1
        
        ngreater = (nis > n).sum()
        vallesser = nis[index - 1]
        valgreater = nis[-ngreater]
        
        weight = (valgreater-n)/(valgreater-vallesser)

        return int(index), weight

    def get_niindex_niweight(self, n_g):
        nis = self.get_nis(n_g)
        nx, ny, nz = n_g.shape
        

        # niindex_g = np.zeros(len(n_g.reshape(-1)), dtype=int)
        # niweight_g = np.zeros(len(n_g.reshape(-1)), dtype=np.float)
        # for index, n in enumerate(n_g.reshape(-1)):
        #     t = self.get_ni_index_weight(n, nis)
        #     niindex_g[index] = t[0]
        #     niweight_g[index] = t[1]
        from time import time
        t1 = time()
        ind_w_g = np.array([self.get_ni_index_weight(n, nis) for n in n_g.reshape(-1)])
        niindex_g = np.array([t[0] for t in ind_w_g], dtype=int)
        niweight_g = np.array([t[1] for t in ind_w_g])
        t2 = time()
        print("This took: {} s".format(t2 - t1))
        assert (niindex_g < len(nis) - 1).all()
        return niindex_g.reshape(nx, ny, nz), niweight_g.reshape(nx, ny, nz)
        
                    
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

        


    def _get_K_G(self, gd):
        assert gd.comm.size == 1 # Construct_reciprocal doesnt work in parallel
        k2_Q, _ = construct_reciprocal(gd)
        k2_Q[0,0,0] = 0
        return k2_Q**(1/2)
        


    def _theta_filter(self, k_F, K_G, n_G):
        
        Theta_G = (K_G**2 <= 4*k_F**2).astype(np.complex128)

        return n_G*Theta_G

    def _fermi_kinetic(self, k_F, K_G, n_G):

        rs = (9 * np.pi / 4)**(1/3) * 1 / (k_F)
        if 1/rs < 1e-3:
            return self._theta_filter(k_F, K_G, n_G)
        
        filter_G = 1 / (np.exp( (K_G**2 - 4 * k_F**2) / (1 / (2 * rs**2))) + 1)

        return n_G * filter_G

    def _fermi_coulomb(self, k_F, K_G, n_G):

        rs = (9 * np.pi / 4)**(1/3) * 1 / (k_F)
        if 1/rs < 1e-3:
            return self._theta_filter(k_F, K_G, n_G)

        filter_G = 1 / (np.exp( (K_G - 2 * k_F) / (1 / rs)) + 1)

        return n_G * filter_G

    def get_nis(self, n_g):
        n_g = np.abs(n_g)
        # maxLogVal = np.log(np.max(n_g) * 1.1 + 1)
        # deltaLog = np.log(self.stepfactor)
        # nis = np.exp(np.arange(0, maxLogVal, deltaLog)) - 1
        # assert len(nis) > 1
        # return nis
        def maxit(a, b):
            if np.isnan(a):
                return b
            if np.isnan(b):
                return a
            else:
                return max(a, b)

        nis = np.arange(0, maxit(np.max(n_g)+2*self.stepsize, 5), self.stepsize)
        stepsize = self.stepsize
        while len(nis) > 30:
            stepsize *= 1.2
            nis = np.arange(0, maxit(np.max(n_g)+2*self.stepsize, 5), stepsize)
        return nis


    def tabulate_weights(self, n_g, gd):
        '''
        Calculate \int  w(r-r', ni)n(r') dr'
        for various values of ni.

        This is later used to calculate nstar(r) = \int w(r-r', n(r))n(r')
        by interpolating the values at the ni to the value at n(r).

        '''
        n_g = np.abs(n_g)
        nis = self.get_nis(n_g)
        K_G = self._get_K_G(gd)
        self.nis = nis
        self.weight_table = np.zeros(n_g.shape+(len(nis),))
        n_G = np.fft.fftn(n_g)

        for i, ni in enumerate(nis):
            k_F = (3*np.pi**2*ni)**(1/3)
            fil_n_G = self.filter_kernel(k_F, K_G, n_G)
            x = np.fft.ifftn(fil_n_G)
            #if (n_g >= 0).all():
            #    assert np.allclose(x, x.real)

            self.weight_table[:,:,:,i] = x.real

    
    def tabulate_weights2(self, n_g, gd):
        n_g = np.abs(n_g)
        nis = self.get_nis(n_g)
        K_G = self._get_K_G(gd)
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
        #C_ng2 = np.array([[self.expand_spline(n, C_sc) for n in n_x] for C_sc in C_nsc]).reshape(len(C_nsc), *gridshape)
        # C_ng = np.array([self.expand_spline2(n_x, C_sc) for C_sc in C_nsc]).reshape(len(C_nsc), *gridshape)
        C_ng = self.expand_spline3(n_x, C_nsc).reshape(len(C_nsc), *gridshape)
        #assert np.allclose(C_ng2, C_ng)
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
        K_G = self._get_K_G(gd)
        w_alphaG = np.zeros((len(alpha_n), *K_G.shape), dtype=np.complex128)

        for ia, alpha in enumerate(alpha_n):
            k_F = self.get_kF(alpha)
            step_fct_G = (K_G**2 <= 4*k_F**2).astype(np.complex128)
            w_alphaG[ia, :, :, :] = step_fct_G
        
        # norm_alpha = np.array([gd.integrate(np.fft.ifftn(w_G)) for w_G in w_alphaG])
        # w_alphaG = np.array([w_G/norm_alpha[a] for a, w_G in enumerate(w_alphaG)])
        #norm = gd.integrate(np.fft.ifftn(w_alphaG[0]))
        #w_alphaG /= gd.dv
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



    def solve_poisson(self, n_g, gd):
        K_G = self._get_K_G(gd)
        K2_G = K_G**2
        K2_G[0,0,0] = 1.0 ##This only works if the norm of nstar is the same as the original density
        n_G = np.fft.fftn(n_g)
        
        v_G = n_G/K2_G
        
        return np.fft.ifftn(v_G)

    
    def calc_hartree_energy(self, n_g, gd):
        v_g = self.solve_poisson(n_g, gd)

        E_H = gd.integrate(n_g*v_g)/2
        
        return E_H.real

    def calc_hartree_potential_correction(self, nstar_g, n_g, gd):
        vstar_g = self.solve_poisson(nstar_g, gd)
        v_g = self.solve_poisson(n_g, gd)
        
        return (vstar_g - v_g).real

    
    def calc_hartree_energy_correction(self, nstar_g, n_g, gd):
        Estar_H = self.calc_hartree_energy(nstar_g, gd)
        E_H = self.calc_hartree_energy(n_g, gd)
        
        return Estar_H - E_H

        

    def calc_corrected_density(self, n_sg, num_l=1):
        # import matplotlib.pyplot as plt
        # before = n_sg.copy()
        # grid_vg = self.wfs.gd.get_grid_point_coordinates()

        # _, nx, ny, nz = n_sg.shape
        # plt.plot( n_sg[0, :, ny//2,nz//2].copy(), label="before")
        setups = self.wfs.setups
        # print("BEFOREDensity norm is: {}".format(self.gd1.integrate(n_sg[0])))
        for a, setup in enumerate(setups):
            spos_ac_indices = list(filter(lambda x : x[1] == setup, enumerate(setups)))
            spos_ac_indices = [x[0] for x in spos_ac_indices]
            spos_ac = self.wfs.spos_ac[spos_ac_indices]
            t = setup.calculate_pseudized_atomic_density(self.rcut, spos_ac)
            t.add(n_sg[0]) # setup.pseudized_atomic_density.add(n_sg[0], 1)

        # plt.plot(n_sg[0, :, ny//2, nz//2].copy(), label="after")
        # plt.plot(n_sg[0, :, ny//2, nz//2] - before[0, :, ny//2, nz//2], label="difference")
        # plt.legend()
        # print("AFTERDensity norm is: {}".format(self.gd1.integrate(n_sg[0])))
        # plt.show()
        # assert not np.allclose(n_sg-before, 0)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        return 0
        from gpaw.xc.lda import calculate_paw_correction
        from gpaw.xc.lda import LDARadialCalculator, LDARadialExpansion
        collinear = True
        rcalc = LDARadialCalculator(self.kernel)
        expansion = LDARadialExpansion(rcalc, collinear)
        corr = calculate_paw_correction(expansion,
                                        setup, D_sp, dEdD_sp,
                                        True, a)
        return corr
        
    def potential_correction(self, v_sg, gd, n_sg):
        _, nx, ny, nz = gd.get_grid_point_coordinates().shape
        kF_i = np.array([(3*np.pi**2*ni)**(1/3) for ni in self.get_nis(n_sg[0])])
        K_G = self._get_K_G(gd)
        # v_G = np.fft.fftn(v_sg[0])
        n_gi = self.get_ni_weights(n_sg[0]).astype(np.complex128)
        w_g = np.zeros_like(v_sg[0], dtype=np.complex128)
        for i, k_F in enumerate(kF_i):
            one_G = np.fft.fftn(v_sg[0] * n_gi[:, :, :, i])
            filv_g = np.fft.ifftn(self.filter_kernel(k_F, K_G, one_G))
            assert np.allclose(filv_g, filv_g.real), "Filtered potential was not real for kF: {}".format(k_F)
            w_g += filv_g
            
        #v_g = np.einsum("ijkl, ijkl -> ijk", n_gi, w_gi)
        assert np.allclose(w_g, w_g.real)
        v_sg[0, :] = w_g.real
        #return v_g.real

        
    def renorm_potential_correction(self, v_sg, gd, n_sg, norm, newnorm):
        _, nx, ny, nz = gd.get_grid_point_coordinates().shape
        N = 1 / newnorm - norm / newnorm**2
        kF_i = np.array([(3 * np.pi**2 * ni)**(1 / 3) for ni in self.get_nis(n_sg[0])])
        K_G = self._get_K_G(gd)
        v_G = np.fft.fftn(v_sg[0])
        n_gi = self.get_ni_weights(n_sg[0]).astype(np.complex128)
        w_g = np.zeros_like(v_sg[0], dtype=np.complex128)
        ofunc_g = np.zeros_like(v_sg[0], dtype=np.complex128)
        for i, k_F in enumerate(kF_i):
            one_G = np.fft.fftn(v_sg[0] * n_gi[:, :, :, i])
            filv_g = np.fft.ifftn(self.filter_kernel(k_F, K_G, one_G))
            assert np.allclose(filv_g, filv_g.real)
            w_g += filv_g

            fi_G = np.fft.ifftn(n_gi[:, :, :, i])
            ofunc_g += np.fft.ifftn(self.filter_kernel(k_F, K_G, fi_G))

        
        w_g = w_g * norm / newnorm +  ofunc_g * gd.integrate(n_sg[0] * v_sg[0]) * N
        assert np.allclose(w_g, w_g.real)

        v_sg[0, :] = w_g.real
        
        
    
