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
        self.nalpha = 5
        self.alpha_n = None
        self.mode = mode
        self.lmax = 1
        self.n_gi = None

        self.gd1 = None

        # self.pot_plotter = Plotter("potential" + mode + filter_kernel, "")
        # self.dens_plotter = Plotter("density" + mode + filter_kernel, "")
        # self.inputdens_plotter = Plotter("inputdensity" + mode + filter_kernel, "")
        # self.corrdens_plotter = Plotter("corrdensity" + mode + filter_kernel, "")

        if filter_kernel.lower() == "fermikinetic":
            self.filter_kernel = self._fermi_kinetic
        elif filter_kernel.lower() == "fermicoulomb":
            self.filter_kernel = self._fermi_coulomb
        else:
            filter_kernel = "step function filter"
            self.filter_kernel = self._theta_filter

        self.num_nis = 10
        self.lda_kernel = PurePythonKernel()
        
    def initialize(self, density, hamiltonian, wfs, occupations):
        self.density = density #.D_asp
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.occupations = occupations

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        if self.mode.lower() == "parallel":
            self.parallel_calc_impl(gd, n_sg, v_sg, e_g)
            return
        if self.mode.lower() == "renorm":
            self.renorm_mode(gd, n_sg, v_sg, e_g)
            return
        if self.mode.lower() == "normal":
            self.normal_mode(gd, n_sg, v_sg, e_g)
            return
        
        assert len(n_sg) == 1
        if self.gd1 is None:
            self.gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        self.alpha_n = None #Reset alpha grid for each calc

        n1_sg = gd.collect(n_sg)
        v1_sg = gd.collect(v_sg)
        e1_g = gd.collect(e_g)
        if gd.comm.rank == 0:
            _, nx, ny, nz = n1_sg.shape
            pre = n1_sg.copy()
            # self.inputdens_plotter.plot(n1_sg[0, :, ny//2, nz//2])
            self.calc_corrected_density(n1_sg)
            # self.corrdens_plotter.plot(n1_sg[0, :, ny//2, nz//2])
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
            elif self.mode.lower() == "new" or self.mode.lower() == "newrenorm":
                real_n_sg = n1_sg.copy()
                self.apply_weighting(self.gd1, n1_sg)
            elif self.mode.lower() == "lda":
                from gpaw.xc.lda import lda_x, lda_c
                n = n1_sg[0]
                n[n < 1e-20] = 1e-40

                # exchange
                lda_x(0, e1_g, n, v1_sg[0])
                # correlation
                lda_c(0, e1_g, n, v1_sg[0], 0)
            else:
                raise ValueError("WLDA mode not recognized")

            from gpaw.xc.lda import lda_c, lda_x
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
                    n = n * norm_s[0] / newnorm_s[0]
                    lda_x(0, e1_g, n, v1_sg[0])

    
                    zeta = 0
                    lda_c(0, e1_g, n, v1_sg, zeta)


                    # rs = (C0I/n)**(1/3) * (newnorm_s[0] / norm_s[0])**(1/3)
                    # ex = C1/rs
                    # dexdrs = -ex/rs
                    # e1_g[:] = n * ex * norm_s[0] / newnorm_s[0]
                    # v1_sg[0] += ex - rs*dexdrs/3            
                elif self.mode.lower() == "new":
                    # Exchange 
                    n = n1_sg[0]
                    n[n < 1e-20] = 1e-40
                    rs = (C0I/n)**(1/3)
                    ex = C1 / rs
                    dexdrs = -ex / rs
                    e1_g[:] = real_n_sg[0] * ex
                    
                    # a is defined by a = de_xc / dn* * n
                    a_x = rs * dexdrs / 3 * real_n_sg[0] / n1_sg[0]
                    self.potential_correction(np.array([a_x]), self.gd1, n1_sg)
                    v1_sg[0] += ex - a_x

                    # Correlation
                    from gpaw.xc.lda import G
                    ec, decdrs_0 = G(rs ** 0.5,
                                     0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

                    e1_g[:] += real_n_sg[0] * ec
                    a_c = rs * decdrs_0 / 3. * real_n_sg[0] / n1_sg[0]
                    self.potential_correction(np.array([a_c]), self.gd1, n1_sg)
                    v1_sg[0] += ec - a_c

                elif self.mode.lower() == "newrenorm":
                    norm = self.gd1.integrate(real_n_sg[0])
                    newnorm = self.gd1.integrate(n1_sg[0])

                    # Exchange
                    n = n1_sg[0] *  norm / newnorm
                    n[n < 1e-20] = 1e-40
                    rs = (C0I/n)**(1/3)
                    ex = C1/rs
                    e1_g[:] = real_n_sg[0] * ex

                    dexdrs = -ex / rs

                    # a is defined by a = de_xc / dn* * n
                    a = rs * dexdrs / 3 * real_n_sg[0] / (n1_sg[0] * norm / newnorm)
                    self.renorm_potential_correction(np.array([a]), self.gd1, n1_sg, norm, newnorm)
                    v1_sg[0] += ex - a

                    # Correlation
                    from gpaw.xc.lda import G
                    ec, decdrs_0 = G(rs ** 0.5,
                                     0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

                    e1_g[:] += real_n_sg[0] * ec
                    a_c = rs * decdrs_0 / 3.0 * real_n_sg[0] / (n1_sg[0] * norm / newnorm)
                    self.renorm_potential_correction(np.array([a_c]), self.gd1, n1_sg, norm, newnorm)
                    v1_sg[0] += ec - a_c

                elif self.mode.lower() == "lda":
                    pass                    
                else:
                    n = n1_sg[0]
                    n[n < 1e-20] = 1e-40
                    
                    lda_x(0, e1_g, n, v1_sg)
                    
                    # rs = (C0I/n)**(1/3)
                    # ex = C1/rs
                    # dexdrs = -ex/rs
                    # e1_g[:] = n*ex
                    # v1_sg[0] += ex - rs*dexdrs/3
            
            
                    zeta = 0
            
                    lda_c(0, e1_g, n, v1_sg, zeta)

                if self.mode.lower() == "":
                    self.potential_correction(v1_sg, self.gd1, n1_sg)
                    
                elif self.mode.lower() == "renorm":
                    self.renorm_potential_correction(v1_sg, self.gd1, n1_sg, norm_s[0], newnorm_s[0])

                _, nx, ny, nz = n1_sg.shape
                # self.dens_plotter.plot(n1_sg[0, :, :, nz//2])
                # self.pot_plotter.plot(v1_sg[0, :, :, nz//2])
                
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
        n_g = n_sg[0]
        n_g[n_g < 1e-20] = 1e-40
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
        n_g[n_g < 1e-20] = 1e-40
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
        n_g[n_g < 1e-20] = 1e-40
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
        n[n < 1e-20] = 1e-40
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
        if np.allclose(k_F, 0):
            return self._theta_filter(k_F, K_G, n_G)
        rs = (9 * np.pi / 4)**(1/3) * 1 / (k_F)
        if 1/rs < 1e-3:
            return self._theta_filter(k_F, K_G, n_G)
        
        filter_G = 1 / (np.exp( (K_G**2 - 4 * k_F**2) / (1 / (2 * rs**2))) + 1)

        return n_G * filter_G

    def _fermi_coulomb(self, k_F, K_G, n_G):
        if np.allclose(k_F, 0):
            return self._theta_filter(k_F, K_G, n_G)
        rs = (9 * np.pi / 4)**(1/3) * 1 / (k_F)
        if 1/rs < 1e-3:
            return self._theta_filter(k_F, K_G, n_G)

        filter_G = 1 / (np.exp( (K_G - 2 * k_F) / (1 / rs)) + 1)

        return n_G * filter_G

    def _gaussian_filter(self, k_F, K_G, n_G):
        if np.allclose(k_F, 0):
            gauss_G = np.zeros_like(K_G).astype(np.complex128)
            assert np.allclose(K_G[0,0,0], 0)
            gauss_G[0,0,0] = 1.0
        else:
            # If the prefactor becomes too large, the positivity of the weighted
            # density is no longer conserved
            # I think this is due to the function no longer being numerically
            # close enough to a gaussian on the discreet, finite grid.
            # This leads to the fourier transform of the kernel having negative values in some
            # regions.
            # We can enforce positivity of the kernel in real space as below
            # or do something else...
            # The problem with enforcing positivity is that the analytical expression for the kernel
            # is no longer so easy to express
            # Or we can take another approach and define the kernel in realspace XX Nope.
            # This doesnt work because we will effectively get a periodic function like
            # this, and we need a gaussian

            prefactor = 1.0
            norm = 1.0
            gauss_G = (np.exp(-K_G**2 / (prefactor * k_F**2)) / norm).astype(np.complex128)
            gauss_g = np.fft.ifftn(gauss_G).real
            gauss_g[gauss_g < 1e-8] = 1e-10
            gauss_G = np.fft.fftn(gauss_g)
            # This kernel does not preserve the integral..

        res = gauss_G*n_G
        return res     

    def get_nis(self, n_g):
        n_g[n_g < 1e-20] = 1e-40
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
        n_g[n_g < 1e-20] = 1e-40
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
        n_g[n_g < 1e-20] = 1e-40
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

        n_x[n_x < 1e-20] = 1e-40
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
        x[x < 1e-20] = 1e-40
        alpha_n = self.alpha_n
        nlessers = np.array([(alpha_n <= xval).astype(int).sum() for xval in x])
        assert (nlessers >= 1).all()
        a, b, c, d = C_sc[nlessers - 1].T
        alphas = alpha_n[nlessers - 1]

        return a + b*(x-alphas) + c*(x-alphas)**2 + d*(x-alphas)**3



    def expand_spline(self, x, C_sc):
        x[x < 1e-20] = 1e-40
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
        if not hasattr(self.wfs.setups[0], "calculate_pseudized_atomic_density"):
            return
        # import matplotlib.pyplot as plt
        # before = n_sg.copy()
        # grid_vg = self.wfs.gd.get_grid_point_coordinates()

        # _, nx, ny, nz = n_sg.shape
        # plt.plot( n_sg[0, :, ny//2,nz//2].copy(), label="before")
        setups = self.wfs.setups
        # print("BEFOREDensity norm is: {}".format(self.gd1.integrate(n_sg[0])))
        dens = n_sg[0].copy()
        for a, setup in enumerate(setups):
            spos_ac_indices = list(filter(lambda x : x[1] == setup, enumerate(setups)))
            spos_ac_indices = [x[0] for x in spos_ac_indices]
            spos_ac = self.wfs.spos_ac[spos_ac_indices]
            t = setup.calculate_pseudized_atomic_density(spos_ac)
            t.add(dens) # setup.pseudized_atomic_density.add(n_sg[0], 1)
            # n_sg[0] += t
        n_sg[0] = dens


        #print(np.argmax(np.abs(awd)))
        # print(awd[:10, :10, nz//2])
        
        # plt.plot(n_sg[0, :, ny//2, nz//2].copy(), label="after")
        # plt.plot(n_sg[0, :, ny//2, nz//2] - before[0, :, ny//2, nz//2], label="difference")
        # plt.legend()
        # print("AFTERDensity norm is: {}".format(self.gd1.integrate(n_sg[0])))
        # plt.show()
        # assert not np.allclose(n_sg-before, 0)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        return 0
        
    def potential_correction(self, v_sg, gd, n_sg):
        kF_i = np.array([(3*np.pi**2*ni)**(1/3) for ni in self.get_nis(n_sg[0])])
        K_G = self._get_K_G(gd)
        n_gi = self.get_ni_weights(n_sg[0]).astype(np.complex128)
        w_g = np.zeros_like(v_sg[0], dtype=np.complex128)
        for i, k_F in enumerate(kF_i):
            one_G = np.fft.fftn(v_sg[0] * n_gi[:, :, :, i])
            filv_g = np.fft.ifftn(self.filter_kernel(k_F, K_G, one_G))
            assert np.allclose(filv_g, filv_g.real), "Filtered potential was not real for kF: {}".format(k_F)
            w_g += filv_g
            
        assert np.allclose(w_g, w_g.real)
        v_sg[0, :] = w_g.real
        
    def renorm_potential_correction(self, v_sg, gd, n_sg, norm, newnorm):
        N = 1 / newnorm - norm / newnorm**2
        kF_i = np.array([(3 * np.pi**2 * ni)**(1 / 3) for ni in self.get_nis(n_sg[0])])
        K_G = self._get_K_G(gd)
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
        
        
    
    def parallel_calc_impl(self, gd, n_sg, v_sg, e_g):
        assert len(n_sg) == 1
        import gpaw.mpi as mpi
        world = mpi.world
        self.world = world
        self.gd1 = gd.new_descriptor(comm=mpi.serial_comm)        

        n1_sg = gd.collect(n_sg, broadcast=True)

        ##PLOT
        _, nx, ny, nz = n1_sg.shape
        # if world.rank == 0:
        #     self.inputdens_plotter.plot(n1_sg[0, :, ny//2, nz//2])
        ##ENDPLOT
        
        self.calc_corrected_density(n1_sg)
        n1norm_s = self.gd1.integrate(n1_sg)

        ##PLOT
        # if world.rank == 0:
        #     self.corrdens_plotter.plot(n1_sg[0, :, ny//2, nz//2])
        ##ENDPLOT            
        
        my_nis, my_lower, my_upper = self.get_my_nis(n1_sg)

        my_f_sig = self.get_my_weights(n1_sg, my_nis, my_lower, my_upper)

        nustar_sg = self.apply_filter_kernel(n1_sg, my_nis, my_f_sig)
        
        world.sum(nustar_sg)
        nunorm_s = self.gd1.integrate(nustar_sg)

        # Calculate LDA evaluated on weighted density
        v_lda_sg = np.zeros_like(n1_sg)
        e1_g = np.zeros_like(n1_sg[0])
        v1_sg = np.zeros_like(n1_sg)
        self.do_lda(v_lda_sg, e1_g, n1_sg, nustar_sg, n1norm_s, nunorm_s)
        
        # Apply corrections
        v1_sg = self.fold_pot_with_weights(v_lda_sg, my_nis, my_f_sig, n1norm_s / nunorm_s)
        v1_sg += self.final_pot_correction(v_lda_sg, my_nis, my_f_sig, nustar_sg, n1norm_s, nunorm_s)
            
        
        world.sum(v1_sg)
        world.sum(e1_g)
        ##PLOT
        # if world.rank == 0:
        #     self.dens_plotter.plot((nustar_sg[0] *n1norm_s[0]/nunorm_s[0])[:, :, nz//2])
        #     self.pot_plotter.plot(v1_sg[0, :, :, nz//2])
        ##ENDPLOT
        gd.distribute(e1_g, e_g)
        gd.distribute(v1_sg, v_sg)

    def get_my_nis(self, full_density_sg):
        assert len(full_density_sg) == 1

        size = self.world.size
        
        num_per_bin = self.num_nis // size
        left_over = self.num_nis % size

        nis = [num_per_bin] * size
        nis[0] = num_per_bin + left_over
        assert np.sum(nis) == self.num_nis

        my_number_of_nis = int(nis[self.world.rank])

        my_start_ni_index = np.sum(nis[:self.world.rank])
        assert np.allclose(my_start_ni_index, int(my_start_ni_index))
        my_start_ni_index = int(my_start_ni_index)

        total_nis = np.linspace(0, np.max(np.abs(full_density_sg)), self.num_nis, dtype=float)
        assert (total_nis >= 0).all()

        
        lower_boundary = total_nis[my_start_ni_index - 1] if my_start_ni_index != 0 else 0
        upper_boundary = total_nis[my_start_ni_index + my_number_of_nis] if my_start_ni_index + my_number_of_nis != len(total_nis) else total_nis[-1]

        return total_nis[my_start_ni_index:my_start_ni_index + my_number_of_nis], lower_boundary, upper_boundary

    def get_my_weights(self, full_density_sg, my_nis, my_lower, my_upper):
        assert len(full_density_sg) == 1
        _, nx, ny, nz = full_density_sg.shape
        
        my_num_nis = len(my_nis)
        my_f_ig = np.zeros((my_num_nis, nx, ny, nz))

        n_xyz = full_density_sg[0]

        # TODO: Make this more efficient by using list comp/numpy stuff by flattening and reshaping array
        for ini, ni in enumerate(my_nis):
            for ix, n_yz in enumerate(n_xyz):
                for iy, n_z in enumerate(n_yz):
                    for iz, n in enumerate(n_z):
                        if n <= my_lower or n >= my_upper:
                            weight = 0
                        else:
                            nlesser = (my_nis <= n).sum()
                            ngreater = (my_nis > n).sum()
                            vallesser = my_nis[nlesser - 1]
                            valgreater = my_nis[ngreater]
                            weight = (valgreater - n) / (valgreater - vallesser)
                        my_f_ig[ini, ix, iy, iz] = weight

        return np.array([my_f_ig])

    def do_lda(self, v_lda_sg, e1_g, n1_sg, nustar_sg, n1norm_s, nunorm_s):
        from gpaw.xc.lda import lda_x, lda_c
        if self.mode.lower() == "renorm" or self.mode.lower() == "parallel":    
            n = nustar_sg[0]
            n[n < 1e-20] = 1e-40
            n = n * n1norm_s[0] / nunorm_s[0]
            lda_x(0, e1_g, n, v_lda_sg[0])
            zeta = 0
            lda_c(0, e1_g, n, v_lda_sg, zeta)
        else:
            raise NotImplementedError

    def apply_filter_kernel(self, full_density_sg, my_nis, my_f_sig):
        assert len(full_density_sg) == 1
        
        n = full_density_sg[0]
        n[n <= 1e-20] = 1e-40
        
        n_G = np.fft.fftn(n)

        shape = n_G.shape

        nustar_ig = np.zeros((len(my_nis), ) + shape)
        K_G = self._get_K_G(self.gd1)

        for ini, ni in enumerate(my_nis):
            k_F = (3*np.pi**2*ni)**(1/3)
            a = np.fft.ifftn(self.filter_kernel(k_F, K_G, n_G))
            assert np.allclose(a, a.real)
            nustar_ig[ini] += a.real

        nustar_g = np.zeros(shape)
        for ini, ni in enumerate(my_nis):
            k_F = (3*np.pi**2*ni)**(1/3)
            a = np.fft.ifftn(self.filter_kernel(k_F, K_G, n_G))
            assert np.allclose(a, a.real)
            nustar_g += a.real * my_f_sig[0, ini]
        
        result_g = np.einsum("iabc, iabc -> abc", my_f_sig[0], nustar_ig)    
        assert np.allclose(result_g, nustar_g)

        return np.array([result_g])
            
    def fold_pot_with_weights(self, v_lda_sg, my_nis, my_f_sig, norm_factor):
        assert len(v_lda_sg) == 1
        v = v_lda_sg[0]
        f_ig = my_f_sig[0]
        K_G = self._get_K_G(self.gd1)

        v_result = np.zeros_like(v)
        for ini, ni in enumerate(my_nis):
            integrand_G = np.fft.fftn(v * f_ig[ini])
            k_F = (3 * np.pi**2 * ni)**(1/3)
            q = np.fft.ifftn(self.filter_kernel(k_F, K_G, integrand_G))
            assert np.allclose(q, q.real)
            v_result += q.real

        v_result *= norm_factor * v_result

        return np.array([v_result])

    def final_pot_correction(self, v_lda_sg, my_nis, my_f_sig, nustar_sg, n1norm_s, nunorm_s):
        assert len(nustar_sg) == 1
        
        prefactor = (1 / nunorm_s[0] - n1norm_s[0] / nunorm_s[0]**2)

        energy_factor = self.gd1.integrate(v_lda_sg[0] * nustar_sg[0])
        assert energy_factor.ndim == 0

        v_g = np.zeros_like(v_lda_sg[0])
        
        K_G = self._get_K_G(self.gd1)
        for ini, ni in enumerate(my_nis):
            k_F = (3 * np.pi**2 * ni)**(1/3)
            f_G = np.fft.fftn(my_f_sig[0, ini])
            q = np.fft.ifftn(self.filter_kernel(k_F, K_G, f_G))
            assert np.allclose(q, q.real)
            v_g += q.real
        
        return np.array([v_g * prefactor * energy_factor])

                    

    
    def renorm_mode(self, gd, n_sg, v_sg, e_g):

        # Get AE density
        nae_sg = self.get_ae_density(gd, n_sg)

        # Set up and distribute n_i grid
        ni_j, ni_lower, ni_upper = self.get_ni_grid(mpi.rank, mpi.size, nae_sg)

        # Calculate f_is based on AE density
        f_isg = self.get_f_isg(ni_j, ni_lower, ni_upper, nae_sg)
        
        # Calculate w_isg = \int dr' w(r-r', n_i)n(r')
        # We need a gd with the full grid
        gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        w_isg = self.get_w_isg(ni_j, nae_sg, gd1, self.filter_kernel)

        # Calculate unnormed, weighted density
        nu_sg = self.weight_density(f_isg, w_isg)
        # Each rank has only a part of the density
        # but later each rank needs the full density, so sum it
        mpi.world.sum(nu_sg)
        nu_sg[nu_sg < 1e-20] = 1e-40

        # Norms are needed for later calculations
        weighted_norm = gd1.integrate(nu_sg)
        ae_norm = gd1.integrate(nae_sg)
        assert (weighted_norm > 0).all()
        assert (ae_norm > 0).all()
        # Calculate WLDA energy, E_WLDA = E_LDA[n^*]
        # How do we handle the fact the density is not positive definite?
        # The energy is the correct energy, hence the WLDA suffix.
        # The potential needs modification hence LDA suffix.
        EWLDA_g, vLDA_sg = self.calculate_lda_energy_and_potential(nu_sg * ae_norm / weighted_norm)

        # Calculate WLDA potential
        vWLDA_sg = self.calculate_wlda_potential(f_isg, w_isg, nu_sg, ae_norm, weighted_norm, vLDA_sg, gd1)
        mpi.world.sum(vWLDA_sg)
        
        # Put calculated quantities into arrays
        gd.distribute(vWLDA_sg, v_sg)
        gd.distribute(EWLDA_g, e_g)

    def get_ae_density(self, gd, n_sg):
        from gpaw.xc.WDAUtils import correct_density
        return correct_density(n_sg, gd, self.wfs.setups, self.wfs.spos_ac)

    def get_ni_grid(self, my_rank, world_size, n_sg):
        from gpaw.xc.WDAUtils import get_ni_grid
        return get_ni_grid(my_rank, world_size, np.max(n_sg), self.num_nis)

    def get_f_isg(self, ni_j, ni_lower, ni_upper, n_sg):
        # We want to construct an array f_isg where f_isg[a,b,c]
        # gives the weight at n_i[a] for spin b at grid point c
        # The weights are determined by linear interpolation:
        # n_i--n----n_i+1 -> f_i: 2/3, f_i+1: 1/3
                            
        if ni_upper != ni_j[-1]:
            augmented = np.hstack([ni_j, [ni_upper]])
        else:
            augmented = ni_j

        if len(n_sg) != 1:
            raise NotImplementedError
        
        f_isg = np.zeros((len(ni_j),) + n_sg.shape)
        for ix, n_yz in enumerate(n_sg[0]):
            for iy, n_z in enumerate(n_yz):
                for iz, n in enumerate(n_z):
                    if n < ni_lower or n > ni_upper:
                        weights = np.zeros(len(ni_j))
                    else:
                        nlesser = (ni_j <= n).sum()
                        weights = np.zeros(len(ni_j))
                        
                        if nlesser != 0:
                            if nlesser != len(ni_j):
                                weight_lesser = (ni_j[nlesser] - n) / (ni_j[nlesser] - ni_j[nlesser-1])
                                weights[nlesser-1] = weight_lesser
                                weights[nlesser] = 1 - weight_lesser
                            else:
                                if ni_j[-1] == ni_upper:
                                    weights[nlesser-1] = 1
                                else:
                                    weights[nlesser-1] = (ni_upper - n) / (ni_upper - ni_j[-1])
                        else:
                            if ni_j[0] != ni_lower:
                                weight_lesser = (ni_j[0] - n) / (ni_j[0] - ni_lower)
                                weights[0] = 1 - weight_lesser
                            else:
                                weights[0] = 1

                    f_isg[:, 0, ix, iy, iz] = weights
        return f_isg
        
    def get_w_isg(self, ni_j, n_sg, gd, kernel):
        if len(n_sg) != 1:
            raise NotImplementedError
        # Calculate convolution of n_sg with the kernel K(r-r; ni_j)
      
        K_G = self._get_K_G(gd)

        # Set up kernel in fourier space for each ni
        w_isg = np.zeros((len(ni_j),) + n_sg.shape)
        for j, ni in enumerate(ni_j):
            k_F = (3 * np.pi**2 * ni)**(1/3)
            n_G = np.fft.fftn(n_sg[0])
            # Calculate convolution via convolution theorem
            fn_G = kernel(k_F, K_G, n_G)
            fn_g = np.fft.ifftn(fn_G)
            assert np.allclose(fn_g, fn_g.real)
            w_isg[j, 0, ...] = fn_g.real

        # This calculation does not exactly preserve the norm.
        # Do we want to fix that manually?
        return w_isg

    def weight_density(self, f_isg, w_isg):
        nu_sg = (f_isg*w_isg).sum(axis=0)
        assert nu_sg.ndim == 4

        return nu_sg

    def calculate_lda_energy_and_potential(self, n_sg):
        assert n_sg.dtype == np.float64
        assert (n_sg >= 0).all()
        n_sg[n_sg < 1e-20] = 1e-40
        v_sg = np.zeros_like(n_sg).astype(np.float64)
        e_g = np.zeros_like(n_sg[0]).astype(np.float64)
        if len(n_sg) == 1:
            lda_x(0, e_g, n_sg[0], v_sg[0])
            lda_c(0, e_g, n_sg[0], v_sg[0], 0)
        else:
            raise NotImplementedError
        e_g[np.isnan(e_g)] = 0
        v_sg[np.isnan(v_sg)] = 0
        assert np.allclose(e_g, e_g.real)
        assert not np.isnan(v_sg).any()
        assert np.allclose(v_sg, v_sg.real), "Mean abs imag: {}. Max val n_sg: {}".format(np.mean(np.abs(v_sg.imag)), np.max(np.abs(n_sg)))
        return e_g, v_sg
        
    def calculate_wlda_potential(self, f_isg, w_isg, nu_sg, ae_norm, weighted_norm, vLDA_sg, gd):
        vWLDA_sg = np.zeros_like(vLDA_sg)
        energy_factor = gd.integrate(nu_sg * vLDA_sg)
        assert energy_factor.ndim == 0 or energy_factor.ndim == 1, "Ndim was: {}".format(energy_factor.ndim)
        pre_factor = (1 / weighted_norm - ae_norm / weighted_norm**2)
        assert pre_factor.ndim == 0 or pre_factor.ndim == 1, pre_factor.ndim
        w1_sg = self.calculate_integral_with_w_isg(f_isg, w_isg)
        w2_sg = self.calculate_integral_with_F_sg_f_isg_w_isg(vLDA_sg, f_isg, w_isg)
        assert w1_sg.shape == w2_sg.shape


        vWLDA_sg = ae_norm / weighted_norm * w2_sg + pre_factor * energy_factor * w1_sg
        return vWLDA_sg

    def calculate_integral_with_w_isg(self, f_isg, w_isg):
        F_isG = np.fft.fftn(f_isg, axes=(2,3,4))
        W_isG = np.fft.fftn(w_isg, axes=(2,3,4))
        res_isg = np.fft.ifftn(W_isG*F_isG, axes=(2,3,4))
        # assert np.allclose(res_isg, res_isg.real)
        return res_isg.sum(axis=0).real.copy()

    def calculate_integral_with_F_sg_f_isg_w_isg(self, F_sg, f_isg, w_isg):
        F_isg = f_isg * F_sg[np.newaxis, ...]
        F_isG = np.fft.fftn(F_isg, axes=(2,3,4))
        w_isG = np.fft.fftn(w_isg, axes=(2,3,4))
        
        res_isg = np.fft.ifftn(w_isG*F_isG, axes=(2,3,4))
        # assert np.allclose(res_isg, res_isg.real)
        return res_isg.sum(axis=0).real.copy()



    def normal_mode(self, gd, n_sg, v_sg, e_g):

        nae_sg = self.get_ae_density(gd, n_sg)
        
        ni_j, ni_lower, ni_upper = self.get_ni_grid(mpi.rank, mpi.size, nae_sg)

        f_isg = self.get_f_isg(ni_j, ni_lower, ni_upper, nae_sg)

        gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        w_isg = self.get_w_isg(ni_j, nae_sg, gd1, self.filter_kernel)

        nu_sg = self.weight_density(f_isg, w_isg)
        mpi.world.sum(nu_sg)

        nu_sg[nu_sg < 1e-20] = 1e-40

        EWLDA_g, vLDA_sg = self.calculate_lda_energy_and_potential(nu_sg)

        vWLDA_sg = self.calculate_unnormed_wlda_pot(f_isg, w_isg, vLDA_sg)
        mpi.world.sum(vWLDA_sg)

        gd.distribute(vWLDA_sg, v_sg)
        gd.distribute(EWLDA_g, e_g)

    def calculate_unnormed_wlda_pot(self, f_isg, w_isg, vLDA_sg):
        # Fold f_isg * vLDA_sg with w_isg
        # Use convolution theorem

        result_sg = np.zeros(f_isg.shape[1:])

        assert np.allclose(f_isg, f_isg.real)
        assert np.allclose(w_isg, w_isg.real)
        assert np.allclose(vLDA_sg, vLDA_sg.real)

        for i, f_sg in enumerate(f_isg):
            F1_sG = np.fft.fftn(f_sg * vLDA_sg, axes=(1,2,3))
            F2_sG = np.fft.fftn(w_isg[i], axes=(1,2,3))
            
            result_sg += np.fft.ifftn(F1_sG * F2_sG, axes=(1,2,3)).real

        return result_sg

    def alt_method(self, gd, n_sg, v_sg, e_g):
        self.nindicators = int(1e4)
        self.setup_indicator_grid()

        my_alpha_indices = self.distribute_alphas()
        
        # 0. Collect density, and get grid_descriptor appropriate for collected density
        wn_sg = gd.collect(n_sg)
        gd1 = gd.new_descriptor(comm=mpi.serial_comm)

        # 1. Correct density
        wn_sg = wn_sg # Or correct via self.get_ae_density(gd, n_sg)

        # 2. calculate weighted density
        # This contains contributions for the alphas at this rank, i.e. we need a world.sum to get all contributions
        nstar_sg = self.alt_weight(wn_sg, my_alpha_indices, gd1)
        mpi.world.sum(nstar_sg)

        # 3. Calculate LDA energy 
        e1_g, v1_sg = self.calculate_lda(wn_sg, nstar_sg)
        mpi.world.sum(e1_g)
        mpi.world.sum(v1_sg)

        # 4. Correct potential
        v2_sg = self.correct_potential(nstar_sg, v1_sg)
        mpi.world.sum(v2_sg)
        
        gd.distribute(e1_g, e_g)
        gd.distribute(v2_sg, v_sg)
        
        # Done

    def distribute_alphas(self):
        rank = mpi.rank
        size = mpi.size
        
        nalphas = self.nindicators // size
        nalphas0 = nalphas + (self.nindicators - nalphas * size)
        assert (nalphas * (size - 1) + nalphas0 == self.nindicators)

        if rank == 0:
            start = 0
            end = nalphas0
        else:
            start = nalphas0 + (rank - 1) * nalphas
            end = start + nalphas

        return range(start, end)

    def setup_indicator_grid(self):
        self.alphas = np.linspace(0, 10, self.nindicators)

    def alt_weight(self, wn_sg, my_alpha_indices, gd):
        nstar_sg = np.zeros_like(wn_sg)
        
        for ia in my_alpha_indices:
            nstar_sg += self.apply_kernel(wn_sg, ia, gd)
            
        return nstar_sg
        
    def apply_kernel(self, wn_sg, ia, gd):
        f_sg = self.get_indicator(wn_sg, ia) * wn_sg
        
        f_sG = np.fft.fftn(f_sg, axes(1, 2, 3))

        w_sG = self.get_weight_function(ia, gd)
        
        r_sg = np.fft.ifftn(w_sG * f_sG)

        assert np.allclose(r_sg, r_sg.real)

        return r_sg.real

    def get_weight_function(self, ia, gd):
        alpha = self.alphas[ia]

        kF = (3 * np.pi**2 * alpha)**(1 / 3)

        K_G = self._get_K_G(gd)

        return (K_G**2 <= 4 * kF**2).astype(np.complex128)

    def calculate_lda(self, wn_sg, nstar_sg):
        raise NotImplementedError

    def correct_potential(self, nstar_sg, v1_sg):
        raise NotImplementedError
        
        
