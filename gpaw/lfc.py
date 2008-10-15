import numpy as np
from gpaw.spherical_harmonics import Y

class LF:
    def __init__(self, spline, spos_c, sdisp_c, gd, mu, a, i):
        self.spline = spline
        self.rcut = spline.get_cutoff()
        self.l = spline.get_angular_momentum_number()
        self.mu = mu
        self.gd = gd
        self.a = a
        self.i = i

        self.spline_to_grid(spline, spos_c)

    def update_position(self, spos_c):
        # Perhaps: don't do checks here, but only call on relevant ones
        if (self.spos_c - spos_c).any():
            self.spline_to_grid(self.spline, spos_c)

    def spline_to_grid(self, spline, spos_c):
        self.spos_c = spos_c
        rcut = self.rcut
        gd = self.gd
        dom = gd.domain
        h_cv = dom.cell_cv / gd.N_c[:, None]
        pos_v = np.dot(spos_c, dom.cell_cv)
        start_c = np.ceil(np.dot(dom.icell_cv, pos_v - rcut) *
                          gd.N_c).astype(int)
        end_c = np.ceil(np.dot(dom.icell_cv, pos_v + rcut) *
                          gd.N_c).astype(int)

        G_B = []
        f_gm = []
        for gx in range(start_c[0], end_c[0]):
            for gy in range(start_c[1], end_c[1]):
                gz1 = None
                gz2 = None
                for gz in range(start_c[2], end_c[2]):
                    d_v = np.dot((gx, gy, gz), h_cv) - pos_v
                    r = (d_v**2).sum()**0.5
                    if r < rcut:
                        if gz1 is None:
                            gz1 = gz
                        gz2 = gz
                        fr = spline(r)
                        f_gm.append([fr * Y(self.l**2 + m, *(d_v / r))
                                     for m in range(2 * self.l + 1)])
                if gz2 is not None:
                    gz2 += 1
                    g1 = gz1 + gd.n_c[2] * (gy + gx * gd.n_c[1])
                    g2 = gz2 + gd.n_c[2] * (gy + gx * gd.n_c[1])
                    G_B.extend((g1, g2))
                    
        self.f_gm = np.array(f_gm)
        self.G_B = np.array(G_B)

    def add_squared(self, a_xG, c_xm):
        a_xG = a_xG.reshape(a_xG.shape[:-3] + (-1,))
        g = 0
        for G1, G2 in self.G_B.reshape((-1, 2)):
            #print G1,G2,g, self.f_gm.shape,a_xG[..., G1:G2].shape,self.f_gm[g:g + G2 - G1,0].shape,c_xm[:, 0].shape
            for m in range(2 * self.l + 1):
                a_xG[..., G1:G2] += np.outer(c_xm[..., m],
                                             self.f_gm[g:g + G2 - G1, m]**2)
                                             
            g += G2 - G1

    def test(self):
        a = self.gd.zeros()
        b = a.ravel()
        g = 0
        for G1, G2 in self.G_B.reshape((-1, 2)):
            b[G1:G2] = self.f_gm[g:g + G2 - G1, 0]
            g += G2 - G1

        import pylab as plt
        plt.contourf(a[:, 15, :])
        plt.show()


class LocalizedFunctionsCollection:
    def __init__(self, lfs):
        self.lfs = lfs

    def update(self):
        nB = sum([len(lf.G_B) for lf in self.lfs])

        G_B = np.empty(nB, int)
        lfindex_B = np.empty(nB, int)
        B1 = 0
        for i, lf in enumerate(self.lfs):
            B2 = B1 + len(lf.G_B)
            G_B[B1:B2] = lf.G_B.ravel()
            lfindex_B[B1:B2:2] = i
            lfindex_B[B1 + 1:B2 + 1:2] = -i - 1
            B1 = B2

        assert B1 == nB
        
        indices = np.argsort(G_B)
        self.G_B = G_B[indices]
        self.lfindex_B = lfindex_B[indices]

    def update_positions(self, spos_ac):
        for lf in self.lfs:
            lf.update_position(spos_ac[lf.a])
        self.update()
        
    def griditer(self):
        """Iterate over grid points."""
        self.g_i = np.zeros(len(self.lfs), int)
        self.current_lfindices = []
        G1 = 0
        for i, G in zip(self.lfindex_B, self.G_B):
            G2 = G

            yield G1, G2
            
            self.g_i[self.current_lfindices] += G2 - G1

            if i >= 0:
                self.current_lfindices.append(i)
            else:
                self.current_lfindices.remove(-1 - i)

            G1 = G2


class BasisFunctions(LocalizedFunctionsCollection):
    def __init__(self, spline_aj, spos_ac, gd, cut=True):
        lfs = []
        mu = 0
        for a, (spline_j, spos_c) in enumerate(zip(spline_aj, spos_ac)):
            i = 0
            for spline in spline_j:
                rcut = spline.get_cutoff()
                for beg_c, end_c, sdisp_c in gd.get_boxes(spos_c, rcut, cut):
                    lfs.append(LF(spline, spos_c, sdisp_c, gd, mu, a, i))
                i += 2 * spline.get_angular_momentum_number() + 1
            mu += i

        LocalizedFunctionsCollection.__init__(self, lfs)

    def construct_density(self, rho_MM, nt_G):
        """Calculate electron density from density matrix.

        rho_MM: ndarray
            Density matrix.
        nt_G: ndarray
            Pseudo electron density.
        """
        
        nt_G = nt_G.ravel()
        nt_G[:] = 0.0

        for G1, G2 in self.griditer():
            for i1 in self.current_lfindices:
                lf1 = self.lfs[i1]
                mu1 = lf1.mu
                f1_gm = lf1.f_gm[self.g_i[i1]:self.g_i[i1] + G2 - G1]
                for i2 in self.current_lfindices:
                    lf2 = self.lfs[i2]
                    mu2 = lf2.mu
                    f2_gm = lf2.f_gm[self.g_i[i2]:self.g_i[i2] + G2 - G1]
                    rho_mm = rho_MM[mu1:mu1 + 2 * lf1.l + 1,
                                    mu2:mu2 + 2 * lf2.l + 1]
                    for m1 in range(2 * lf1.l + 1):
                        for m2 in range(2 * lf2.l + 1):
                            nt_G[G1:G2] += (rho_mm[m1, m2] *
                                            f1_gm[:, m1] * f2_gm[:, m2])

    def calculate_effective_potential_matrix(self, vt_sG, Vt_skMM):
        """Calculate electron density from density matrix.

        rho_MM: ndarray
            Density matrix.
        nt_G: ndarray
            Pseudo electron density.
        """
        
        vt_sG = nt_G.reshape((len(vt_sG), -1))
        Vt_skMM[:] = 0.0

        for G1, G2 in self.griditer():
            for i1 in current_lfindices:
                lf1 = self.lfs[i1]
                mu1 = lf1.mu
                f1_gm = lf1.f_gm[g_i[i1]:g_i[i1] + G2 - G1]
                for i2 in current_lfindices:
                    lf2 = self.lfs[i2]
                    mu2 = lf2.mu
                    f2_gm = lf2.f_gm[g_i[i2]:g_i[i2] + G2 - G1]
                    Vt_mm = Vt_MM[mu1:mu1 + 2 * lf1.l + 1,
                                  mu2:mu2 + 2 * lf2.l + 1]
                    for m1 in range(2 * lf1.l + 1):
                        for m2 in range(2 * lf2.l + 1):
                            Vt_mm[m1, m2] += np.dot(vt_sG[G1:G2],
                                                    f1_gm[:, m1] *
                                                    f2_gm[:, m2])

    def add_to_density(self, nt_sG, f_asi):
        for G1, G2 in self.griditer():
            for q in self.current_lfindices:
                lf = self.lfs[q]
                a = lf.a
                i = lf.i
                lf.add_squared(nt_sG, f_asi[a][:, i:i + 2 * lf.l + 1])

        

def test():
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.domain import Domain
    from gpaw.poisson import PoissonSolver
    import gpaw.mpi as mpi

    ngpts = 40
    h = 1.0 / ngpts
    N_c = (ngpts, ngpts, ngpts)
    a = h * ngpts
    domain = Domain((a, a, a))
    domain.set_decomposition(mpi.world, N_c=N_c)
    gd = GridDescriptor(domain, N_c)
    
    from gpaw.spline import Spline
    a = np.array([1, 0.9, 0.8, 0.0])
    s = Spline(0, 0.2, a)
    x = LF(s, (0.5, 0.45, 0.5), (0, 0, 0), gd, 0)
    y = LF(s, (0.5, 0.55, 0.5), (0, 0, 0), gd, 1)
    xy = LocalizedFunctionsCollection([x,y])
    #x.test()
    n_G = gd.zeros()
    xy.construct_density(np.array(([(1.0, 0),(0.0,-1)])), n_G)
    #xy.f(np.array(([(2.0,)])), n_G)
    import pylab as plt
    plt.contourf(n_G[20, :, :])
    plt.axis('equal')
    plt.show()
    return xy
    
    
if __name__ == '__main__':
    test()
