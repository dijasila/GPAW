import numpy as np
from gpaw.spherical_harmonics import Y
import _gpaw

"""

===  =================================================
 M   Global localized function number.
 I   Global sphere number.
 G   Global grid point number.
 g   Local (inside sphere) grid point number.
 i   Index into list of current spheres for current G.
===  =================================================

l
m

Global grid point number (*G*) for a 7*6 grid::

   -------------
  |5 . . . . . .|
  |4 . . . . . .|
  |3 9 . . . . .|
  |2 8 . . . . .|
  |1 7 . . . . .|
  |0 6 . . . . .|
   -------------

For this example *G* runs from 0 to 41.

Here is a sphere inside the box with grid points (*g*) numbered from 0
to 7::

   -------------
  |. . . . . . .|
  |. . . . 5 . .|
  |. . . 1 4 7 .|
  |. . . 0 3 6 .|
  |. . . . 2 . .|
  |. . . . . . .|
   -------------


"""

class LF:
    def __init__(self, spline, spos_c, sdisp_c, gd, M, a, i):
        self.spline = spline
        self.l = spline.get_angular_momentum_number()
        self.M = M
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
        rcut = spline.get_cutoff()
        gd = self.gd
        dom = gd.domain
        h_cv = dom.cell_cv / gd.N_c[:, None]
        pos_v = np.dot(spos_c, dom.cell_cv)
        start_c = np.ceil(np.dot(dom.icell_cv, pos_v - rcut) *
                          gd.N_c).astype(int)
        end_c = np.ceil(np.dot(dom.icell_cv, pos_v + rcut) *
                          gd.N_c).astype(int)

        self.A_gm, self.G_B = self._spline_to_grid(spline, start_c, end_c,
                                                   pos_v, h_cv)
        
    def _spline_to_grid(self, spline, start_c, end_c, pos_v, h_cv):
        rcut = spline.get_cutoff()
        G_B = []
        A_gm = []
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
                        A_gm.append([fr * Y(self.l**2 + m, *(d_v / r))
                                     for m in range(2 * self.l + 1)])
                if gz2 is not None:
                    gz2 += 1
                    G1 = self.gd.flat_index((gx, gy, gz1))
                    G2 = self.gd.flat_index((gx, gy, gz2))
                    G_B.extend((G1, G2))
                    
        return np.array(A_gm), np.array(G_B)

    def add_squared(self, a_xG, c_xm):
        a_xG = a_xG.reshape(a_xG.shape[:-3] + (-1,))
        g = 0
        for G1, G2 in self.G_B.reshape((-1, 2)):
            for m in range(2 * self.l + 1):
                a_xG[..., G1:G2] += np.outer(c_xm[..., m],
                                             self.A_gm[g:g + G2 - G1, m]**2)
                                             
            g += G2 - G1

    def test(self):
        a = self.gd.zeros()
        b = a.ravel()
        g = 0
        for G1, G2 in self.G_B.reshape((-1, 2)):
            b[G1:G2] = self.A_gm[g:g + G2 - G1, 0]
            g += G2 - G1

        import pylab as plt
        plt.contourf(a[:, 15, :])
        plt.show()


class CLF(LF):
    def _spline_to_grid(self, spline, start_c, end_c, pos_v, h_cv):
        return _gpaw.spline_to_grid(spline.spline, start_c, end_c, pos_v, h_cv,
                                    self.gd.n_c, self.gd.beg_c)

LF = CLF

class LocalizedFunctionsCollection:
    def __init__(self, gd, lfs):
        self.gd = gd
        self.lfs = lfs

    def update(self):
        nB = sum([len(lf.G_B) for lf in self.lfs])

        nI = len(self.lfs)
        self.M_I = np.empty(nI, np.intc)
        
        G_B = np.empty(nB, np.intc)
        I_B = np.empty(nB, np.intc)
        B1 = 0
        for I, lf in enumerate(self.lfs):
            B2 = B1 + len(lf.G_B)
            G_B[B1:B2] = lf.G_B.ravel()
            I_B[B1:B2:2] = I
            I_B[B1 + 1:B2 + 1:2] = -I - 1
            self.M_I[I] = lf.M
            B1 = B2

        assert B1 == nB
        
        indices = np.argsort(G_B)
        self.G_B = G_B[indices]
        self.I_B = I_B[indices]

        ngmax = (self.G_B[1:] - self.G_B[:-1]).max() # XXX vacuum!!
        lmax = 2
        self.A_gm = np.empty((ngmax, 2 * lmax + 1))

        nimax = np.add.accumulate((self.I_B >= 0) * 2 - 1).max()
        self.I_i = np.empty(nimax, np.intc)
        self.g_I = np.empty(len(self.lfs), np.intc)
        self.i_I = np.empty(len(self.lfs), np.intc)

    def update_positions(self, spos_ac):
        for lf in self.lfs:
            lf.update_position(spos_ac[lf.a])
        self.update()
        
    def griditer(self):
        """Iterate over grid points."""
        self.g_I[:] = 0
        self.current_lfindices = []
        G1 = 0
        for I, G in zip(self.I_B, self.G_B):
            G2 = G

            yield G1, G2
            
            self.g_I[self.current_lfindices] += G2 - G1

            if I >= 0:
                self.current_lfindices.append(I)
            else:
                self.current_lfindices.remove(-1 - I)

            G1 = G2


class BasisFunctions(LocalizedFunctionsCollection):
    def __init__(self, spline_aj, spos_ac, gd, cut=True):
        lfs = []
        M = 0
        for a, (spline_j, spos_c) in enumerate(zip(spline_aj, spos_ac)):
            i = 0
            for spline in spline_j:
                rcut = spline.get_cutoff()
                for beg_c, end_c, sdisp_c in gd.get_boxes(spos_c, rcut, cut):
                    lfs.append(LF(spline, spos_c, sdisp_c, gd, M, a, i))
                i += 2 * spline.get_angular_momentum_number() + 1
            M += i

        LocalizedFunctionsCollection.__init__(self, gd, lfs)

    def add_to_density(self, nt_sG, f_asi):
        nspins = len(nt_sG)
        nt_sG = nt_sG.reshape((nspins, -1))

        for G1, G2 in self.griditer():
            for I in self.current_lfindices:
                lf = self.lfs[I]
                a = lf.a
                i = lf.i
                A_gm = lf.A_gm[self.g_I[I]:self.g_I[I] + G2 - G1]
                for m in range(2 * lf.l + 1):
                    nt_sG[0, G1:G2] += (f_asi[a][0, i + m] *
                                        A_gm[:, m]**2)

    def construct_density(self, rho_MM, nt_G):
        """Calculate electron density from density matrix.

        rho_MM: ndarray
            Density matrix.
        nt_G: ndarray
            Pseudo electron density.
        """
        
        nt_G = nt_G.ravel()

        for G1, G2 in self.griditer():
            for I1 in self.current_lfindices:
                lf1 = self.lfs[I1]
                M1 = lf1.M
                f1_gm = lf1.A_gm[self.g_I[I1]:self.g_I[I1] + G2 - G1]
                for I2 in self.current_lfindices:
                    lf2 = self.lfs[I2]
                    M2 = lf2.M
                    f2_gm = lf2.A_gm[self.g_I[I2]:self.g_I[I2] + G2 - G1]
                    rho_mm = rho_MM[M1:M1 + 2 * lf1.l + 1,
                                    M2:M2 + 2 * lf2.l + 1]
                    for m1 in range(2 * lf1.l + 1):
                        for m2 in range(2 * lf2.l + 1):
                            nt_G[G1:G2] += (rho_mm[m1, m2] *
                                            f1_gm[:, m1] * f2_gm[:, m2])

    def calculate_potential_matrix(self, vt_sG, Vt_skMM):
        vt_sG = vt_sG.reshape((len(vt_sG), -1))
        Vt_skMM[:] = 0.0
        assert Vt_skMM.shape[:2] == (1, 1)
        dv = self.gd.dv

        for G1, G2 in self.griditer():
            for I1 in self.current_lfindices:
                lf1 = self.lfs[I1]
                M1 = lf1.M
                f1_gm = lf1.A_gm[self.g_I[I1]:self.g_I[I1] + G2 - G1]
                for I2 in self.current_lfindices:
                    lf2 = self.lfs[I2]
                    M2 = lf2.M
                    f2_gm = lf2.A_gm[self.g_I[I2]:self.g_I[I2] + G2 - G1]
                    Vt_mm = Vt_skMM[0, 0,
                                    M1:M1 + 2 * lf1.l + 1,
                                    M2:M2 + 2 * lf2.l + 1]
                    for m1 in range(2 * lf1.l + 1):
                        for m2 in range(2 * lf2.l + 1):
                            Vt_mm[m1, m2] += np.dot(vt_sG[0, G1:G2],
                                                    f1_gm[:, m1] *
                                                    f2_gm[:, m2]) * dv

    def _calculate_potential_matrix(self, vt_sG, Vt_skMM):
        A_Igm = [lf.A_gm for lf in self.lfs]
        self.g_I[:] = 0
        Vt_skMM[:] = 0.0
        _gpaw.calculate_potential_matrix(A_Igm, vt_sG,
                                         self.M_I, self.G_B, self.I_B,
                                         self.g_I, self.I_i, self.i_I,
                                         self.gd.dv,
                                         self.A_gm, Vt_skMM)


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
