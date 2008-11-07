import numpy as np
from gpaw.spherical_harmonics import Y
import _gpaw

"""

===  =================================================
 M   Global localized function number.
 W   Global volume number.
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

class Sphere:
    def __init__(self, spline_j):
        self.spline_j = spline_j
        self.spos_c = None

    def set_position(self, spos_c, gd, cut, ibzk_kc):
        if self.spos_c is not None and not (self.spos_c - spos_c).any():
            return False

        self.A_wgm = []
        self.G_wb = []
        self.M_w = []
        if ibzk_kc is not None:
            self.sdisp_wc = []
        ng = 0
        M = 0
        for spline in self.spline_j:
            rcut = spline.get_cutoff()
            l = spline.get_angular_momentum_number()
            for beg_c, end_c, sdisp_c in gd.get_boxes(spos_c, rcut, cut):
                A_gm, G_b = self.spline_to_grid(spline, gd, beg_c, end_c,
                                                spos_c - sdisp_c)
                if len(G_b) > 0:
                    self.A_wgm.append(A_gm)
                    self.G_wb.append(G_b)
                    self.M_w.append(M)
                    if ibzk_kc is not None:
                        self.sdisp_wc.append(sdisp_c)
                    ng += A_gm.shape[0]
            M += 2 * l + 1

        if ng > 0:
            self.Mmax = M
        else:
            self.Mmax = 0
            self.A_wgm = []
            self.G_wb = []
            self.M_w = []
            if ibzk_kc is not None:
                self.sdisp_wc = []
            
        self.spos_c = spos_c
        return True

    def spline_to_grid(self, spline, gd, start_c, end_c, spos_c):
        dom = gd.domain
        h_cv = dom.cell_cv / gd.N_c[:, None]
        pos_v = np.dot(spos_c, dom.cell_cv)
        return _gpaw.spline_to_grid(spline.spline, start_c, end_c, pos_v, h_cv,
                                    gd.n_c, gd.beg_c)



class LocalizedFunctionsCollection:
    def __init__(self, gd, spline_aj, cut=False):
        self.gd = gd
        self.sphere_a = [Sphere(spline_j) for spline_j in spline_aj]
        self.cut = cut
        self.ibzk_kc = None
        
    def set_k_points(self, ibzk_kc):
        sdfgjkll
        self.ibzk_kc = ibzk_kc
        
    def set_positions(self, spos_ac):
        movement = False
        for spos_c, sphere in zip(spos_ac, self.sphere_a):
            movement |= sphere.set_position(spos_c, self.gd, self.cut,
                                            self.ibzk_kc)

        if movement:
            self._update(spos_ac)
    
    def _update(self, spos_ac):
        nB = 0
        nW = 0
        self.atom_indices = []
        for a, sphere in enumerate(self.sphere_a):
            G_wb = sphere.G_wb
            if len(G_wb) > 0:
                nB += sum([len(G_b) for G_b in G_wb])
                nW += len(G_wb)
                self.atom_indices.append(a)

        self.M_W = np.empty(nW, np.intc)
        self.G_B = np.empty(nB, np.intc)
        self.W_B = np.empty(nB, np.intc)
        self.A_Wgm = []
        if self.ibzk_kc is not None:
            sdisp_Wc = np.empty((nW, 3))
            
        B1 = 0
        W = 0
        M = 0
        for sphere in self.sphere_a:
            self.A_Wgm.extend(sphere.A_wgm)
            nw = len(sphere.M_w)
            self.M_W[W:W + nw] = M + np.array(sphere.M_w)
            if self.ibzk_kc is not None:
                self.sdisp_Wc[W:W + nw] = sphere.sdisp_wc
            for G_b in sphere.G_wb:
                B2 = B1 + len(G_b)
                self.G_B[B1:B2] = G_b
                self.W_B[B1:B2:2] = W
                self.W_B[B1 + 1:B2 + 1:2] = -W - 1
                B1 = B2
                W += 1
            M += sphere.Mmax
        self.Mmax = M
        assert B1 == nB

        if self.ibzk_kc is not None:
            self.phase_kW = np.exp(2j * pi * np.inner(ibzk_kc, sdisp_Wc))
        else:
            self.phase_kW = np.empty((0, nW), complex)
        
        indices = np.argsort(self.G_B, kind='mergesort')
        self.G_B = self.G_B[indices]
        self.W_B = self.W_B[indices]

        self.lfc = _gpaw.LFC(self.A_Wgm, self.M_W, self.G_B, self.W_B,
                             self.gd.dv, self.phase_kW)

        nimax = np.add.accumulate((self.W_B >= 0) * 2 - 1).max()
        self.W_i = np.empty(nimax, np.intc)
        self.g_W = np.empty(nW, np.intc)
        self.i_W = np.empty(nW, np.intc)

    def griditer(self):
        """Iterate over grid points."""
        self.g_W[:] = 0
        self.current_lfindices = []
        G1 = 0
        for W, G in zip(self.W_B, self.G_B):
            G2 = G

            yield G1, G2
            
            self.g_W[self.current_lfindices] += G2 - G1

            if W >= 0:
                self.current_lfindices.append(W)
            else:
                self.current_lfindices.remove(-1 - W)

            G1 = G2


class BasisFunctions(LocalizedFunctionsCollection):
    def add_to_density(self, nt_sG, f_sM):
        for nt_G, f_M in zip(nt_sG, f_sM):
            self.lfc.construct_density1(f_M, nt_G)

    def construct_density(self, rho_MM, nt_G, k):
        self.lfc.construct_density(rho_MM, nt_G, k)

    def calculate_potential_matrix(self, vt_G, Vt_MM, k):
        """Calculate lower part of potential matrix."""
        Vt_MM[:] = 0.0
        self.lfc.calculate_potential_matrix(vt_G, Vt_MM, k)

    # XXXXXXXXXXXXXXXXXXXXXX
    # Python implementations:

    def _add_to_density(self, nt_sG, f_sM):
        nspins = len(nt_sG)
        nt_sG = nt_sG.reshape((nspins, -1))

        for G1, G2 in self.griditer():
            for W in self.current_lfindices:
                M = self.M_W[W]
                A_gm = self.A_Wgm[W][self.g_W[W]:self.g_W[W] + G2 - G1]
                nm = A_gm.shape[1]
                nt_sG[0, G1:G2] += np.dot(A_gm**2, f_sM[0, M:M + nm])

    def _construct_density(self, rho_MM, nt_G, k):
        """Calculate electron density from density matrix.

        rho_MM: ndarray
            Density matrix.
        nt_G: ndarray
            Pseudo electron density.
        """
        nt_G = nt_G.ravel()

        for G1, G2 in self.griditer():
            for W1 in self.current_lfindices:
                M1 = self.M_W[W1]
                f1_gm = self.A_Wgm[W1][self.g_W[W1]:self.g_W[W1] + G2 - G1]
                nm1 = f1_gm.shape[1]
                for W2 in self.current_lfindices:
                    M2 = self.M_W[W2]
                    f2_gm = self.A_Wgm[W2][self.g_W[W2]:self.g_W[W2] + G2 - G1]
                    nm2 = f2_gm.shape[1]
                    rho_mm = rho_MM[M1:M1 + nm1, M2:M2 + nm2]
                    nt_G[G1:G2] += (np.dot(f1_gm, rho_mm) * f2_gm).sum(1)

    def _calculate_potential_matrix(self, vt_G, Vt_MM, k):
        vt_G = vt_G.ravel()
        Vt_MM[:] = 0.0
        dv = self.gd.dv

        for G1, G2 in self.griditer():
            for W1 in self.current_lfindices:
                M1 = self.M_W[W1]
                f1_gm = self.A_Wgm[W1][self.g_W[W1]:self.g_W[W1] + G2 - G1]
                nm1 = f1_gm.shape[1]
                for W2 in self.current_lfindices:
                    M2 = self.M_W[W2]
                    f2_gm = self.A_Wgm[W2][self.g_W[W2]:self.g_W[W2] + G2 - G1]
                    nm2 = f2_gm.shape[1]
                    Vt_mm = Vt_MM[M1:M1 + nm1, M2:M2 + nm2]
                    Vt_mm += np.dot(f1_gm.T,
                                    vt_G[G1:G2, None] * f2_gm) * dv
                    
    def lcao_to_grid0(self, c_M, psit_G):
        psit_G = psit_G.ravel()
        for G1, G2 in self.griditer():
            for W in self.current_lfindices:
                A_gm = self.A_Wgm[W].A_gm[self.g_W[W]:self.g_W[W] + G2 - G1]
                M1 = self.M_W[W]
                M2 = M + A_gm.shape[1]
                psit_G[G1:G2] += np.dot(A_gm, c_M[M1:M2])

    def lcao_to_grid(self, c_nM, psit_nG):
        for c_M, psit_G in zip(c_nM, psit_nG):
            self.lcao_to_grid0(c_M, psit_G)


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
    
class Sphere0(Sphere):
    def spline_to_grid(self, spline, gd, start_c, end_c, spos_c):
        dom = gd.domain
        h_cv = dom.cell_cv / gd.N_c[:, None]
        pos_v = np.dot(spos_c, dom.cell_cv)
        rcut = spline.get_cutoff()
        l = spline.get_angular_momentum_number()
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
                        A_gm.append([fr * Y(l**2 + m, *d_v)
                                     for m in range(2 * l + 1)])
                if gz2 is not None:
                    gz2 += 1
                    G1 = gd.flat_index((gx, gy, gz1))
                    G2 = gd.flat_index((gx, gy, gz2))
                    G_B.extend((G1, G2))
                    
        return np.array(A_gm), np.array(G_B)
#Sphere = Sphere0
if __name__ == '__main__':
    test()
