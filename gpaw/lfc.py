from math import pi

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

~  _  ^  ~  ~
p  v  g  n  F  
 i     L  c  M

i
d  d  d  d  d
s     s
   s     s
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



class NewLocalizedFunctionsCollection:
    def __init__(self, gd, spline_aj, cut=False):
        self.gd = gd
        self.sphere_a = [Sphere(spline_j) for spline_j in spline_aj]
        self.cut = cut
        self.ibzk_kc = None
        
    def set_k_points(self, ibzk_kc):
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
            sdisp_Wc = np.empty((nW, 3), int)
            
        B1 = 0
        W = 0
        M = 0
        for sphere in self.sphere_a:
            self.A_Wgm.extend(sphere.A_wgm)
            nw = len(sphere.M_w)
            self.M_W[W:W + nw] = M + np.array(sphere.M_w)
            if self.ibzk_kc is not None:
                sdisp_Wc[W:W + nw] = sphere.sdisp_wc
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
            self.phase_kW = np.exp(2j * pi * np.inner(self.ibzk_kc, sdisp_Wc))
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

    def x(self):
        # Find out which ranks have a piece of the
        # localized functions:
        natoms = len(self.sphere_a)
        x_a = np.zeros(natoms, bool)
        x_a[self.atom_indices] = True
        x_ra = np.empty((self.gd.comm.size, natoms), bool)
        self.gd.comm.all_gather(x_a, x_ra)
        self.ranks_a = [x_ra[:, a].nonzeros()[0] for a in self.atom_indices]
        # use get_rank_for_position .....
    
    def add(self, c_axm, a_xG, k=-1):
        dtype = a_xG.dtype
        nx = len(a_xG)
        c_xM = np.empty((nx, self.Mmax), dtype)
        requests = []
        M1 = 0
        for a in self.atom_indices:
            c_xm =  c_axm.get(a)
            M2 = M1 + self.sphere_a[a].Mmax
            if c_xm is None:
                requests.append(comm.receive(c_xM[M1:M2], r, a, False))
            else:
                for r in ranks_a[a]:
                    requests.append(comm.send(c_xm, r, a, False))

        for request in requests:
            comm.wait(request)

        self.lfc.add(c_xM, a_xG, k)

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


class BasisFunctions(NewLocalizedFunctionsCollection):
    def add_to_density(self, nt_sG, f_sM):
        for nt_G, f_M in zip(nt_sG, f_sM):
            self.lfc.construct_density1(f_M, nt_G)

    def construct_density(self, rho_MM, nt_G, k):
        self.lfc.construct_density(rho_MM, nt_G, k)

    def calculate_potential_matrix(self, vt_G, Vt_MM, k):
        """Calculate lower part of potential matrix."""
        Vt_MM[:] = 0.0
        self.lfc.calculate_potential_matrix(vt_G, Vt_MM, k)

    def lcao_to_grid(self, C_nM, psit_nG, k):
        for C_M, psit_G in zip(C_nM, psit_nG):
            self.lfc.lcao_to_grid(C_M, psit_G, k)

    # Python implementations:
    if 0:
        def add_to_density(self, nt_sG, f_sM):
            nspins = len(nt_sG)
            nt_sG = nt_sG.reshape((nspins, -1))
            for G1, G2 in self.griditer():
                for W in self.current_lfindices:
                    M = self.M_W[W]
                    A_gm = self.A_Wgm[W][self.g_W[W]:self.g_W[W] + G2 - G1]
                    nm = A_gm.shape[1]
                    nt_sG[0, G1:G2] += np.dot(A_gm**2, f_sM[0, M:M + nm])

        def construct_density(self, rho_MM, nt_G, k):
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
                        f2_gm = self.A_Wgm[W2][self.g_W[W2]:
                                               self.g_W[W2] + G2 - G1]
                        nm2 = f2_gm.shape[1]
                        rho_mm = rho_MM[M1:M1 + nm1, M2:M2 + nm2]
                        if self.ibzk_kc is not None:
                            rho_mm = (rho_mm *
                                      self.phase_kW[k, W1] *
                                      self.phase_kW[k, W2].conj()).real
                        nt_G[G1:G2] += (np.dot(f1_gm, rho_mm) * f2_gm).sum(1)

        def calculate_potential_matrix(self, vt_G, Vt_MM, k):
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
                        f2_gm = self.A_Wgm[W2][self.g_W[W2]:
                                               self.g_W[W2] + G2 - G1]
                        nm2 = f2_gm.shape[1]
                        Vt_mm = np.dot(f1_gm.T,
                                       vt_G[G1:G2, None] * f2_gm) * dv
                        if self.ibzk_kc is not None:
                            Vt_mm = (Vt_mm *
                                     self.phase_kW[k, W1].conj() *
                                     self.phase_kW[k, W2])
                        Vt_MM[M1:M1 + nm1, M2:M2 + nm2] += Vt_mm

        def lcao_to_grid(self, C_nM, psit_nG, k):
            for C_M, psit_G in zip(C_nM, psit_nG):
                self._lcao_band_to_grid(C_M, psit_G, k)

        def _lcao_band_to_grid(self, C_M, psit_G, k):
            psit_G = psit_G.ravel()
            for G1, G2 in self.griditer():
                for W in self.current_lfindices:
                    A_gm = self.A_Wgm[W][self.g_W[W]:self.g_W[W] + G2 - G1]
                    M1 = self.M_W[W]
                    M2 = M1 + A_gm.shape[1]
                    if self.ibzk_kc is None:
                        psit_G[G1:G2] += np.dot(A_gm, C_M[M1:M2])
                    else:
                        psit_G[G1:G2] += np.dot(A_gm,
                                                C_M[M1:M2] /
                                                self.phase_kW[k, W])


from gpaw.localized_functions import LocFuncs, LocFuncBroadcaster
from gpaw.mpi import run

class LocalizedFunctionsCollection:
    def __init__(self, gd, spline_aj, cut=False, forces=False, dtype=float):
        self.gd = gd
        self.spline_aj = spline_aj
        self.cut = cut
        self.forces = forces
        self.dtype = dtype
        self.spos_ac = None
        self.lfs_a = {}
        self.ibzk_kc = None
        
    def set_k_points(self, ibzk_kc):
        self.ibzk_kc = ibzk_kc
        
    def set_positions(self, spos_ac, kpt_comm=None):
        if kpt_comm is None:
            lfbc = None
        else:
            lfbc = LocFuncBroadcaster(kpt_comm)

        for a, spline_j in enumerate(self.spline_aj):
            if self.spos_ac is None or (self.spos_ac[a] != spos_ac[a]).any():
                lfs = LocFuncs(spline_j, self.gd, spos_ac[a],
                               self.dtype, self.cut, self.forces, lfbc)
                if len(lfs.box_b) > 0:
                    if self.ibzk_kc is not None:
                        lfs.set_phase_factors(self.ibzk_kc)
                    self.lfs_a[a] = lfs
                else:
                    del self.lfs_a[a]

        if kpt_comm is not None:
            lfbc.broadcast()

    def add(self, a_xG, c_axm=None, k=-1):
        if c_axm is None:
            c_xm = np.array([1.0])
            run([lfs.iadd(a_xG, c_xm, k) for lfs in self.lfs_a.values()])
        else:
            run([lfs.iadd(a_xG, c_axm[a], k) for a, lfs in self.lfs_a.items()])

    def integrate(self, a_xG, c_axm, k=-1):
        run([lfs.iintegrate(c_axm[a], a_xG, k) for lfs in self.lfs_a])


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
    x = LocalizedFunctionsCollection(gd, [[s], [s]])
    x.set_positions([(0.5, 0.45, 0.5), (0.5, 0.55, 0.5)])
    n_G = gd.zeros()
    x.add(n_G)
    #xy.f(np.array(([(2.0,)])), n_G)
    import pylab as plt
    plt.contourf(n_G[20, :, :])
    plt.axis('equal')
    plt.show()

test()
