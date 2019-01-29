from __future__ import print_function
""" Maximally localized Wannier Functions

    Find the set of maximally localized Wannier functions
    using the spread functional of Marzari and Vanderbilt
    (PRB 56, 1997 page 12847).
"""
from time import time
from math import sqrt, pi
from pickle import dump, load

import numpy as np

from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize

dag = dagger


def gram_schmidt(U):
    """Orthonormalize columns of U according to the Gram-Schmidt procedure."""
    for i, col in enumerate(U.T):
        for col2 in U.T[:i]:
            col -= col2 * np.dot(col2.conj(), col)
        col /= np.linalg.norm(col)


def gram_schmidt_single(U, n):
    """Orthogonalize columns of U to column n"""
    N = len(U.T)
    v_n = U.T[n]
    indices = list(range(N))
    del indices[indices.index(n)]
    for i in indices:
        v_i = U.T[i]
        v_i -=  v_n * np.dot(v_n.conj(), v_i)


def lowdin(U, S=None):
    """Orthonormalize columns of U according to the Lowdin procedure.

    If the overlap matrix is know, it can be specified in S.
    """
    if S is None:
        S = np.dot(dag(U), U)
    eig, rot = np.linalg.eigh(S)
    rot = np.dot(rot / np.sqrt(eig), dag(rot))
    U[:] = np.dot(U, rot)


def neighbor_k_search(k_c, G_c, kpt_kc, tol=1e-4):
    # search for k1 (in kpt_kc) and k0 (in alldir), such that
    # k1 - k - G + k0 = 0
    alldir_dc = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                           [1,1,0],[1,0,1],[0,1,1]], int)
    for k0_c in alldir_dc:
        for k1, k1_c in enumerate(kpt_kc):
            if np.linalg.norm(k1_c - k_c - G_c + k0_c) < tol:
                return k1, k0_c

    print('Wannier: Did not find matching kpoint for kpt=', k_c)
    print('Probably non-uniform k-point grid')
    raise NotImplementedError


def calculate_weights(cell_cc):
    """ Weights are used for non-cubic cells, see PRB **61**, 10040"""
    alldirs_dc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int)
    g = np.dot(cell_cc, cell_cc.T)
    # NOTE: Only first 3 of following 6 weights are presently used:
    w = np.zeros(6)
    w[0] = g[0, 0] - g[0, 1] - g[0, 2]
    w[1] = g[1, 1] - g[0, 1] - g[1, 2]
    w[2] = g[2, 2] - g[0, 2] - g[1, 2]
    w[3] = g[0, 1]
    w[4] = g[0, 2]
    w[5] = g[1, 2]
    # Make sure that first 3 Gdir vectors are included -
    # these are used to calculate Wanniercenters.
    Gdir_dc = alldirs_dc[:3]
    weight_d = w[:3]
    for d in range(3, 6):
        if abs(w[d]) > 1e-5:
            Gdir_dc = np.concatenate((Gdir_dc, alldirs_dc[d:d + 1]))
            weight_d = np.concatenate((weight_d, w[d:d + 1]))
    weight_d /= max(abs(weight_d))
    return weight_d, Gdir_dc


def random_orthogonal_matrix(dim, seed=None, real=False):
    """Generate a random orthogonal matrix"""
    if seed is not None:
        np.random.seed(seed)

    H = np.random.rand(dim, dim)
    np.add(dag(H), H, H)
    np.multiply(.5, H, H)

    if real:
        gram_schmidt(H)
        return H
    else:
        val, vec = np.linalg.eig(H)
        return np.dot(vec * np.exp(1.j * val), dag(vec))


def steepest_descent(func, step=.005, tolerance=1e-6, **kwargs):
    fvalueold = 0.
    fvalue = fvalueold + 10
    count=0
    while abs((fvalue - fvalueold) / fvalue) > tolerance:
        fvalueold = fvalue
        dF = func.get_gradients()
        func.step(dF * step, **kwargs)
        fvalue = func.get_functional_value()
        count += 1
        print('SteepestDescent: iter=%s, value=%s' % (count, fvalue))


def md_min(func, step=.25, tolerance=1e-6, verbose=False, **kwargs):
    if verbose:
        print('Localize with step =', step, 'and tolerance =', tolerance)
        t = -time()
    fvalueold = 0.
    fvalue = fvalueold + 10
    count = 0
    V = np.zeros(func.get_gradients().shape, dtype=complex)
    while abs((fvalue - fvalueold) / fvalue) > tolerance:
        fvalueold = fvalue
        dF = func.get_gradients()
        V *= (dF * V.conj()).real > 0
        V += step * dF
        func.step(V, **kwargs)
        fvalue = func.get_functional_value()
        if fvalue < fvalueold:
            step *= 0.5
        count += 1
        if verbose:
            print('MDmin: iter=%s, step=%s, value=%s' % (count, step, fvalue))
    if verbose:
        t += time()
        print('%d iterations in %0.2f seconds (%0.2f ms/iter), endstep = %s' %(
            count, t, t * 1000. / count, step))


def rotation_from_projection2(proj_nw, fixed):
    V_ni = proj_nw
    Nb, Nw = proj_nw.shape
    M = fixed
    L = Nw - M
    print('M=%i, L=%i, Nb=%i, Nw=%i' % (M, L, Nb, Nw))
    U_ww = np.zeros((Nw, Nw), dtype=proj_nw.dtype)
    c_ul = np.zeros((Nb-M, L), dtype=proj_nw.dtype)
    for V_n in V_ni.T:
        V_n /= np.linalg.norm(V_n)

    # Find EDF
    P_ui = V_ni[M:].copy()
    la = np.linalg
    for l in range(L):
        norm_list = np.array([la.norm(v) for v in P_ui.T])
        perm_list = np.argsort(-norm_list)
        P_ui = P_ui[:, perm_list].copy()    # largest norm to the left
        P_ui[:, 0] /= la.norm(P_ui[:, 0])   # normalize
        c_ul[:, l] = P_ui[:, 0]             # save normalized EDF
        gram_schmidt_single(P_ui, 0)        # ortho remain. to this EDF
        P_ui = P_ui[:, 1:].copy()           # remove this EDF

    U_ww[:M] = V_ni[:M, :]
    U_ww[M:] = np.dot(c_ul.T.conj(), V_ni[M:])
    gram_schmidt(U_ww)
    return U_ww, c_ul


def rotation_from_projection(proj_nw, fixed, ortho=True):
    """Determine rotation and coefficient matrices from projections

    proj_nw = <psi_n|p_w>
    psi_n: eigenstates
    p_w: localized function

    Nb (n) = Number of bands
    Nw (w) = Number of wannier functions
    M  (f) = Number of fixed states
    L  (l) = Number of extra degrees of freedom
    U  (u) = Number of non-fixed states
    """

    Nb, Nw = proj_nw.shape
    M = fixed
    L = Nw - M

    U_ww = np.empty((Nw, Nw), dtype=proj_nw.dtype)
    U_ww[:M] = proj_nw[:M]

    if L > 0:
        proj_uw = proj_nw[M:]
        eig_w, C_ww = np.linalg.eigh(np.dot(dag(proj_uw), proj_uw))
        C_ul = np.dot(proj_uw, C_ww[:, np.argsort(-eig_w.real)[:L]])
        #eig_u, C_uu = np.linalg.eigh(np.dot(proj_uw, dag(proj_uw)))
        #C_ul = C_uu[:, np.argsort(-eig_u.real)[:L]]

        U_ww[M:] = np.dot(dag(C_ul), proj_uw)
    else:
        C_ul = np.empty((Nb - M, 0))

    normalize(C_ul)
    if ortho:
        lowdin(U_ww)
    else:
        normalize(U_ww)

    return U_ww, C_ul

def get_atoms_object_from_wfs(wfs):
    from ase.units import Bohr
    from ase import Atoms

    spos_ac = wfs.spos_ac
    cell_cv = wfs.gd.cell_cv
    positions =  spos_ac * cell_cv.diagonal() * Bohr

    string = ''
    for a, atoms in enumerate(wfs.setups):
        string += atoms.symbol

    atoms = Atoms(string)
    atoms.positions = positions
    atoms.cell = cell_cv * Bohr

    return atoms


class WannierLocalization:
    """Maximally localized Wannier Functions
       for n_occ only - for ODD calculations
    """

    def __init__(self, wfs,
                 calc=None,
                 spin=0,
                 initialwannier='random',
                 verbose=False):

        # Bloch phase sign convention
        sign = -1
        self.wfs = wfs
        self.gd = self.wfs.gd
        self.ns = self.wfs.nspins

        if hasattr(self.wfs, 'mode'):
            self.mode = self.wfs.mode
        else:
            self.mode = None

        if calc is not None:
            self.atoms = calc.atoms
        else:
            self.atoms = get_atoms_object_from_wfs(self.wfs)

        # Determine nocc: integer occupations only
        k_rank, u  = divmod(0 + len(self.wfs.kd.ibzk_kc) * spin,
                            len(self.wfs.kpt_u))

        f_n = self.wfs.kpt_u[u].f_n
        self.nwannier = int(np.rint(f_n.sum())/ \
                            (3 - self.ns)) # No fractional occ

        self.spin = spin
        self.verbose = verbose
        self.kpt_kc = self.wfs.kd.bzk_kc
        assert len(self.wfs.kd.ibzk_kc) == len(self.kpt_kc)

        self.kptgrid = get_monkhorst_pack_size_and_offset(self.kpt_kc)[0]
        self.kpt_kc *= sign

        self.Nk = len(self.kpt_kc)
        self.unitcell_cc = self.atoms.get_cell()
        self.largeunitcell_cc = (self.unitcell_cc.T * self.kptgrid).T
        self.weight_d, self.Gdir_dc = calculate_weights(self.largeunitcell_cc)
        self.Ndir = len(self.weight_d) # Number of directions

        # Set the list of neighboring k-points k1, and the "wrapping" k0,
        # such that k1 - k - G + k0 = 0
        #
        # Example: kpoints = (-0.375,-0.125,0.125,0.375), dir=0
        # G = [0.25,0,0]
        # k=0.375, k1= -0.375 : -0.375-0.375-0.25 => k0=[1,0,0]
        #
        # For a gamma point calculation k1 = k = 0,  k0 = [1,0,0] for dir=0
        if self.Nk == 1:
            self.kklst_dk = np.zeros((self.Ndir, 1), int)
            k0_dkc = self.Gdir_dc.reshape(-1, 1, 3)
        else:
            self.kklst_dk = np.empty((self.Ndir, self.Nk), int)
            k0_dkc = np.empty((self.Ndir, self.Nk, 3), int)

            # Distance between kpoints
            kdist_c = np.empty(3)
            for c in range(3):
                # make a sorted list of the kpoint values in this direction
                slist = np.argsort(self.kpt_kc[:, c], kind='mergesort')
                skpoints_kc = np.take(self.kpt_kc, slist, axis=0)
                kdist_c[c] = max([skpoints_kc[n + 1, c] - skpoints_kc[n, c]
                                  for n in range(self.Nk - 1)])

            for d, Gdir_c in enumerate(self.Gdir_dc):
                for k, k_c in enumerate(self.kpt_kc):
                    # setup dist vector to next kpoint
                    G_c = np.where(Gdir_c > 0, kdist_c, 0)
                    if max(G_c) < 1e-4:
                        self.kklst_dk[d, k] = k
                        k0_dkc[d, k] = Gdir_c
                    else:
                        self.kklst_dk[d, k], k0_dkc[d, k] = \
                                       neighbor_k_search(k_c, G_c, self.kpt_kc)

        # Set the inverse list of neighboring k-points
        self.invkklst_dk = np.empty((self.Ndir, self.Nk), int)
        for d in range(self.Ndir):
            for k1 in range(self.Nk):
                self.invkklst_dk[d, k1] = self.kklst_dk[d].tolist().index(k1)

        Nw = self.nwannier
        Z_dknn = np.zeros((self.Ndir, self.Nk, Nw, Nw), complex)
        self.Z_dkww = np.empty((self.Ndir, self.Nk, Nw, Nw), complex)

        if self.mode == 'lcao' and self.wfs.kpt_u[0].psit_nG is None:
            self.wfs.initialize_wave_functions_from_lcao()

        for d, dirG in enumerate(self.Gdir_dc):
            for k in range(self.Nk):
                k1 = self.kklst_dk[d, k]
                k0_c = k0_dkc[d, k]
                k_kc = self.wfs.kd.bzk_kc
                Gc = k_kc[k1] - k_kc[k] - k0_c
                # Det. kpt/spin
                kr, u = divmod(k + len(self.wfs.kd.ibzk_kc) * spin,
                               len(self.wfs.kpt_u))
                kr1, u1 = divmod(k1 + len(self.wfs.kd.ibzk_kc) * spin,
                                 len(self.wfs.kpt_u))

                cmo = self.wfs.kpt_u[u].psit_nG[:Nw]
                cmo1 = self.wfs.kpt_u[u1].psit_nG[:Nw]

                #
                e_G = np.exp(-2.j * pi * 
                             np.dot(np.indices(self.gd.n_c).T + 
                             self.gd.beg_c, Gc / self.gd.N_c).T)
                pw = (e_G * cmo.conj()).reshape((Nw, -1))

                Z_dknn[d, k] += np.inner(pw,
                                         cmo1.reshape((Nw,
                                         -1))) * self.gd.dv
                # PAW corrections
                P_ani1 = self.wfs.kpt_u[u1].P_ani
                spos_ac = self.atoms.get_scaled_positions()

                for A, P_ni in self.wfs.kpt_u[u].P_ani.items():
                    dS_ii = self.wfs.setups[A].dO_ii
                    P_n  = P_ni[:Nw]
                    P_n1 = P_ani1[A][:Nw]
                    e = np.exp(-2.j * pi * np.dot(Gc, spos_ac[A]))
                    Z_dknn[d, k] += e * P_n.conj().dot(dS_ii.dot(P_n1.T))

        self.gd.comm.sum(Z_dknn)
        self.Z_dknn = Z_dknn.copy()

        self.initialize(initialwannier=initialwannier)

    def initialize(self, initialwannier='random'):
        """Re-initialize current rotation matrix.

        Keywords are identical to those of the constructor.
        """
        Nw = self.nwannier

        # Set U to random (orthogonal) matrix
        self.U_kww = np.zeros((self.Nk, Nw, Nw), complex)
        
        #for k in range(self.Nk):
        self.U_kww[:] = random_orthogonal_matrix(Nw, None, real=False)

        self.update()

    def update(self):

        # Calculate the Zk matrix from the rotation matrix:
        # Zk = U^d[k] Zbloch U[k1]
        for d in range(self.Ndir):
            for k in range(self.Nk):
                k1 = self.kklst_dk[d, k]
                self.Z_dkww[d, k] = np.dot(dag(self.U_kww[k]), np.dot(
                    self.Z_dknn[d, k], self.U_kww[k1]))

        # Update the new Z matrix
        self.Z_dww = self.Z_dkww.sum(axis=1) / self.Nk

    def get_centers(self, scaled=False):
        """Calculate the Wannier centers

        ::

          pos =  L / 2pi * phase(diag(Z))
        """
        coord_wc = np.angle(self.Z_dww[:3].diagonal(0, 1, 2)).T / (2 * pi) % 1
        if not scaled:
            coord_wc = np.dot(coord_wc, self.largeunitcell_cc)
        return coord_wc


    def localize(self, step=0.25, tolerance=1e-08,
                 updaterot=True):
        """Optimize rotation to give maximal localization"""
        md_min(self, step, tolerance, verbose=self.verbose,
               updaterot=updaterot)

    def get_functional_value(self):
        """Calculate the value of the spread functional.

        ::

          Tr[|ZI|^2]=sum(I)sum(n) w_i|Z_(i)_nn|^2,

        where w_i are weights."""
        a_d = np.sum(np.abs(self.Z_dww.diagonal(0, 1, 2))**2, axis=1)
        return np.dot(a_d, self.weight_d).real

    def get_gradients(self):

        Nw = self.nwannier

        dU = []
        for k in range(self.Nk):
            Utemp_ww = np.zeros((Nw, Nw), complex)

            for d, weight in enumerate(self.weight_d):
                if abs(weight) < 1.0e-6:
                    continue

                diagZ_w = self.Z_dww[d].diagonal()
                Zii_ww = np.repeat(diagZ_w, Nw).reshape(Nw, Nw)
                k1 = self.kklst_dk[d, k]
                k2 = self.invkklst_dk[d, k]
                Z_kww = self.Z_dkww[d]


                temp = Zii_ww.T * Z_kww[k].conj() - Zii_ww * Z_kww[k2].conj()
                Utemp_ww += weight * (temp - dag(temp))
            dU.append(Utemp_ww.ravel())

        return np.concatenate(dU)

    def step(self, dX, updaterot=True):
        Nw = self.nwannier
        Nk = self.Nk
        if updaterot:
            A_kww = dX[:Nk * Nw**2].reshape(Nk, Nw, Nw)
            for U, A in zip(self.U_kww, A_kww):
                H = -1.j * A.conj()
                epsilon, Z = np.linalg.eigh(H)
                # Z contains the eigenvectors as COLUMNS.
                # Since H = iA, dU = exp(-A) = exp(iH) = ZDZ^d
                dU = np.dot(Z * np.exp(1.j * epsilon), dag(Z))
                if U.dtype == float:
                    U[:] = np.dot(U, dU).real
                else:
                    U[:] = np.dot(U, dU)

        self.update()
