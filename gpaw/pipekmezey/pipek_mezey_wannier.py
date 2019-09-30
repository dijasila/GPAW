"""
    Objective function class for
    Generalized Pipek-Mezey orbital localization.

    Given a spin channel index the objective function is:
           __ __
           \\ \\  | A  |p
    P(W) = /  /  |Q(W)|           Eq.1
           -- -- | ii |
            A  i

    where p is a penalty degree: p>1, p<1, not p=1,
    (note that p<1 corresponds to minimization)
    and
           __
     A     \\    A
    Q(W) = / W* Q  W              Eq.2
     jj    -- rj rs sj
           rs

    rs run over occupied states only.

     A
    Q    can be defined with two methods:
     rs

    Hirshfeld scheme: 'H'

     A    /    *   A
    Q   = | Phi(r)w(r)Phi(r) dr   Eq.4
     rs   /    r         s

          A
    with w(r) a weight function with center on atom A.
     A
    w(r) is constructed from simple and general gaussians.

    and Wigner-Seitz scheme: 'W'

     A    /    *   A
    Q   = | Phi(r)O(r)Phi(r) dr   Eq.5
     rs   /    r         s

          A              A      B
    with O(r) = 1 if |r-R |>|r-R |, 0 otherwise

    All integrals are performed over the course gd.
"""
from time import time
import numpy as np
from scipy.linalg import inv, sqrtm
from math import pi
from ase.transport.tools import dagger
from gpaw.pipekmezey.weightfunction import WeightFunc, WignerSeitz
from ase.dft.wannier import neighbor_k_search, calculate_weights
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset


# from ase.parallel import parprint


def md_min(func, step=.25, tolerance=1e-6, verbose=False, gd=None):
    if verbose:
        if gd is not None:
            if gd.comm.rank == 0:
                print('Localize with step =', step,
                      'and tolerance =', tolerance)
    t = -time()
    fvalueold = 0.
    fvalue = fvalueold + 10
    count = 0
    V = np.zeros(func.get_gradients().shape, dtype=complex)
    #
    while abs((fvalue - fvalueold) / fvalue) > tolerance:
        fvalueold = fvalue
        dF = func.get_gradients()
        V *= (dF * V.conj()).real > 0
        V += step * dF
        func.step(V)
        fvalue = func.get_function_value()
        #
        if fvalue < fvalueold:
            step *= 0.5
        count += 1
        func.niter = count
        #
        if verbose:
            if gd is not None:
                if gd.comm.rank == 0:
                    print('MDmin: iter=%s, step=%s, value=%s'
                          % (count, step, fvalue))
    t += time()
    if verbose:
        if gd is not None:
            if gd.comm.rank == 0:
                print('%d iterations in %0.2f seconds(%0.2f ms/iter),'
                      ' endstep = %s'
                      % (count, t, t * 1000. / count, step))


def get_kklists(Nk, Gd, Nd, kpt):
    #
    list_dk = np.zeros((Nd, Nk), int)
    k0_dk = np.zeros((Nd, Nk, 3), int)
    #
    kd_c = np.empty(3)
    #
    for c in range(3):
        #
        sl = np.argsort(kpt[:, c], kind='mergesort')
        sk_kc = np.take(kpt, sl, axis=0)
        kd_c[c] = max([sk_kc[n + 1, c] - sk_kc[n, c]
                       for n in range(Nk - 1)])
    #
    for d, Gdir in enumerate(Gd):
        #
        for k, k_c in enumerate(kpt):
            #
            G_c = np.where(Gdir > 0, kd_c, 0)
            if max(G_c) < 1e-4:
                list_dk[d, k] = k
                k0_dk[d, k] = Gdir
            else:
                list_dk[d, k], k0_dk[d, k] = \
                    neighbor_k_search(k_c, G_c, kpt)
    #
    return list_dk, k0_dk


def get_atoms_object_from_wfs(wfs):
    from ase.units import Bohr
    from ase import Atoms

    spos_ac = wfs.spos_ac
    cell_cv = wfs.gd.cell_cv
    positions = spos_ac * cell_cv.diagonal() * Bohr

    string = ''
    for a, atoms in enumerate(wfs.setups):
        string += atoms.symbol

    atoms = Atoms(string)
    atoms.positions = positions
    atoms.cell = cell_cv * Bohr

    return atoms


def random_orthogonal(s, dtype=float):
    # Make a random orthogonal matrix of dim s x s, 
    # such that WW* = I = W*W
    w_r = np.random.rand(s, s)
    if dtype == complex:
        w_r = w_r + 1.j * np.random.rand(s, s)
    return w_r.dot(inv(sqrtm(w_r.T.conj().dot(w_r))))


class PipekMezey:
    #
    def __init__(self, wfs=None, calc=None,
                 method='W', penalty=2.0, spin=0,
                 mu=None, dtype=None):
        #
        assert wfs or calc is not None

        if calc is not None:
            self.wfs = calc.wfs
        else:
            self.wfs = wfs  # CMOs

        if hasattr(self.wfs, 'mode'):
            self.mode = self.wfs.mode
        else:
            self.mode = None
        #
        self.method = method  # Charge partitioning scheme
        self.penalty = abs(penalty)  # penalty exponent
        self.mu = mu  # WF variance (if 'H')
        #
        self.gd = self.wfs.gd
        # Allow complex rotations
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.wfs.dtype

        self.setups = self.wfs.setups

        # Make atoms object from setups
        if calc is not None:
            self.atoms = calc.atoms
        else:
            self.atoms = get_atoms_object_from_wfs(self.wfs)

        self.Na = len(self.atoms)
        self.ns = self.wfs.nspins
        self.spin = spin
        self.niter = 0

        # Determine nocc: integer occupations only
        k_rank, u = divmod(0 + len(self.wfs.kd.ibzk_kc) * spin,
                           len(self.wfs.kpt_u))

        f_n = self.wfs.kpt_u[u].f_n
        self.nocc = int(np.rint(f_n.sum()) /
                        (3 - self.ns))  # no fractional occ

        # Hold on to
        self.P = 0
        self.P_n = []
        self.Qa_nn = np.zeros((self.Na, self.nocc, self.nocc))

        # kpts and dirs
        self.k_kc = self.wfs.kd.bzk_kc
        #
        assert len(self.wfs.kd.ibzk_kc) == len(self.k_kc)
        #
        self.kgd = get_monkhorst_pack_size_and_offset(self.k_kc)[0]
        self.k_kc *= -1  # Bloch phase sign conv. GPAW

        # pbc-lattice
        self.Nk = len(self.k_kc)
        self.W_k = np.zeros((self.Nk, self.nocc, self.nocc),
                            dtype=self.dtype)

        # Expand cell to capture Bloch states
        largecell = (self.atoms.cell.T * self.kgd).T
        self.wd, self.Gd = calculate_weights(largecell)
        self.Nd = len(self.wd)

        # Get neighbor kpt lists 
        if self.Nk == 1:
            self.lst_dk = np.zeros((self.Nd, 1), int)
            k0_dk = self.Gd.reshape(-1, 1, 3)
        else:
            self.lst_dk, k0_dk = get_kklists(self.Nk,
                                             self.Gd,
                                             self.Nd,
                                             self.k_kc)
        #
        self.invlst_dk = np.empty((self.Nd, self.Nk), int)
        for d in range(self.Nd):
            for k1 in range(self.Nk):
                self.invlst_dk[d, k1] = \
                    self.lst_dk[d].tolist().index(k1)

        # Using WFa and k-d lists make overlap matrix
        Qadk_nm = np.zeros((self.Na,
                            self.Nd,
                            self.Nk,
                            self.nocc, self.nocc), complex)

        if calc is not None and self.wfs.kpt_u[0].psit_nG is None:
            self.wfs.initialize_wave_functions_from_restart_file()

        # If LCAO need to make sure wave function array is available
        if self.mode == 'lcao' and self.wfs.kpt_u[0].psit_nG is None:
            self.wfs.initialize_wave_functions_from_lcao()

        # a = time()

        for d, dG in enumerate(self.Gd):
            #
            for k in range(self.Nk):
                #
                k1 = self.lst_dk[d, k]
                k0 = k0_dk[d, k]
                k_kc = self.wfs.kd.bzk_kc
                Gc = k_kc[k1] - k_kc[k] - k0
                # Det. kpt/spin
                kr, u = divmod(k + len(self.wfs.kd.ibzk_kc) * spin,
                               len(self.wfs.kpt_u))
                kr1, u1 = divmod(k1 + len(self.wfs.kd.ibzk_kc) * spin,
                                 len(self.wfs.kpt_u))
                #
                cmo = self.wfs.kpt_u[u].psit_nG[:self.nocc]
                cmo1 = self.wfs.kpt_u[u1].psit_nG[:self.nocc]
                # Inner product
                e_G = np.exp(-2j * pi *
                             np.dot(np.indices(self.gd.n_c).T +
                                    self.gd.beg_c,
                                    Gc / self.gd.N_c).T)
                # for each atom
                for atom in self.atoms:
                    WF = self.get_weight_function_atom(atom.index)
                    pw = (e_G * WF * cmo1)
                    Qadk_nm[atom.index, d, k] += \
                        self.gd.integrate(cmo, pw,
                                          global_integral=False)
                # PAW corrections
                P_ani1 = self.wfs.kpt_u[u1].P_ani

                spos_ac = self.atoms.get_scaled_positions()
                #
                for A, P_ni in self.wfs.kpt_u[u].P_ani.items():
                    #
                    dS_ii = self.setups[A].dO_ii
                    P_n = P_ni[:self.nocc]
                    P_n1 = P_ani1[A][:self.nocc]
                    # Phase factor is an approx. PRB 72, 125119 (2005)
                    e = np.exp(-2j * pi * np.dot(Gc, spos_ac[A]))
                    Qadk_nm[A, d, k] += \
                        e * P_n.conj().dot(dS_ii.dot(P_n1.T))
        #
        # Sum over domains
        self.gd.comm.sum(Qadk_nm)
        self.Qadk_nm = Qadk_nm.copy()
        self.Qadk_nn = np.zeros_like(self.Qadk_nm)
        #
        # b = time()
        # parprint('Time spent initializing', b - a)

        # Initial W_k: Start from random WW*=I
        for k in range(self.Nk):
            self.W_k[k] = random_orthogonal(self.nocc,
                                            dtype=self.dtype)

        # Given all matrices, update
        self.update()
        self.initialized = True

    def step(self, dX):
        No = self.nocc
        Nk = self.Nk
        #
        A_kww = dX[:Nk * No ** 2].reshape(Nk, No, No)
        for U, A in zip(self.W_k, A_kww):
            H = -1.j * A.conj()
            epsilon, Z = np.linalg.eigh(H)
            dU = np.dot(Z * np.exp(1.j * epsilon), dagger(Z))
            if U.dtype == float:
                U[:] = np.dot(U, dU).real
            else:
                U[:] = np.dot(U, dU)

        self.update()

    def get_weight_function_atom(self, index):
        if self.method == 'H':
            WFa = WeightFunc(self.gd,
                             self.atoms,
                             [index],
                             mu=self.mu
                             ).construct_weight_function()
        elif self.method == 'W':
            WFa = WignerSeitz(self.gd,
                              self.atoms,
                              index
                              ).construct_weight_function()
        else:
            raise ValueError('check method')
        return WFa

    def localize(self, step=0.25, tolerance=1e-8, verbose=False):
        md_min(self, step, tolerance, verbose, self.gd)

    def update(self):
        for a in range(self.Na):
            for d in range(self.Nd):
                for k in range(self.Nk):
                    k1 = self.lst_dk[d, k]
                    self.Qadk_nn[a, d, k] = \
                        np.dot(self.W_k[k].T.conj(),
                               np.dot(self.Qadk_nm[a, d, k],
                                      self.W_k[k1]))

        # Update PCM
        self.Qad_nn = self.Qadk_nn.sum(axis=2) / self.Nk

    def update_matrices(self):
        # Using new W_k rotate states
        for a in range(self.Na):
            for d in range(self.Nd):
                for k in range(self.Nk):
                    k1 = self.lst_dk[d, k]
                    self.Qadk_nn[a, d, k] = \
                        np.dot(self.W_k[k].T.conj(),
                               np.dot(self.Qadk_nm[a, d, k],
                                      self.W_k[k1]))

    def get_function_value(self):
        # Over k
        Qad_nn = np.sum(abs(self.Qadk_nn), axis=2) / self.Nk
        # Over d
        Qa_nn = 0
        self.P = 0
        for d in range(self.Nd):
            Qa_nn += Qad_nn[:, d] ** 2 * self.wd[d]
        # Over a and diag
        for a in range(self.Na):
            self.P += np.sum(Qa_nn[a].diagonal())
        #
        self.P /= np.sum(self.wd)
        #
        self.P_n.append(self.P)
        #
        return self.P

    def get_gradients(self):
        #
        No = self.nocc
        #
        dW = []
        #
        for k in range(self.Nk):
            #
            Wtemp = np.zeros((No, No), complex)
            #
            for a in range(self.Na):
                for d, wd in enumerate(self.wd):
                    #
                    diagQ = self.Qad_nn[a, d].diagonal()
                    Qa_ii = np.repeat(diagQ, No).reshape(No, No)
                    k2 = self.invlst_dk[d, k]
                    Qk_nn = self.Qadk_nn[a, d]
                    temp = Qa_ii.T * Qk_nn[k].conj() - \
                        Qa_ii * Qk_nn[k2].conj()

                    Wtemp += wd * (temp - dagger(temp))
            #
            dW.append(Wtemp.ravel())
        #
        return np.concatenate(dW)
