""" 
    Objective function class for
    Generalized Pipek-Mezey orbital localization.

    Given a spin channel index the objective function is
    (hence spin index omitted):
           __ __  
           \  \  | A  |p
    P(W) = /  /  |Q(W)|           Eq.1
           -- -- | ii |
            A  i  

    where p is a penalty degree: p>1, p<1, not p=1,
    (note that p<1 corresponds to minimization)
    and
           __
     A     \     A
    Q(W) = / W* Q  W              Eq.2
     jj    -- rj rs sj
           rs

    Note that rs run over occupied states only.

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

    All integral operations are performed on the course gd.

"""
from math import pi
from time import time

import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.dft.wannier import neighbor_k_search, calculate_weights
from ase.transport.tools import dagger
from gpaw.odd.unitary_tools import random_orthogonal
from gpaw.odd.weightfunction import WeightFunc, WignerSeitz


def md_min(func, step=.25, tolerance=1e-6, verbose=False, gd=None,
           **kwargs):
    if gd is not None:
        if gd.comm.rank == 0:
            if verbose:
                print('Localize with step =', step, 'and tolerance =',
                      tolerance)
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
                    print('MDmin: iter=%s, step=%s, value=%s' % (
                    count, step, fvalue))
    if not verbose:
        t += time()
        if gd is not None:
            if gd.comm.rank == 0:
                print(
                    '%d iterations in %0.2f seconds (%0.2f ms/iter), endstep = %s' % (
                        count, t, t * 1000. / count, step))


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


class PipekMezey:
    #
    def __init__(self, calc, method='H', ftol=1e-8,
                 penalty=2.0, uoiter=None, spin=0,
                 rnd=True, mu=None, dtype=None):
        #
        self.calc = calc
        if hasattr(self.calc, 'mode'):
            self.mode = calc.mode
        else:
            self.mode = None
        #
        self.method = method  # Charge partitioning scheme
        self.ftol = ftol  # Obj. function conv. criteria
        self.penalty = abs(penalty)  # penalty exponent
        self.mu = mu  # WF variance (if 'H')
        self.uoiter = uoiter  # noit. for UO at each step
        self.rnd = rnd  # Init. matrix with rnd or I
        #
        self.wfs = calc.wfs  # CMOs
        self.gd = calc.density.gd  # grid descriptor
        # Allow complex rotations
        if dtype != None:
            self.dtype = dtype
        else:
            self.dtype = calc.wfs.dtype
        #
        self.cmplx = self.dtype == complex
        self.setups = calc.wfs.setups
        #
        self.interpolator = calc.density.interpolator
        #
        self.atoms = calc.atoms.copy()
        self.atoms.set_constraint()
        self.Na = len(self.atoms)
        self.ns = calc.wfs.nspins
        self.spin = spin  # Which channel
        self.niter = 0
        #
        # Determine nocc: integer occupations only
        k_rank, u = divmod(0 + len(calc.wfs.kd.ibzk_kc) * spin,
                           len(calc.wfs.kpt_u))
        f_n = calc.wfs.kpt_u[u].f_n  # Error if fractional?
        self.nocc = int(np.rint(f_n.sum()) / \
                        (3 - self.ns))  # No. occ states <-- CHECK robustness
        # Poor hack in the meantime...
        # self.nocc = calc.wfs.nvalence / (3 - self.ns)
        #
        # Hold on to
        self.P = 0  # Obj. func value
        self.P_n = []  # -/- at all other steps
        self.Qa_ii = np.zeros((self.Na, self.nocc))  # Diag of PCM
        self.Qa_nn = np.zeros((self.Na, self.nocc, self.nocc))  # Norm. PCM
        self.UO_k = {}  # UO object for each kpt
        # kpts and dirs
        self.k_kc = calc.get_bz_k_points()
        #
        assert len(calc.get_ibz_k_points()) == len(self.k_kc)
        #
        self.kgd = get_monkhorst_pack_size_and_offset(self.k_kc)[0]
        self.k_kc *= -1  # Bloch phase sign conv. GPAW
        #
        # pbc-lattice
        self.Nk = len(self.k_kc)
        self.W_k = np.zeros((self.Nk, self.nocc, self.nocc),
                            dtype=self.dtype)
        # Expand cell to capture Bloch states etc.
        unitcell = calc.atoms.cell
        self.largecell = (unitcell.T * self.kgd).T
        self.wd, self.Gd = calculate_weights(self.largecell)
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
                self.invlst_dk[d, k1] = self.lst_dk[d].tolist().index(
                    k1)
        # Make atom centered weightfunctions
        self.WFa = self.gd.zeros(self.Na)
        #
        for atom in self.atoms:
            # Weight function
            if self.method == 'H':
                self.WFa[atom.index] = WeightFunc(self.gd, self.atoms,
                                                  [atom.index],
                                                  mu=self.mu).construct_weight_function()
            #
            elif self.method == 'W':
                self.WFa[atom.index] = WignerSeitz(self.gd,
                                                   self.atoms,
                                                   atom.index).construct_weight_function()
        # Using WFa and k-d lists make overlap matrix
        Qadk_nm = np.zeros(
            (self.Na, self.Nd, self.Nk, self.nocc, self.nocc),
            complex)
        #
        # If calc is a save file, read in tar references to memory
        self.wfs.initialize_wave_functions_from_restart_file()
        # IF LCAO need to make sure wave function array is available
        if self.mode == 'lcao':
            for k in range(self.Nk):
                self.wfs.kpt_u[k].psit_nG = self.gd.empty(self.nocc)
                for n in range(self.nocc):
                    self.wfs.kpt_u[k].psit_nG[n] = \
                        self.wfs._get_wave_function_array(k, n,
                                                          periodic=True)
        #
        #
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
                e_G = np.exp(
                    -2j * pi * np.dot(np.indices(self.gd.n_c).T +
                                      self.gd.beg_c,
                                      Gc / self.gd.N_c).T)
                # for each atom
                for atom in self.atoms:
                    WF = self.WFa[atom.index]
                    pw = (e_G * WF * cmo.conj()).reshape(self.nocc,
                                                         -1)
                    #
                    Qadk_nm[atom.index, d, k] += np.inner(pw,
                                                          cmo1.reshape(
                                                              self.nocc,
                                                              -1)) * \
                                                 self.gd.dv
                # PAW corrections
                P_ani1 = self.wfs.kpt_u[u1].P_ani
                # In correct cell (?) - put to large cell
                spos_ac = self.atoms.get_scaled_positions()
                #
                for A, P_ni in self.wfs.kpt_u[u].P_ani.items():
                    #
                    dS_ii = self.setups[A].dO_ii
                    P_n = P_ni[:self.nocc]
                    P_n1 = P_ani1[A][:self.nocc]
                    # Phase factor is an approx.: PRB 72, 125119 (2005)
                    e = np.exp(-2j * pi * np.dot(Gc, spos_ac[A]))
                    #
                    Qadk_nm[A, d, k] += e * P_n.conj().dot(
                        dS_ii.dot(P_n1.T))
        #
        # Sum over domains
        self.gd.comm.sum(Qadk_nm)
        self.Qadk_nm = Qadk_nm.copy()
        self.Qadk_nn = np.zeros_like(self.Qadk_nm)
        #
        # Initial W_k: Starting from a random is encouraged!
        for k in range(self.Nk):
            if rnd:
                self.W_k[k] = random_orthogonal(self.nocc,
                                                dtype=self.dtype)
            else:
                self.W_k[k] = np.eye(self.nocc, dtype=self.dtype)

        # Given all matrices, update
        self.update()
        self.initialized = True
        #

    #

    def get_cutoff_value(self, chi_i, max_d=0.85):
        # Find cutoff such that isosurface of isoval c
        # corresponds to max_d% of the orbital density.

        c = np.zeros(len(chi_i))
        dv = self.gd.dv

        # chi*chi ~ 1.0, simple 'renorm'
        for i, chi in enumerate(chi_i):
            # --> e/dV in Bohr**-3 --> e
            dens = chi.conj() * chi * dv
            rn = 1.0 / np.sum(dens)
            # Ravel and 're-norm'
            d = np.ravel(dens * rn)
            d_i = np.sort(d)[::-1]
            p = 0
            # From highest to lowest
            for n, d in enumerate(d_i):
                #
                p += d
                #
                if p >= max_d:
                    #
                    c[i] = np.sqrt((d + d_i[n + 1]) / 2)
                    break
        #
        return c
        #

    #

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
        #

    #

    def localize(self, step=0.25, tolerance=1e-8, verbose=False):
        #
        md_min(self, step, tolerance, verbose, self.gd)

    def update(self):
        for a in range(self.Na):
            #
            for d in range(self.Nd):
                #
                for k in range(self.Nk):
                    k1 = self.lst_dk[d, k]
                    self.Qadk_nn[a, d, k] = np.dot(
                        self.W_k[k].T.conj(),
                        np.dot(self.Qadk_nm[a, d, k], self.W_k[k1]))

        # Update PCM
        self.Qad_nn = self.Qadk_nn.sum(axis=2) / self.Nk

    def update_matrices(self):
        # Using new W_k rotate states
        for a in range(self.Na):
            for d in range(self.Nd):
                for k in range(self.Nk):
                    k1 = self.lst_dk[d, k]
                    self.Qadk_nn[a, d, k] = np.dot(
                        self.W_k[k].T.conj(), np.dot(
                            self.Qadk_nm[a, d, k], self.W_k[k1]))
        #

    #

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

    #

    def get_normalized_pcm(self):
        # Over k
        Qad_nn = np.sum(abs(self.Qadk_nn), axis=2) / self.Nk
        # Over d
        Qa_nn = 0
        for d in range(self.Nd):
            Qa_nn += Qad_nn[:, d] * self.wd[d] / np.sum(self.wd)
        self.Qa_nn = Qa_nn.copy()
        # over a
        for a in range(self.Na):
            self.Qa_ii[a] = Qa_nn[a].diagonal()
        #

    #

    def translate_all_to_cell(self, cell=[0, 0, 0]):
        for a in range(self.Na):
            Qd_nn = self.Qad_nn[a]
            sc_c = np.angle(Qd_nn[:3].diagonal(0, 1, 2)).T * \
                   self.kgd / (2 * pi)
            tr_c = np.array(cell)[None] - np.floor(sc_c)
            for k_c, W in zip(self.k_kc, self.W_k):
                W *= np.exp(2.j * pi * np.dot(tr_c, k_c))
        self.update()

    def get_gradients(self):
        #
        No = self.nocc
        #
        dW = []
        #
        for k in range(self.Nk):
            #
            W_k = self.W_k[k]
            Wtemp = np.zeros((No, No), complex)
            #
            for a in range(self.Na):

                for d, wd in enumerate(self.wd):
                    #
                    Qak_nm = self.Qadk_nm[a, d]
                    diagQ = self.Qad_nn[a, d].diagonal()
                    Qa_ii = np.repeat(diagQ, No).reshape(No, No)
                    k1 = self.lst_dk[d, k]
                    k2 = self.invlst_dk[d, k]
                    Qk_nn = self.Qadk_nn[a, d]

                    temp = Qa_ii.T * Qk_nn[k].conj() - Qa_ii * Qk_nn[
                        k2].conj()
                    Wtemp += wd * (temp - dagger(temp))
            #
            dW.append(Wtemp.ravel())
        #
        return np.concatenate(dW)

    #

    def get_function(self, index, repeat=None, cell=None):
        #
        # Default size of plotting cell is the one corresponding to k-points.
        if repeat is None:
            repeat = self.kgd
        N1, N2, N3 = repeat

        dim = self.calc.get_number_of_grid_points()
        largedim = dim * [N1, N2, N3]

        pmgrid = np.zeros(largedim, dtype=complex)
        for k, k_kc in enumerate(self.k_kc):
            # The coordinate vector of wannier functions
            if isinstance(index, int):
                vec_n = self.W_k[k, :, index]
            else:
                vec_n = np.dot(self.W_k[k], index)

            pm_G = np.zeros(dim, complex)
            for n, coeff in enumerate(vec_n):
                pm_G += coeff * self.calc.get_pseudo_wave_function(
                    n, k, self.spin, pad=True)

            # Distribute the small wavefunction over large cell:
            for n1 in range(N1):
                na = n1
                for n2 in range(N2):
                    nb = n2
                    for n3 in range(N3):  # sign?
                        nc = n3
                        if cell != None:
                            na, nb, nc = cell
                        e = np.exp(
                            -2.j * pi * np.dot([n1, n2, n3], k_kc))
                        pmgrid[na * dim[0]:(na + 1) * dim[0],
                        nb * dim[1]:(nb + 1) * dim[1],
                        nc * dim[2]:(nc + 1) * dim[2]] += e * pm_G

        # Normalization
        pmgrid /= np.sqrt(self.Nk)
        return pmgrid
        #

    #

    def write_cube(self, index, fname, repeat=None,
                   real=True, cell=None):
        """Dump specified PM function to a cube file"""
        from ase.io import write

        # Default size of plotting cell is the one corresponding to k-points.
        if repeat is None:
            repeat = self.kgd
        atoms = self.calc.get_atoms() * repeat
        func = self.get_function(index, repeat, cell)

        # Handle separation of complex wave into real parts
        if real:
            if self.Nk == 1:
                func *= np.exp(-1.j * np.angle(func.max()))
                if 0: assert max(abs(func.imag).flat) < 1e-4
                func = func.real
            else:
                func = (func * func.max() / (abs(func.max()))).real
        else:
            phase_fname = fname.split('.')
            phase_fname.insert(1, 'phase')
            phase_fname = '.'.join(phase_fname)
            write(phase_fname, atoms, data=np.angle(func))
            func = abs(func)
        # print(fname)
        # write_cube(fname, atoms, data=func)
        write(fname, atoms, data=func)

    # def transform_to_lcao_coefficients(self):
    #
    #
    #     for kpt in self.wfs.kpt_u:
    #         if kpt.s == self.spin:
    #             kpt.C_nM[:self.W_k.shape[1]] = \
    #                 np.dot(self.W_k[kpt.q],kpt.C_nM[:self.W_k.shape[1]])

# def min(func, tolerance=1e-6, verbose=False, gd=None, **kwargs):
#
#     t = -time()
#     fvalueold = 0.
#     fvalue = fvalueold + 10
#     count = 0
#
#     # W is unitary matrix
#
#     G = func.get_gradients()
#     A = np.zeros_like(G)
#     P = search_direction.get_search_direction(A, G)
#
#     while abs((fvalue - fvalueold) / fvalue) > tolerance:
#         # calculate step length
#         alpha = 1.0
#         # calculate rotation
#         A = -alpha * P
#         func.step(A)  # do I need to pass gradients?
#         # calculate new value of objective function
#         fvalueold = fvalue
#         fvalue = func.get_function_value()
#         # calculate gradients
#         G = func.get_gradients()
#         # calculate search direction
#         P = search_direction.get_search_direction(A, G)
#         count += 1
#         func.niter = count
#         #
#         if verbose:
#             if gd is not None:
#                 if gd.comm.rank == 0:
#                     print('MDmin: iter=%s, step=%s, value=%s' % (count, step, fvalue))
#     if not verbose:
#         t += time()
#         if gd is not None:
#             if gd.comm.rank == 0:
#                 print('%d iterations in %0.2f seconds (%0.2f ms/iter), endstep = %s' %(
#                 count, t, t * 1000. / count, step))
