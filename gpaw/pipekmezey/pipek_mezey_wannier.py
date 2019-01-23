""" 
    Objective function class for
    Generalized Pipek-Mezey orbital localization.

    Given a spin channel index the objective function is:
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
#
import numpy as np
from scipy.linalg import inv, sqrtm
from math import pi
from ase.units import Bohr
from ase.transport.tools import dagger
from gpaw.pipekmezey.weightfunction import WeightFunc, WignerSeitz
from ase.dft.wannier import neighbor_k_search, calculate_weights
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset


def md_min(func, step=.25, tolerance=1e-6, 
           verbose=False, gd=None, **kwargs):
    if verbose:
        if gd is None:
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
            if gd is None:
                if gd.comm.rank == 0:
                    print('MDmin: iter=%s, step=%s, value=%s' 
                          % (count, step, fvalue))
    t += time()
    if verbose:
        if gd is None:
            if gd.comm.rank == 0:
                print('%d iterations in %0.2f seconds'+ 
                      ' (%0.2f ms/iter), endstep = %s' 
                      %(count, t, t * 1000. / count, step))


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
    positions =  spos_ac * cell_cv.diagonal() * Bohr

    string = ''
    for a, atoms in enumerate(wfs.setups):
        string += atoms.symbol

    atoms = Atoms(string)
    atoms.positions = positions
    atoms.cell = cell_cv * Bohr

    return atoms

def random_orthogonal(s, dtype = float):
    # Make a random orthogonal matrix of dim s x s, 
    # such that WW* = I = W*W
    w_r = np.random.rand(s,s)
    if dtype == complex:
        w_r = w_r + 1.j * np.random.rand(s,s)
    return w_r.dot(inv(sqrtm(w_r.T.conj().dot(w_r))))


class PipekMezey:
    #
    def __init__(self, wfs=None, calc=None, 
                 method='W', penalty=2.0, spin=0, 
                 rnd=True, mu=None, dtype=None):
        #
        assert wfs or calc is not None

        if calc is not None:
            self.wfs = calc.wfs
        else:
            self.wfs = wfs # CMOs

        if hasattr(self.wfs, 'mode'):
            self.mode = self.wfs.mode
        else:
            self.mode = None
        #
        self.method  = method # Charge partitioning scheme
        self.penalty = abs(penalty) # penalty exponent
        self.mu      = mu     # WF variance (if 'H')
        #
        self.gd      = wfs.gd
        # Allow complex rotations
        if dtype != None:
            self.dtype = dtype
        else:
            self.dtype = self.wfs.dtype
        #
        self.cmplx   = self.dtype == complex
        self.setups  = self.wfs.setups

        # Make atoms object from setups
        self.atoms = get_atoms_object_from_wfs(self.wfs)        
        self.Na    = len(self.atoms)
        self.ns    = self.wfs.nspins
        self.spin  = spin
        self.niter = 0

        # Determine nocc: integer occupations only
        k_rank, u   = divmod(0 + len(self.wfs.kd.ibzk_kc) * spin,
                             len(self.wfs.kpt_u))

        f_n = self.wfs.kpt_u[u].f_n
        self.nocc = int(np.rint(f_n.sum())/ \
                        (3 - self.ns)) # No fractional occ

        # Hold on to
        self.P   = 0
        self.P_n = [] 
        self.Qa_ii = np.zeros((self.Na, self.nocc)) # Diag
        self.Qa_nn = np.zeros((self.Na, self.nocc, self.nocc)) # Norm.

        # kpts and dirs
        self.k_kc = self.wfs.kd.bzk_kc
        #
        assert len(self.wfs.kd.ibzk_kc) == len(self.k_kc)
        #
        self.kgd   = get_monkhorst_pack_size_and_offset(self.k_kc)[0]
        self.k_kc *= -1 # Bloch phase sign conv. GPAW

        # pbc-lattice
        self.Nk = len(self.k_kc)
        self.W_k = np.zeros((self.Nk, self.nocc, self.nocc), 
                            dtype=self.dtype)

        # Expand cell to capture Bloch states etc.
        unitcell = self.atoms.cell
        self.largecell   = (unitcell.T * self.kgd).T
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
                self.invlst_dk[d, k1] = \
                     self.lst_dk[d].tolist().index(k1)

        # Make atom centered weightfunctions
        self.WFa = self.gd.zeros(self.Na)
        #
        for atom in self.atoms:
            # Weight function
            if self.method == 'H':
                self.WFa[atom.index] = WeightFunc(self.gd,
                                       self.atoms,
                                       [atom.index],
                                       mu=self.mu
                                       ).construct_weight_function()
            #
            elif self.method == 'W':
                self.WFa[atom.index] = WignerSeitz(self.gd,
                                       self.atoms,
                                       atom.index
                                       ).construct_weight_function()

        # Using WFa and k-d lists make overlap matrix
        Qadk_nm = np.zeros((self.Na,
                            self.Nd,
                            self.Nk,
                            self.nocc, self.nocc), complex)

        if calc is not None:
            self.wfs.initialize_wave_functions_from_restart_file()

        # IF LCAO need to make sure wave function array is available
        if self.mode == 'lcao' and self.wfs.kpt_u[0].psit_nG is None:
            self.wfs.initialize_wave_functions_from_lcao()

        for d, dG in enumerate(self.Gd):
            #
            for k in range(self.Nk):
                #
                k1 = self.lst_dk[d, k]
                k0 = k0_dk[d, k]
                k_kc = self.wfs.kd.bzk_kc
                Gc   = k_kc[k1] - k_kc[k] - k0 
                # Det. kpt/spin
                kr, u = divmod(k + len(self.wfs.kd.ibzk_kc) * spin,
                               len(self.wfs.kpt_u))
                kr1, u1 = divmod(k1 + len(self.wfs.kd.ibzk_kc) * spin,
                                 len(self.wfs.kpt_u))
                #
                cmo  = self.wfs.kpt_u[u].psit_nG[:self.nocc]
                cmo1 = self.wfs.kpt_u[u1].psit_nG[:self.nocc]
                # Inner product
                e_G = np.exp(-2j * pi * 
                             np.dot(np.indices(self.gd.n_c).T +
                             self.gd.beg_c, Gc / self.gd.N_c).T)
                # for each atom
                for atom in self.atoms:
                    WF = self.WFa[atom.index]
                    pw = (e_G * WF * cmo.conj()).reshape(self.nocc, -1)
                    #
                    Qadk_nm[atom.index,d,k] += np.inner(pw,
                                               cmo1.reshape(self.nocc,
                                               -1)) *\
                                               self.gd.dv
                # PAW corrections
                P_ani1 = self.wfs.kpt_u[u1].P_ani

                spos_ac = self.atoms.get_scaled_positions()
                #
                for A, P_ni in self.wfs.kpt_u[u].P_ani.items():
                    #
                    dS_ii = self.setups[A].dO_ii
                    P_n  = P_ni[:self.nocc]
                    P_n1 = P_ani1[A][:self.nocc]
                    # Phase factor is an approx.: PRB 72, 125119 (2005)
                    e = np.exp(-2j * pi * np.dot(Gc, spos_ac[A]))
                    #
                    Qadk_nm[A,d,k] += e * P_n.conj().dot(dS_ii.dot(P_n1.T))
        #
        # Sum over domains
        self.gd.comm.sum(Qadk_nm)
        self.Qadk_nm = Qadk_nm.copy()
        self.Qadk_nn = np.zeros_like(self.Qadk_nm)
        #
        # Initial W_k: Start from random WW*=I
        for k in range(self.Nk):
            self.W_k[k] = random_orthogonal(self.nocc,
                                            dtype=self.dtype)

        # Given all matrices, update
        self.update()
        self.initialized = True
        #
    #

    def get_cutoff_value(self, chi_i, max_d=0.85):
        # Find cutoff such that isosurface of isoval c
        # corresponds to max_d% of the orbital density.

        c  = np.zeros(len(chi_i))
        dv = self.gd.dv

        # chi*chi ~ 1.0, simple 'renorm'
        for i, chi in enumerate(chi_i):
            # --> e/dV in Bohr**-3 --> e
            dens = chi.conj()*chi * dv
            rn   = 1.0 / np.sum(dens)
            # Ravel and 're-norm'
            d    = np.ravel(dens*rn)
            d_i  = np.sort(d)[::-1]
            p    = 0
            # From highest to lowest
            for n, d in enumerate(d_i):
                #
                p += d
                #
                if p >= max_d:
                    #
                    c[i] = np.sqrt((d + d_i[n+1]) / 2)
                    break
        #
        return c
        #
    #

    def step(self, dX):
        No = self.nocc
        Nk = self.Nk
        #
        A_kww = dX[:Nk * No**2].reshape(Nk, No, No)
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
                    k1 = self.lst_dk[d,k]
                    self.Qadk_nn[a,d,k] = np.dot(self.W_k[k].T.conj(),
                            np.dot(self.Qadk_nm[a,d,k], self.W_k[k1]))

        # Update PCM
        self.Qad_nn = self.Qadk_nn.sum(axis=2) / self.Nk

    def update_matrices(self):
        # Using new W_k rotate states
        for a in range(self.Na):
            for d in range(self.Nd):
                for k in range(self.Nk):
                    k1 = self.lst_dk[d, k]
                    self.Qadk_nn[a,d,k] = \
                         np.dot(self.W_k[k].T.conj(), np.dot(
                         self.Qadk_nm[a,d,k], self.W_k[k1]))
        #
    #

    def get_function_value(self):
        # Over k
        Qad_nn = np.sum(abs(self.Qadk_nn), axis=2) / self.Nk
        # Over d
        Qa_nn  = 0
        self.P = 0
        for d in range(self.Nd):
            Qa_nn += Qad_nn[:,d]**2 * self.wd[d]
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
            Qa_nn += Qad_nn[:,d] * self.wd[d] / np.sum(self.wd)
        self.Qa_nn = Qa_nn.copy()
        # over a
        for a in range(self.Na):
            self.Qa_ii[a] = Qa_nn[a].diagonal()
        #
    #

    def translate_all_to_cell(self, cell=[0,0,0]):
        for a in range(self.Na):
            Qd_nn = self.Qad_nn[a]
            sc_c = np.angle(Qd_nn[:3].diagonal(0,1,2)).T * \
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
            Wtemp = np.zeros((No,No), complex)
            #
            for a in range(self.Na):

                for d, wd in enumerate(self.wd):
                    #
                    Qak_nm = self.Qadk_nm[a,d] 
                    diagQ  = self.Qad_nn[a,d].diagonal()
                    Qa_ii  = np.repeat(diagQ, No).reshape(No, No)
                    k1 = self.lst_dk[d, k]
                    k2 = self.invlst_dk[d, k]
                    Qk_nn = self.Qadk_nn[a,d]

                    temp = Qa_ii.T * Qk_nn[k].conj() - Qa_ii * Qk_nn[k2].conj()
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

        dim = self.gd.N_c
        largedim = dim * [N1, N2, N3]
        pmgrid = np.zeros(largedim, dtype=complex)

        #

        for k, k_kc in enumerate(self.k_kc):
            # The coordinate vector of wannier functions
            if isinstance(index, int):
                vec_n = self.W_k[k, :, index]
            else:
                vec_n = np.dot(self.W_k[k], index)

            pm_G = np.zeros(dim, complex)
            for n, coeff in enumerate(vec_n):
                pm_G += coeff * self.get_pseudo_wave_function(
                                     n, k, self.spin, pad=True)

            # Distribute the small wavefunction over large cell:
            for n1 in range(N1):
                na = n1
                for n2 in range(N2):
                    nb = n2
                    for n3 in range(N3): # sign?
                        nc = n3
                        if cell!=None:
                            na, nb, nc = cell
                        e = np.exp(-2.j * pi * np.dot([n1, n2, n3], k_kc))
                        pmgrid[na * dim[0]:(na + 1) * dim[0],
                               nb * dim[1]:(nb + 1) * dim[1],
                               nc * dim[2]:(nc + 1) * dim[2]] += e * pm_G

        # Normalization
        pmgrid /= np.sqrt(self.Nk)
        return pmgrid
        #
    #

    def get_pseudo_wave_function(self, band, kpt=0, spin=0,
                                 broadcast=True,
                                 pad=True):
        if pad:
            psit_G = self.get_pseudo_wave_function(band, kpt,
                                                   spin, True,
                                                   pad=False)
            if psit_G is None:
                return
            else:
                return self.wfs.gd.zero_pad(psit_G)

        psit_G = self.wfs.get_wave_function_array(band, kpt, spin,
                                                  periodic=False)
        if broadcast:
            if not self.wfs.world.rank == 0:
                psit_G = self.wfs.gd.empty(dtype=self.wfs.dtype,
                                           global_array=True)
            self.wfs.world.broadcast(psit_G, 0)
            return psit_G / Bohr**1.5
        elif self.wfs.world.rank == 0:
            return psit_G / Bohr**1.5


    def write_cube(self, index, fname, repeat=None, 
                   real=True, cell=None):

        from ase.io.cube import write_cube
        from ase.io import write

        # Default size of plotting cell is the one corresponding to k-points.
        if repeat is None:
            repeat = self.kgd
        atoms = self.atoms * repeat
        func = self.get_function(index, repeat, cell)

        # Handle separation of complex wave into real parts
        if real:
            if self.Nk == 1:
                func *= np.exp(-1.j * np.angle(func.max()))
                if 0: assert max(abs(func.imag).flat) < 1e-4
                func = func.real
            else:
                func = (func * func.max()/(abs(func.max()))).real
        else:
            phase_fname = fname.split('.')
            phase_fname.insert(1, 'phase')
            phase_fname = '.'.join(phase_fname)
            write(phase_fname, atoms, data=np.angle(func))
            func = abs(func)
        write(fname, atoms, data=func)
