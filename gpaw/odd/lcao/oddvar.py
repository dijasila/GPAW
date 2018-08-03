from ase.units import Hartree, Bohr
from gpaw.xc import XC
from gpaw.poisson import PoissonSolver, FFTPoissonSolver
from gpaw.transformers import Transformer
from gpaw.odd.lcao.potentials import *
from gpaw.odd.lcao.tools import *
from gpaw.odd.lcao.search_directions import LBFGSdirection as LBFGS_new
from gpaw.odd.lcao.search_directions import QuickMin, HZcg, \
    FRcg, LBFGSdirection_prec
from gpaw.odd.lcao.line_search import *
from gpaw.odd.lcao.wave_function_guess import get_initial_guess, loewdin

import scipy
from scipy.linalg import expm, eigh
from scipy.sparse import csc_matrix

import numpy as np
import copy
import time

from gpaw.forces import calculate_forces
from ase.calculators.calculator import Calculator
from gpaw.utilities.partition import AtomPartition
from gpaw.output import print_positions
import gpaw.mpi as mpi

all_changes = ['positions', 'numbers', 'cell', 'pbc',
               'initial_charges', 'initial_magmoms']


def calculate_kinetic_energy(density, wfs, setups):
    # pseudo-part
    e_kinetic = 0.0
    e_kin_paw = 0.0

    for kpt in wfs.kpt_u:
        rho_MM = \
            wfs.calculate_density_matrix(kpt.f_n,
                                         kpt.C_nM)
        e_kinetic += np.einsum('ij,ji->',
                               kpt.T_MM,
                               rho_MM)

    e_kinetic = wfs.kd.comm.sum(e_kinetic)

    # paw corrections
    for a, D_sp in density.D_asp.items():
        setup = setups[a]
        D_p = D_sp.sum(0)
        e_kin_paw += np.dot(setup.K_p, D_p) + setup.Kc

    e_kin_paw = density.gd.comm.sum(e_kin_paw)

    return e_kinetic.real + e_kin_paw


class ODDvarLcao(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, calc, odd='Zero', method='LBFGS_prec',
                 beta=(1.0, 1.0), initial_orbitals='KS',
                 initial_rotation='zero',
                 occupied_only=False,
                 g_tol=1.0e-4,
                 n_counter=1000, poiss_eps=1e-16, reset_orbitals=False,
                 line_search_method='SWC', awc=True,
                 check_forces=False,
                 memory_lbfgs=10, sic_coarse_grid=True,
                 max_iter_line_search=10, turn_off_swc=False,
                 names_of_files=('energy.npy', 'forces.npy'),
                 sparse=False, diag_prec=True,
                 prec='prec_3', save_orbitals=False):
        """
        :param calc: GPAW obj.
        :param odd: ODD potential
        :param method: optimisation algorithm
        :param beta: scaling factor for SIC term
        :param initial_orbitals: initial guess for orbitals
        :param initial_rotation: initial guess for skew-herm. matrix
        :param occupied_only: unitary optimisation of occup. states
        :param g_tol: ||*||_{inf} gradient tolerance for termination
        :param n_counter: maximum number of iterations
        """

        Calculator.__init__(self)

        self.poiss_eps = poiss_eps
        self.calc = calc
        self.odd = odd
        self.beta = beta
        self.method = method
        self.line_search_method = line_search_method
        self.calc_required = True
        self.check_forces = check_forces

        self.initial_rotation = initial_rotation
        self.initial_orbitals = initial_orbitals
        self.occupied_only = occupied_only
        self.reset_orbitals = reset_orbitals
        self.g_tol = g_tol / Hartree
        self.n_counter = n_counter

        self.sic_s = {}  # Self-interaction correction per spin
        self.sic_n = {}  # Self-interaction correction per orbital
        self.e_ks = 0.0  # Kohn-Sham energy

        self.get_en_and_grad_iters = 0
        self.awc = awc

        self.memory_lbfgs = memory_lbfgs
        self.need_initialization = True

        self.sic_coarse_grid = sic_coarse_grid

        self.max_iter_line_search = max_iter_line_search
        self.turn_off_swc = turn_off_swc

        self.names_of_files = names_of_files

        self.sparse = sparse
        self.diag_prec = diag_prec

        self.prec = prec
        self.save_orbitals = save_orbitals

    def initialize(self):

        if self.calc_required:
            # run usual KS
            self.calc.calculate()
            self.calc_required = False
            self.atoms = self.calc.atoms.copy()

        if self.atoms is None:
            self.atoms = self.calc.atoms.copy()

        self.initial_arrays()

        self.evecs = {}
        self.evals = {}

        self.log("Poisson for orbitals: eps = ", self.poiss_eps)
        self.log("Self-interaction corrections: ", self.odd)
        self.log("Minimisation algrorithm: ", self.method)
        self.log("Occupied states only: ", self.occupied_only)
        self.log("Scale of corrections: ", self.beta)
        self.log("gradient tolerance for termination: ",
                 self.g_tol * Hartree)

        if self.odd is 'PZ_SIC':

            self.pot = PZpotentialLcao(self.gd, self.xc,
                                       self.poiss, self.ghat,
                                       self.restrictor,
                                       self.interpolator,
                                       self.setups,
                                       self.beta,
                                       self.dtype,
                                       self.timer,
                                       self.wfs.basis_functions,
                                       self.spos_ac,
                                       sic_coarse_grid=self.sic_coarse_grid,
                                       )

        elif self.odd is 'Zero':

            if self.occupied_only is True:
                self.log('Need to include unoccupied states')
                self.log('for zero-odd-potential')

                raise Exception('Cannot run zero-potential'
                                ' with occupied states only')

            self.pot = ZeroOddLcao(self.dtype, self.timer)
        else:
            raise Exception('I do not know this odd potential.')

        # initial matrix
        self.n_dim = {}
        if self.occupied_only is True:
            for kpt in self.wfs.kpt_u:
                u = kpt.s * self.n_kps + kpt.q
                n_occ = 0
                for f in kpt.f_n:
                    if f > 1.0e-10:
                        n_occ += 1
                self.n_dim[u] = n_occ
        else:
            n_b = self.calc.get_number_of_bands()
            for kpt in self.wfs.kpt_u:
                u = kpt.s * self.n_kps + kpt.q
                self.n_dim[u] = n_b

        self.G_s = {}
        for kpt in self.wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            self.G_s[k] = np.zeros(shape=(self.n_dim[k],
                                          self.n_dim[k]),
                                   dtype=self.dtype)

        self.log("Initial guess for orbitals:", self.initial_orbitals)
        self.log("Initial guess for skew-herm. matrix:",
                 self.initial_rotation)

        self.initial_guess(self.n_dim)

        self.log("Initial guess is done")
        self.log(flush=True)

        n_d = 0
        for kpt in self.wfs.kpt_u:
            u = kpt.s * self.n_kps + kpt.q

            if self.dtype == complex:
                n_d += self.n_dim[u] ** 2
            else:
                n_d += self.n_dim[u] * (self.n_dim[u] - 1) // 2

        if self.method is 'LBFGS':
            self.search_direction = LBFGS_new(self.wfs,
                                              m=self.memory_lbfgs)
        elif self.method is 'LBFGS_prec':
            self.search_direction = \
                LBFGSdirection_prec(self.wfs,
                                    m=self.memory_lbfgs,
                                    diag=self.diag_prec)
        elif self.method is 'QuickMin':
            self.search_direction = QuickMin(self.wfs)
        elif self.method == 'HZcg':
            self.search_direction = HZcg(self.wfs)
        elif self.method == 'FRcg':
            self.search_direction = FRcg(self.wfs)
        else:
            raise NotImplementedError

        if self.line_search_method is 'SWC':

            self.line_search = \
                StrongWolfeConditions(self.evaluate_phi_and_der_phi,
                                      self.log, method=self.method,
                                      awc=self.awc,
                                      max_iter=self.max_iter_line_search
                                      )

        elif self.line_search_method is 'NoLineSearch':

            self.line_search = \
                UnitStepLength(self.evaluate_phi_and_der_phi,
                               self.log)

        elif self.line_search_method is 'Parabola':
            self.line_search = \
                Parabola(self.evaluate_phi_and_der_phi,
                               self.log)
        else:
            raise NotImplementedError

        self.log("Dimension of space: %6d " % n_d)

        self.log(flush=True)

        self.need_initialization = False

    def initial_arrays(self):

        # From converged KS-DFT calculation
        self.setups = self.calc.setups
        self.wfs = self.calc.wfs
        self.n_kps = self.wfs.kd.nks // self.wfs.kd.nspins
        self.dtype = self.wfs.dtype
        self.timer = self.calc.timer
        self.log = self.calc.log
        self.log("Type of orbitals: ", self.dtype)
        self.check_assertions()
        self.xc = XC(self.calc.parameters.xc)
        self.ham = self.calc.hamiltonian
        self.ham.poisson.direct_min_zero_init_phi = False
        self.den = self.calc.density
        self.den.direct_min = True
        self.occ = self.calc.occupations
        self.nspins = self.wfs.nspins
        self.xc.initialize(self.ham, self.den,
                           self.wfs, self.occ)
        # Force int. occ
        if self.odd == 'PZ_SIC':
            self.occ.magmom = int(round(self.occ.magmom))

        # Grid-descriptor
        self.finegd = self.den.finegd
        self.gd = self.den.gd  # this is coarse grid
        self.ghat = self.den.ghat  # they are on fine grid
        self.spos_ac = self.calc.atoms.get_scaled_positions()
        self.interpolator = Transformer(self.gd, self.finegd, 3)
        self.restrictor = Transformer(self.finegd, self.gd, 3)

        if self.odd is not 'Zero':
            self.poiss = PoissonSolver(relax='GS',
                                       eps=self.poiss_eps,
                                       sic_gg=True)

            if self.sic_coarse_grid is True:
                self.poiss.set_grid_descriptor(self.gd)
            else:
                self.poiss.set_grid_descriptor(self.finegd)
        else:
            self.poiss = None

    def initial_guess(self, n_dim):

        self.A_s = {}
        if self.initial_rotation is 'small_random':
            for kpt in self.wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                self.A_s[k] = \
                    random_skew_herm_matrix(n_dim[k],
                                            -1e-3, 1e-3, self.dtype)

        elif self.initial_rotation is 'random':
            for kpt in self.wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                self.A_s[k] = \
                    random_skew_herm_matrix(n_dim[k],
                                            -1e1, 1e1, self.dtype)

        elif self.initial_rotation is 'zero':
            for kpt in self.wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                self.A_s[k] = np.zeros(shape=(n_dim[k],
                                              n_dim[k]),
                                              dtype=self.dtype)
        else:
            raise Exception(
                'Check the \'initial rotation\' parameter!')

        # Make coefficients the same for all
        for kpt in self.wfs.kpt_u:
            self.wfs.gd.comm.broadcast(kpt.C_nM, 0)

        self.C_nM_init = get_initial_guess(self.dtype,
                                           self.calc, n_dim,
                                           self.initial_orbitals,
                                           self.occupied_only,
                                           self.log)

        for kpt in self.wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if self.n_dim[k] > 0:
                U = expm(self.A_s[k])
                self.C_nM_init[k][:n_dim[k]] = \
                    np.dot(U.T, self.C_nM_init[k][:n_dim[k]])

                self.A_s[k] = np.zeros(shape=(n_dim[k],
                                              n_dim[k]),
                                              dtype=self.dtype)

        # Make init guess the same for all procs
        for kpt in self.wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            self.wfs.gd.comm.broadcast(self.C_nM_init[k], 0)

    def calculate(self, atoms, properties,
                  system_changes):

        Calculator.calculate(self, atoms)

        # Start with KS-DFT
        if self.calc_required:
            self.calc.atoms = atoms
            self.calc.calculate()
            self.calc_required = False
            self.initialize()

        if self.need_initialization:
            self.initialize()

        if 'positions' in system_changes:
            self.update_positions(atoms)
            self.need_initialization = False
            self.run()

        if 'energy' in properties:
            self.results['energy'] = (self.e_ks +
                                      self.total_sic) * Hartree

            self.results['free_energy'] = (self.e_ks +
                                           self.total_sic) * Hartree

        if 'forces' in properties:
            self.results['forces'] = self.calculate_forces_2()

    def update_positions(self, atoms=None):

        mpi.synchronize_atoms(atoms, self.wfs.world)

        spos_ac = atoms.get_scaled_positions() % 1.0

        rank_a = self.wfs.gd.get_ranks_from_positions(spos_ac)
        atom_partition = AtomPartition(self.wfs.gd.comm,
                                       rank_a, name='gd')

        self.wfs.set_positions(spos_ac, atom_partition)
        self.den.set_positions(spos_ac, atom_partition)
        self.ham.set_positions(spos_ac, atom_partition)

        totmom_v, magmom_av = self.den.estimate_magnetic_moments()
        print_positions(atoms, self.log, magmom_av)

        self.den.sic = True

        self.setups = self.wfs.setups

        if self.odd is 'PZ_SIC':

            self.log('Reinitialize ODD potential..')
            self.log(flush=True)

            self.xc.initialize(self.ham, self.den,
                               self.wfs, self.occ)

            # Grid-descriptor
            self.finegd = self.den.finegd
            self.gd = self.den.gd  # this is coarse grid
            self.ghat = self.den.ghat  # they are on fine grid
            self.interpolator = Transformer(self.gd, self.finegd, 3)
            self.restrictor = Transformer(self.finegd, self.gd, 3)

            self.spos_ac = spos_ac
            self.poiss = PoissonSolver(relax='GS',
                                       eps=self.poiss_eps,
                                       sic_gg=True)

            if self.sic_coarse_grid is True:
                self.poiss.set_grid_descriptor(self.gd)
            else:
                self.poiss.set_grid_descriptor(self.finegd)

            self.pot = PZpotentialLcao(self.gd, self.xc,
                                       self.poiss, self.ghat,
                                       self.restrictor,
                                       self.interpolator,
                                       self.setups,
                                       self.beta,
                                       self.dtype,
                                       self.timer,
                                       self.wfs.basis_functions,
                                       self.spos_ac,
                                       sic_coarse_grid=self.sic_coarse_grid)

        for kpt in self.wfs.kpt_u:

            k = self.n_kps * kpt.s + kpt.q
            if sum(kpt.f_n) < 1.0e-10:
                continue

            U = expm(self.A_s[k])
            self.C_nM_init[k][:self.n_dim[k]] = \
                np.dot(U.T, self.C_nM_init[k][:self.n_dim[k]])
            self.C_nM_init[k] = \
                loewdin(self.C_nM_init[k],
                        self.wfs.S_qMM[kpt.q].conj())
            self.A_s[k] = np.zeros_like(self.A_s[k])
            # Make init guess the same
            self.wfs.gd.comm.broadcast(self.C_nM_init[k], 0)
            kpt.rho_MM = None

        self.get_en_and_grad_iters = 0

        if self.method is 'LBFGS':
            self.search_direction = LBFGS_new(self.wfs,
                                              m=self.memory_lbfgs)
        elif self.method is 'LBFGS_prec':
            self.search_direction = \
                LBFGSdirection_prec(self.wfs,
                                    m=self.memory_lbfgs,
                                    diag=self.diag_prec)
        elif self.method is 'QuickMin':
            self.search_direction = QuickMin(self.wfs)
        elif self.method == 'HZcg':
            self.search_direction = HZcg(self.wfs)
        elif self.method == 'FRcg':
            self.search_direction = FRcg(self.wfs)

        if self.line_search_method is 'SWC':
            self.line_search = \
                StrongWolfeConditions(self.evaluate_phi_and_der_phi,
                                      self.log, method=self.method,
                                      awc=self.awc,
                                      max_iter=self.max_iter_line_search
                                      )
        elif self.line_search_method is 'NoLineSearch':
            self.line_search = \
                UnitStepLength(self.evaluate_phi_and_der_phi,
                               self.log)
        elif self.line_search_method is 'Parabola':
            self.line_search = \
                Parabola(self.evaluate_phi_and_der_phi,
                               self.log)
        else:
            raise NotImplementedError

    def get_energy_and_gradients(self, A_s, n_dim, C_nM0=None):

        """
        Energy E = E[C exp(A)]. Gradients G_ij[C, A] = dE/dA_ij
        If C_nM0 = None, then function returns energy and gradients
        ,E[C, A] and G[C, A], at C = C_init, A = A_s.
        Otherwise, function returns E[C, A] and G[C, A] at
        C = C_nM0 exp(A_s), A = 0

        :param A_s:
        :param n_dim:
        :param C_nM0:
        :return:
        """

        occupied_only = self.occupied_only
        assert (isinstance(C_nM0, dict) or C_nM0 is None)

        self.timer.start('ODD Unitary rotation')

        for kpt in self.wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q

            if n_dim[k] == 0:
                continue
            self.timer.start('ODD calc matrix exponential')
            U, self.evecs[k], self.evals[k] = expm_ed(A_s[k],
                                                      evalevec=True)
            self.timer.stop('ODD calc matrix exponential')

            self.timer.start('ODD unitary rotation')
            if C_nM0 is None:
                kpt.C_nM[:n_dim[k]] = \
                    np.dot(U.T, self.C_nM_init[k][:n_dim[k]])
            else:
                kpt.C_nM[:n_dim[k]] = \
                    np.dot(U.T, C_nM0[k][:n_dim[k]])
            self.timer.stop('ODD unitary rotation')
            del U

        self.timer.stop('ODD Unitary rotation')

        self.timer.start('ODD update_ks_energy_and_hamiltonian')
        # FIXME: if you run PZ-SIC for occupeied states only
        # and if KS functional is not unitary invariant on occupied states
        # then you need to update update_ks_energy_and_hamiltonian
        # every iteration.
        if occupied_only is not True or \
                self.get_en_and_grad_iters == 0:
            self.update_ks_energy_and_hamiltonian()
        self.timer.stop('ODD update_ks_energy_and_hamiltonian')

        self.timer.start('ODD get gradients')
        for kpt in self.wfs.kpt_u:

            self.H_MM = \
                self.wfs.eigensolver.calculate_hamiltonian_matrix(
                            self.ham,
                            self.wfs,
                            kpt)
            # make matrix hermitian
            ind_l = np.tril_indices(self.H_MM.shape[0], -1)
            self.H_MM[(ind_l[1], ind_l[0])] = \
                self.H_MM[ind_l].conj()

            k = self.n_kps * kpt.s + kpt.q
            if n_dim[k] == 0:
                continue
            f_n = kpt.f_n
            C_nM = kpt.C_nM
            wfs = self.wfs
            setup = self.setups

            if max(f_n) < 1.0e-10:
                self.G_s[k] = np.zeros_like(self.A_s[k])
                self.sic_n[k] = np.zeros(shape=(1, 2), dtype=float)
                self.sic_s[k] = self.sic_n[k].sum()
                continue
            if self.reset_orbitals is False:
                self.G_s[k], self.sic_n[k] = \
                    self.pot.get_gradients(f_n, C_nM,
                                           kpt,
                                           wfs,
                                           setup,
                                           self.evecs[k],
                                           self.evals[k],
                                           self.H_MM,
                                           A_s[k],
                                           occupied_only=
                                           occupied_only)
            else:
                self.G_s[k], self.sic_n[k] = \
                    self.pot.get_gradients(f_n, C_nM,
                                           kpt,
                                           wfs,
                                           setup,
                                           self.evecs[k],
                                           self.evals[k],
                                           self.H_MM,
                                           A=None,
                                           occupied_only=
                                           occupied_only)
            self.sic_s[k] = self.sic_n[k].sum()

        self.timer.stop('ODD get gradients')
        self.get_en_and_grad_iters += 1
        self.total_sic = sum(self.sic_s.values())
        self.total_sic = self.wfs.kd.comm.sum(self.total_sic)
        self.total_sic *= float(3 - self.nspins)

        return (self.e_ks + self.total_sic), \
                copy.deepcopy(self.G_s)

    def update_ks_energy_and_hamiltonian(self):

        # Using new states update KS
        for kpt in self.wfs.kpt_u:
            self.wfs.atomic_correction.calculate_projections(self.wfs,
                                                             kpt)
        self.den.update(self.wfs)
        self.ham.update(self.den)
        # self.occ.calculate(self.wfs)
        self.e_ks = self.ham.get_energy(self.occ)

        e_kin = calculate_kinetic_energy(self.den,
                                         self.wfs,
                                         self.setups)

        self.e_ks = self.e_ks - self.ham.e_kinetic + e_kin
        self.ham.e_kinetic = e_kin
        self.ham.e_total_free = self.e_ks
        self.ham.e_total_extrapolated = self.e_ks

    def get_search_direction(self, A_s, G_s):

        # structure of vector is
        # (x_1+,x_2+,..,y_1+, y_2+,..,x_1-,x_2-,..,y_1-, y_2-,.. )

        g_k = {}
        a_k = {}

        for k in A_s.keys():
            if self.dtype is complex:
                il1 = np.tril_indices(A_s[k].shape[0])
            else:
                il1 = np.tril_indices(A_s[k].shape[0], -1)

            g_k[k] = G_s[k][il1]
            a_k[k] = A_s[k][il1]

        if str(self.search_direction) == 'LBFGS_prec':

            if self.odd == 'Zero':
                update_counter = 20
            else:
                update_counter = 10

            if self.search_direction.k % update_counter == 0:
                self.heiss = {}
                self.heiss_inv = {}
                for kpt in self.wfs.kpt_u:
                    k = self.n_kps * kpt.s + kpt.q
                    self.H_MM = \
                        self.wfs.eigensolver.calculate_hamiltonian_matrix(
                            self.ham,
                            self.wfs,
                            kpt)

                    # make matrix hermitian
                    ind_l = np.tril_indices(self.H_MM.shape[0], -1)
                    self.H_MM[(ind_l[1], ind_l[0])] = \
                        self.H_MM[ind_l].conj()

                    if self.odd == 'Zero':
                        self.heiss[k] = self.pot.get_hessian(kpt,
                                                             self.H_MM)
                    elif self.odd == 'PZ_SIC':
                        self.heiss[k] = self.pot.get_hessian(kpt, self.H_MM,
                                                     self.n_dim,
                                                     self.wfs, self.setups,
                                                     diag_heiss=self.diag_prec,
                                                     occupied_only=False,
                                                     h_type='ks'
                                                     )
                    else:
                        raise NotImplementedError

            if self.search_direction.k >= 0:
                for kpt in self.wfs.kpt_u:
                    k = self.n_kps * kpt.s + kpt.q

                    if self.diag_prec:
                        if self.prec == 'prec_1':
                            if self.dtype is float:
                                self.heiss_inv[k] = np.zeros_like(
                                    self.heiss[k])

                                for i in range(self.heiss[k].shape[0]):

                                    if abs(self.heiss[k][i]) < 1.0e-3:
                                        self.heiss_inv[k][i] = 1.0
                                    else:
                                        self.heiss_inv[k][i] = 1.0 / (
                                                self.heiss[k][i].real)

                            else:
                                self.heiss_inv[k] = np.zeros_like(
                                    self.heiss[k])

                                for i in range(self.heiss[k].shape[0]):

                                    if abs(self.heiss[k][i]) < 1.0e-3:
                                        self.heiss_inv[k][i] = 1.0 + 1.0j
                                    else:
                                        self.heiss_inv[k][i] = 1.0 / (
                                            self.heiss[k][i].real) + \
                                            1.0j / (self.heiss[k][i].imag)
                        elif self.prec == 'prec_2':
                            if self.dtype is float:
                                self.heiss_inv[k] = np.zeros_like(
                                    self.heiss[k])

                                self.heiss_inv[k] = 1.0 / (
                                        self.heiss[k].real +
                                        self.search_direction.beta_0 ** (-1))
                            else:
                                self.heiss_inv[k] = np.zeros_like(self.heiss[k])

                                self.heiss_inv[k] = \
                                    1.0 / (self.heiss[k].real +
                                           self.search_direction.beta_0 ** (-1)) + \
                                    1.0j / (self.heiss[k].imag +
                                            self.search_direction.beta_0 ** (-1))
                        elif self.prec == 'prec_3':
                            if self.dtype is float:
                                self.heiss_inv[k] = np.zeros_like(
                                    self.heiss[k])

                                self.heiss_inv[k] = 1.0 / (
                                        0.75 * self.heiss[k] +
                                        0.25 * self.search_direction.beta_0 ** (
                                            -1))
                            else:
                                self.heiss_inv[k] = np.zeros_like(self.heiss[k])

                                self.heiss_inv[k] = \
                                    1.0 / (0.75 * self.heiss[k].real +
                                           0.25 * self.search_direction.beta_0 ** (-1)) + \
                                    1.0j / (0.75 * self.heiss[k].imag +
                                            0.25 * self.search_direction.beta_0 ** (-1))
                        else:
                            raise NotImplementedError

                    elif self.sparse:
                        self.heiss[k] = \
                            scipy.sparse.linalg.inv(self.heiss[k] +
                                                    scipy.sparse.eye(self.heiss[k].shape[0]))
                    else:
                        self.heiss[k] = np.linalg.inv(self.heiss[k] +
                                                      (1.0) *\
                                                      np.eye(self.heiss[k].shape[0]))

            p_k = self.search_direction.update_data(self.wfs, a_k,
                                                    g_k,
                                                    self.heiss_inv)
        else:
            p_k = self.search_direction.update_data(self.wfs, a_k, g_k)

        del a_k, g_k

        P_s = {}

        for k in p_k.keys():
            P_s[k] = np.zeros_like(A_s[k])
            if self.dtype is complex:
                il1 = np.tril_indices(A_s[k].shape[0])
            else:
                il1 = np.tril_indices(A_s[k].shape[0], -1)
            P_s[k][il1] = p_k[k]
            # make it skew-hermitian
            ind_l = np.tril_indices(P_s[k].shape[0], -1)
            P_s[k][(ind_l[1], ind_l[0])] = -P_s[k][ind_l].conj()
        del p_k

        return P_s

    def evaluate_phi_and_der_phi(self, A_s, P_s, n_dim, alpha=0.0,
                                 phi=None, G_s=None, C_nM0=None):
        """
        phi = f(x_k + alpha_k*p_k)
        der_phi = \grad f(x_k + alpha_k*p_k) \cdot p_k
        :return:  phi, der_phi # floats
        """
        if phi is None or G_s is None:
            X_s = {k: A_s[k] + alpha * P_s[k] for k in A_s.keys()}
            phi_1, G_s_0 = \
                self.get_energy_and_gradients(X_s, n_dim)
            del X_s
        else:
            phi_1 = phi
            G_s_0 = G_s

        der_phi_v = {}
        der_phi = 0.0
        for k in P_s.keys():
            if self.dtype is complex:
                il1 = np.tril_indices(P_s[k].shape[0])
            else:
                il1 = np.tril_indices(P_s[k].shape[0], -1)
            p_k = P_s[k][il1]
            der_phi_v[k] = G_s_0[k][il1]
            der_phi += np.dot(der_phi_v[k].conj(), p_k).real
        der_phi = self.wfs.kd.comm.sum(der_phi)
        # FIXME: ?
        del P_s

        return phi_1, der_phi, G_s_0

    def estimate_gradients(self, delta=1.0e-6):

        if self.need_initialization:
            self.initialize()
        assert type(delta) is list or \
               type(delta) is float or \
               type(delta) is int
        if type(delta) is int or float:
            delta = [delta]

        n_dim = self.n_dim
        delta = sorted(delta, reverse=True)
        g_num_norm = {}
        g_an_norm = {}

        for h in delta:
            G_an, G_num = \
                self.get_numerical_gradients(self.A_s,
                                             n_dim,
                                             dtype=self.dtype,
                                             eps=h)
            g_an = {}
            g_num = {}
            g_an_norm_0 = 0.0
            g_num_norm_0 = 0.0
            for k in G_an.keys():
                if self.dtype is complex:
                    il1 = np.tril_indices(G_an[k].shape[0])
                else:
                    il1 = np.tril_indices(G_an[k].shape[0], -1)
                g_an[k] = G_an[k][il1]
                g_num[k] = G_num[k][il1]
                g_an_norm_0 += np.dot(g_an[k].conj(), g_an[k]).real
                g_num_norm_0 += np.dot(g_num[k].conj(), g_num[k]).real
            g_num_norm[h] = np.sqrt(g_num_norm_0)
            g_an_norm[h] = np.sqrt(g_an_norm_0)
            np.save('g_an_' + str(h), g_an)
            np.save('g_num_' + str(h), g_num)
            np.save('G_an_' + str(h), G_an)
            np.save('G_num_' + str(h), G_num)
        np.save('g_num_norm', g_num_norm)
        np.save('g_an_norm', g_an_norm)

    def get_numerical_gradients(self, A_s, n_dim, eps=1.0e-5,
                                dtype=complex):
        h = [eps, -eps]
        coef = [1.0, -1.0]
        Gr_n_x = {}
        Gr_n_y = {}
        E_0, G = self.get_energy_and_gradients(A_s, n_dim)
        self.log("Estimating gradients using finite differences..")
        self.log(flush=True)
        if dtype == complex:
            for kpt in self.wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                dim = A_s[k].shape[0]
                for z in range(2):
                    grad = np.zeros(shape=(dim * dim),
                                    dtype=self.dtype)
                    for i in range(dim):
                        for j in range(dim):
                            self.log(k, z, i, j)
                            self.log(flush=True)
                            A = A_s[k][i][j]
                            for l in range(2):
                                if z == 1:
                                    if i == j:
                                        A_s[k][i][j] = A + 1.0j * h[l]
                                    else:
                                        A_s[k][i][j] = A + 1.0j * h[
                                            l]
                                        A_s[k][j][i] = -np.conjugate(
                                            A + 1.0j * h[l])
                                else:
                                    if i == j:
                                        A_s[k][i][j] = A + 0.0j * h[l]
                                    else:
                                        A_s[k][i][j] = A + h[
                                            l]
                                        A_s[k][j][i] = -np.conjugate(
                                            A + h[l])

                                E = self.get_energy_and_gradients(A_s,
                                                                  n_dim)[
                                    0]
                                grad[i * dim + j] += E * coef[l]
                            grad[i * dim + j] *= 1.0 / (2.0 * eps)
                            if i == j:
                                A_s[k][i][j] = A
                            else:
                                A_s[k][i][j] = A
                                A_s[k][j][i] = -np.conjugate(A)
                    if z == 0:
                        Gr_n_x[k] = grad[:].reshape(
                            int(np.sqrt(grad.shape[0])),
                            int(np.sqrt(grad.shape[0])))
                    else:
                        Gr_n_y[k] = grad[:].reshape(
                            int(np.sqrt(grad.shape[0])),
                            int(np.sqrt(grad.shape[0])))
            Gr_n = {k: (Gr_n_x[k] + 1.0j * Gr_n_y[k]) for k in
                    Gr_n_x.keys()}
        else:
            for kpt in self.wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                dim = A_s[k].shape[0]
                grad = np.zeros(shape=(dim * dim), dtype=self.dtype)
                for i in range(dim):
                    for j in range(dim):
                        self.log(k, i, j)
                        self.log(flush=True)
                        A = A_s[k][i][j]
                        for l in range(2):
                            if i == j:
                                A_s[k][i][j] = A
                            else:
                                A_s[k][i][j] = A + h[l]
                                A_s[k][j][i] = -(A + h[l])
                            E = self.get_energy_and_gradients(A_s,
                                                              n_dim)[
                                0]
                            grad[i * dim + j] += E * coef[l]
                        grad[i * dim + j] *= 1.0 / (2.0 * eps)
                        if i == j:
                            A_s[k][i][j] = A
                        else:
                            A_s[k][i][j] = A
                            A_s[k][j][i] = -A
                Gr_n_x[k] = grad[:].reshape(
                    int(np.sqrt(grad.shape[0])),
                    int(np.sqrt(grad.shape[0])))
            Gr_n = {k: (Gr_n_x[k]) for k in Gr_n_x.keys()}
        return G, Gr_n

    def save_data(self):
        # TODO: modify this for parallel version
        name = ''  # self.calc.atoms.get_chemical_formula()
        np.save(name + 'E_ks', self.E_ks)
        np.save(name + 'E_sic', self.E_sic)
        np.save(name + 'sic_n', self.sic_n)
        np.save(name + 'pot_counter', self.potential_counter)

    def save_coefficients(self):
        for kpt in self.wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            np.save('C_nM_' + str(k), kpt.C_nM)

    def log_f(self, log, niter, g_max, e_ks, e_sic):

        T = time.localtime()
        if niter == 0:

            header = '                      Kohn-Sham          SIC' \
                     '        Total    ||g||_inf\n' \
                     '           time         energy:      energy:' \
                     '      energy:    gradients:'

            log(header)
        log('iter: %3d  %02d:%02d:%02d ' %
            (niter,
             T[3], T[4], T[5]
             ), end='')
        log('%11.6f  %11.6f  %11.6f  %11.1e' %
            (Hartree * e_ks,
             Hartree * e_sic,
             Hartree * (e_ks + e_sic),
             Hartree * g_max), end='')
        log(flush=True)

    def write_final_output(self, log, e_ks, e_sic, sic_n, eval=None, f_sn=None):

        log('Energy contributions relative to reference atoms:',
            '(reference = {0:.6f})\n'.format(self.setups.Eref *
                                             Hartree))
        energies = [('Kinetic:      ', self.ham.e_kinetic),
                    ('Potential:    ', self.ham.e_coulomb),
                    ('External:     ', self.ham.e_external),
                    ('XC:           ', self.ham.e_xc),
                    ('Local:        ', self.ham.e_zero)]
        for name, e in energies:
            log('%-14s %+11.6f' % (name, Hartree * e))
        log()
        log('Kohn-Sham energy: {0:>11.6f}'.format(Hartree *
                                                       e_ks))
        log('SIC energy:       {0:>11.6f}'.format(Hartree *
                                                      e_sic))
        log('-----------------------------')
        log('Total energy:     {0:>11.6f} eV\n'.format(
            Hartree * (e_ks + e_sic)))
        if not self.wfs.kd.gamma:
            return 0
        if self.odd == 'PZ_SIC':
            log('Orbital corrections from PZ_SIC:')
            for s in range(self.nspins):
                log('Spin: %3d ' % (s))
                header = """\
            Hartree    XC        Hartree + XC
            energy:    energy:   energy:    """
                log(header)
                i = 0
                u_s = 0.0
                xc_s = 0.0
                for u, xc in sic_n[s]:
                    log('band: %3d ' %
                        (i), end='')
                    i += 1
                    log('%11.6f%11.6f%11.6f' %
                        (-Hartree * u,
                         -Hartree * xc,
                         -Hartree * (u + xc)
                         ), end='')
                    log(flush=True)
                    u_s += u
                    xc_s += xc
                log('---------------------------------------------')
                log('Total     ', end='')
                log('%11.6f%11.6f%11.6f' %
                    (-Hartree * u_s,
                     -Hartree * xc_s,
                     -Hartree * (u_s + xc_s)
                     ), end='')
                log("\n")
        if eval is not None and f_sn is not None:
            if self.odd == 'Zero':
                if self.nspins == 1:
                    header = " Band  Eigenvalues  Occupancy"
                    log(header)
                    for i in range(len(eval[0])):
                        log('%5d  %11.5f  %9.5f' % (
                            i, Hartree * eval[0][i], f_sn[0][i]))
                if self.nspins == 2:
                    log('                  Up'
                        '                     Down')
                    log('Band  Eigenvalues  Occupancy  Eigenvalues  '
                        'Occupancy')
                    for n in range(len(eval[0])):
                        log('%5d  %11.5f  %9.5f  %11.5f  %9.5f' %
                            (n, Hartree * eval[0][n], f_sn[0][n],
                             Hartree * eval[1][n], f_sn[1][n]))
                log("\n")
            elif self.odd == 'PZ_SIC':
                log("For SIC calculations there are\n"
                    "diagonal elements of Lagrange matrix and "
                    "its eigenvalues:\n")
                if self.nspins == 1:
                    header = " Band         L_ii  Eigenvalues  " \
                             "Occupancy"
                    log(header)
                    lagr = sorted(self.pot.lagr_diag_s[0])
                    for i in range(len(eval[0])):
                        log('%5d  %11.5f  %11.5f  %9.5f' % (
                            i, Hartree * lagr[i],
                            Hartree * eval[0][i], f_sn[0][i]))
                if self.nspins == 2:
                    log('                        Up                 '
                        '                 Down')
                    log(' Band         L_ii  Eigenvalues  '
                        'Occupancy  '
                        '       L_ii  Eigenvalues  Occupancy')
                    lagr_0 = np.sort(self.pot.lagr_diag_s[0])
                    lagr_1 = np.sort(self.pot.lagr_diag_s[1])
                    for n in range(len(eval[0])):
                        log('%5d  %11.5f  %11.5f  '
                            '%9.5f  %11.5f  %11.5f  '
                            '%9.5f' %
                            (n, Hartree * lagr_0[n],
                             Hartree * eval[0][n],
                             f_sn[0][n],
                             Hartree * lagr_1[n],
                             Hartree * eval[1][n],
                             f_sn[1][n]))
                log("\n")
        self.log(flush=True)

    def check_assertions(self):
        """
        hopefully we won't need this method in future
        """
        assert self.wfs.bd.comm.size == 1  # band. paral not supported

        nsp = self.calc.get_number_of_spins()

        if self.odd == 'PZ_SIC':
            if nsp == 1:
                for f in self.calc.wfs.kpt_u[0].f_n:
                    if f < 1.0e-1:
                        continue
                    else:
                        if abs(f - 2.0) > 1.0e-1:
                            raise Exception('Use both spin channels for '
                                            'spin-polarized systems')
        if self.odd == 'PZ_SIC':
            for kpt in self.wfs.kpt_u:
                for f in kpt.f_n:
                    if f < 1.0e-6:
                        continue
                    if abs(f - 1.0) < 1.0e-6 or abs(f - 2.0) < 1.0e-6:
                        continue
                    else:
                        self.log('Fractional occupations\n')
                        self.log(flush=True)
                        raise Exception('Fractional occupations are not'
                                        'implemented in SIC yet')

    def calculate_forces_2(self):

        if self.odd is 'Zero':
            for kpt in self.wfs.kpt_u:
                kpt.rho_MM = \
                    self.wfs.calculate_density_matrix(kpt.f_n,
                                                      kpt.C_nM)

            if self.check_forces is True:
                F_av = calculate_forces(self.wfs, self.den,
                                        self.ham)
            else:
                F_av = calculate_forces(self.wfs, self.den,
                                        self.ham, self.log)
            for kpt in self.wfs.kpt_u:
                kpt.rho_MM = None
            return F_av * (Hartree / Bohr)

        elif self.odd is 'PZ_SIC':
            for kpt in self.wfs.kpt_u:
                kpt.rho_MM = \
                    self.wfs.calculate_density_matrix(kpt.f_n,
                                                      kpt.C_nM)
            F_av = calculate_forces(self.wfs, self.den,
                                    self.ham)
            F_av += self.pot.get_odd_corrections_to_forces(self.wfs,
                                                           self.den)
            F_av = self.wfs.kd.symmetry.symmetrize_forces(F_av)
            self.log('\nForces in eV/Ang:')
            c = Hartree / Bohr
            for a, setup in enumerate(self.wfs.setups):
                self.log('%3d %-2s %10.5f %10.5f %10.5f' %
                     ((a, setup.symbol) + tuple(F_av[a] * c)))
            self.log()
            return F_av * c

    def get_potential_energy_2(self):
        return (self.e_ks + self.total_sic) * Hartree

    def change_to_lbfgs(self, A_s, G_s):

        g_k = {}
        a_k = {}
        for k in A_s.keys():
            if self.dtype is complex:
                il1 = np.tril_indices(A_s[k].shape[0])
            else:
                il1 = np.tril_indices(A_s[k].shape[0], -1)
            g_k[k] = G_s[k][il1]
            a_k[k] = A_s[k][il1]
        self.log('Changed to LBFGS')
        self.log(flush=True)
        if self.method == 'LBFGS':
            self.search_direction = \
                LBFGS_new(self.wfs, m=self.memory_lbfgs)
        elif self.method == 'LBFGS_prec':
            self.search_direction = \
                LBFGSdirection_prec(self.wfs,
                                    m=self.memory_lbfgs,
                                    diag=self.diag_prec)
        sd = self.search_direction
        sd.kp[sd.k] = sd.p
        sd.x_k = sd.get_x(a_k)
        sd.g_k = sd.get_x(g_k)
        sd.s_k[sd.kp[sd.k]] = sd.zeros(g_k)
        sd.y_k[sd.kp[sd.k]] = sd.zeros(g_k)
        sd.k += 1
        sd.p += 1
        sd.kp[sd.k] = sd.p
        del g_k, a_k, G_s, A_s

    def run(self):

        if self.need_initialization:
            self.initialize()
            self.need_initialization = False

        n_dim = self.n_dim
        counter = 0
        # get initial energy and gradients
        e_total, g_0 = \
            self.get_energy_and_gradients(self.A_s, n_dim)
        # get maximum of gradients
        g_max = np.array([])
        for kpt in self.wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            g_max = np.append(g_max, np.append(g_0[k].imag,
                                               g_0[k].real))
        g_max = np.max(np.absolute(g_max))
        g_max = self.wfs.world.max(g_max)

        # stuff which are needed for minim.
        phi_0 = e_total
        phi_old = None
        der_phi_old = None
        phi_old_2 = None
        der_phi_old_2 = None
        self.E_ks = []
        self.E_sic = []
        self.G_m = []
        self.alpha_stars = []
        self.potential_counter = []

        self.log_f(self.log, counter, g_max,
                   self.e_ks, self.total_sic)
        if self.check_forces is True:
            energy = np.load(self.names_of_files[0])
            forces = np.load(self.names_of_files[1])
        ev = 2.0
        alpha = 1.0
        change_to_swc = False
        while g_max > self.g_tol and counter < self.n_counter:

            if self.turn_off_swc:
                if g_max * Hartree < 1.0e-3:
                    self.line_search = \
                        UnitStepLength(self.evaluate_phi_and_der_phi,
                                       self.log)
            if counter == 1:
                if self.line_search_method is 'NoLineSearch':
                    self.line_search = \
                    UnitStepLength(self.evaluate_phi_and_der_phi,
                                   self.log)
            self.timer.start('ODD get search direction')
            P_s = self.get_search_direction(self.A_s, g_0)
            self.timer.stop('ODD get search direction')

            if str(self.search_direction) == 'LBFGS' or \
                    str(self.search_direction) == 'LBFGS_prec':
                if not self.search_direction.stable:
                    self.search_direction = QuickMin(self.wfs)
                    self.log('Changed to QuickMin with unit step')
                    self.log(flush=True)

                    self.timer.start('ODD get search direction')
                    P_s = self.get_search_direction(self.A_s, g_0)
                    self.timer.stop('ODD get search direction')

                    self.line_search = \
                        UnitStepLength(self.evaluate_phi_and_der_phi,
                                       self.log)
                    if self.line_search_method == 'SWC':
                        change_to_swc = True
                    else:
                        change_to_swc = False
                    self.change_to_lbfgs(self.A_s, g_0)

            if self.reset_orbitals is True:
                for s in range(self.nspins):
                    self.A_s[s] = np.zeros_like(self.A_s[s])
            phi_0, der_phi_0, g_0 = \
                self.evaluate_phi_and_der_phi(self.A_s, P_s, n_dim,
                                              alpha=0.0,
                                              phi=phi_0, G_s=g_0)
            if counter > 0:
                phi_old = phi_0
                der_phi_old = der_phi_0
            if counter == 0 or ev < 0.1:
                if self.line_search_method is 'Parabola':
                    ev = eigh(g_0[0]*1.0j)[0]
                    ev = np.max(np.absolute(ev))
                    try:
                        ev1 = eigh(g_0[1]*1.0j)[0]
                        ev1 = np.max(np.absolute(ev1))
                    except:
                        ev1 = ev
                    ev = 0.1 * np.pi / max(ev, ev1)
                else:
                    ev = 3.0
            else:
                if self.line_search_method is 'Parabola':
                    ev = 100.0 * alpha
                else:
                    ev = 3.0
            alpha, phi_0, der_phi_0, g_0 = \
                self.line_search.step_length_update(self.A_s, P_s,
                                                    n_dim,
                                                    phi_0=phi_0,
                                                    der_phi_0=
                                                    der_phi_0,
                                                    phi_old=phi_old_2,
                                                    der_phi_old=
                                                    der_phi_old_2,
                                                    alpha_max=ev,
                                                    alpha_old=alpha)
            if self.line_search_method is 'SWC':
                if self.wfs.gd.comm.size > 1:
                    alpha_phi_der_phi = np.array([alpha, phi_0,
                                                  der_phi_0])
                    self.wfs.gd.comm.broadcast(alpha_phi_der_phi, 0)
                    alpha = alpha_phi_der_phi[0]
                    phi_0 = alpha_phi_der_phi[1]
                    der_phi_0 = alpha_phi_der_phi[2]
                    self.timer.start('ODD broadcast gradients')
                    for kpt in self.wfs.kpt_u:
                        k = self.n_kps * kpt.s + kpt.q
                        g = g_0[k].copy()
                        self.wfs.gd.comm.broadcast(g, 0)
                        g_0[k] = g.copy()
                    self.timer.stop('ODD broadcast gradients')

            if self.line_search_method is 'Parabola':
                if alpha > ev or alpha < 0.0:
                    alpha = ev
                else:
                    ev = 1.1 * alpha
            phi_old_2 = phi_old
            der_phi_old_2 = der_phi_old

            if alpha > 1.0e-10:
                if change_to_swc:
                    self.line_search = \
                        StrongWolfeConditions(
                            self.evaluate_phi_and_der_phi,
                            self.log, method=self.method,
                            awc=self.awc,
                            max_iter=self.max_iter_line_search
                            )
                    change_to_swc = False
                self.A_s = {s: self.A_s[s] + alpha * P_s[s]
                            for s in self.A_s.keys()}
                if self.reset_orbitals is True:
                    pass
                if (str(self.search_direction) == 'LBFGS_prec' or
                    str(self.search_direction) == 'LBFGS'):
                    if self.odd == 'PZ_SIC':
                        update_counter = 20
                    else:
                        update_counter = 20

                    if self.search_direction.k % update_counter == 0 \
                            and counter > 0:
                        for kpt in self.wfs.kpt_u:
                            k = self.n_kps * kpt.s + kpt.q
                            U = expm(self.A_s[k])
                            self.C_nM_init[k][:n_dim[k]] = \
                                np.dot(U.T, self.C_nM_init[k][
                                            :n_dim[k]])
                            self.A_s[k] = np.zeros_like(self.A_s[k])

                            if self.odd == 'Zero':
                                self.H_MM = \
                                    self.wfs.eigensolver.calculate_hamiltonian_matrix(
                                        self.ham,
                                        self.wfs,
                                        kpt)
                                # make matrix hermitian
                                ind_l = np.tril_indices(
                                    self.H_MM.shape[0], -1)
                                self.H_MM[(ind_l[1], ind_l[0])] = \
                                    self.H_MM[ind_l].conj()
                                C_nM_k = \
                                    self.pot.update_eigenval_2(self.C_nM_init[k],
                                                               kpt,
                                                               self.H_MM)
                                self.wfs.gd.comm.broadcast(C_nM_k, 0)
                                self.C_nM_init[k] = C_nM_k.copy()
                                kpt.C_nM = C_nM_k.copy()

                        if self.odd == 'Zero':
                            self.occ.calculate(self.wfs)
                            phi_0, der_phi_0, g_0 = \
                                self.evaluate_phi_and_der_phi(self.A_s,
                                                              P_s, n_dim,
                                                              alpha=0.0,
                                                              phi=None,
                                                              G_s=None)
                            for kpt in self.wfs.kpt_u:
                                k = self.n_kps * kpt.s + kpt.q
                                g = g_0[k].copy()
                                self.wfs.gd.comm.broadcast(g, 0)
                                g_0[k] = g.copy()
                        if str(self.search_direction) == 'LBFGS_prec':
                            self.search_direction = \
                                LBFGSdirection_prec(self.wfs,
                                                    m=self.memory_lbfgs,
                                                    diag=self.diag_prec)
                        if str(self.search_direction) == 'LBFGS':
                            self.search_direction = LBFGS_new(self.wfs,
                                                              m=self.memory_lbfgs)
                g_max = np.array([])
                for kpt in self.wfs.kpt_u:
                    k = self.n_kps * kpt.s + kpt.q
                    g_max = np.append(g_max, np.append(g_0[k].imag,
                                                       g_0[k].real))
                g_max = np.max(np.absolute(g_max))
                g_max = self.wfs.world.max(g_max)
                counter += 1
                self.E_ks.append(self.e_ks)
                self.E_sic.append(self.total_sic)
                self.G_m.append(g_max)
                self.alpha_stars.append(alpha)
                self.potential_counter.append(self.get_en_and_grad_iters)

                self.log_f(self.log, counter, g_max,
                           self.e_ks, self.total_sic)
                if self.save_orbitals and (counter % 10 == 0):
                     self.save_coefficients()
                if self.check_forces is True:
                    e_current = self.e_ks + sum(self.sic_s.values())
                    e_current *= Hartree
                    f_current = self.calculate_forces_2()
                    de = np.abs(e_current - energy) / \
                         self.calc.get_number_of_electrons()
                    df = np.max(np.absolute(f_current - forces))
                    self.log('dE, dF = {:11.2e} {:11.2e}'.format(de,
                                                                 df))
                    self.log(flush=True)
                    if de < 1.0e-7 and \
                            df < 1.0e-4:
                        break
            else:
                break

        self.E_ks.append(self.e_ks)
        self.E_sic.append(self.total_sic)
        self.G_m.append(g_max)
        self.alpha_stars.append(alpha)
        self.potential_counter.append(self.get_en_and_grad_iters)
        self.log("\n")
        self.log("Minimisiation is done!")
        self.log("Number of calls energy"
                 " and derivative: {} \n".format(self.get_en_and_grad_iters))

        totmom_v, magmom_av = self.den.estimate_magnetic_moments()
        print_positions(self.atoms, self.log, magmom_av)
        self.log("Calculating eigenvalues ...")
        self.get_energy_and_gradients(self.A_s, n_dim)

        if self.save_orbitals:
            self.save_coefficients()

        for kpt in self.wfs.kpt_u:
            self.H_MM = \
                self.wfs.eigensolver.calculate_hamiltonian_matrix(
                    self.ham,
                    self.wfs,
                    kpt)

            # make matrix hermitian
            ind_l = np.tril_indices(self.H_MM.shape[0], -1)
            self.H_MM[(ind_l[1], ind_l[0])] = \
                self.H_MM[ind_l].conj()

            self.pot.update_eigenval(kpt.f_n, kpt.C_nM,
                                     kpt, self.wfs, self.setups,
                                     self.H_MM,
                                     occupied_only=self.occupied_only)
        if self.odd == 'Zero':
            self.occ.calculate(self.wfs)
        self.log("... done!\n")
        f_sn = {}
        for kpt in self.wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            f_sn[k] = np.copy(kpt.f_n)
        if self.wfs.kd.comm.size == 1 and self.odd != 'Zero':
            self.write_final_output(self.log, self.e_ks,
                       self.total_sic, self.sic_n,
                       self.pot.eigv_s, f_sn)
        else:
            dipole_v = self.den.calculate_dipole_moment() * Bohr
            self.log(
                'Dipole moment: ({0:.6f}, {1:.6f}, {2:.6f}) |e|*Ang\n'
                    .format(*dipole_v))
            if self.wfs.nspins == 2 or not self.den.collinear:
                totmom_v, magmom_av = self.den.estimate_magnetic_moments()
                self.log('Total magnetic moment: ({:.6f}, {:.6f}, {:.6f})'
                         .format(*totmom_v))
                self.log('Local magnetic moments:')
                symbols = self.atoms.get_chemical_symbols()
                for a, mom_v in enumerate(magmom_av):
                    self.log('{:4} {:2} ({:9.6f}, {:9.6f}, {:9.6f})'
                             .format(a, symbols[a], *mom_v))
            self.log(
                'SIC energy: %11.6f' % (self.total_sic * Hartree))
            self.log(flush=True)
            self.calc.summary()
        self.log(flush=True)

        return self.get_en_and_grad_iters, counter + 1.0,\
               g_max * Hartree, \
               (self.e_ks + sum(self.sic_s.values())) * Hartree

    def test_second_der(self):

        G_0 = self.get_energy_and_gradients(self.A_s, self.n_dim)[1]

        h = 1.0e-4

        A = self.A_s[0][0][1]
        self.A_s[0][0][1] = A + h
        self.A_s[0][1][0] = -np.conjugate(A + h)

        E_p, G_p = self.get_energy_and_gradients(self.A_s, self.n_dim)

        self.A_s[0][0][1] = A - h
        self.A_s[0][1][0] = -np.conjugate(A - h)

        E_m, G_m = self.get_energy_and_gradients(self.A_s, self.n_dim)

        self.A_s[0][0][1] = A
        self.A_s[0][1][0] = -np.conjugate(A)

        G_01_02 = (G_p[0][0][2] - G_m[0][0][2])/(2.0 * abs(h))

        print(G_0[0][0][1], (E_p - E_m) / (2.0 * abs(h)))

        # now G_02_01
        A = self.A_s[0][0][2]
        self.A_s[0][0][2] = A + h
        self.A_s[0][2][0] = -np.conjugate(A + h)

        G_p = self.get_energy_and_gradients(self.A_s, self.n_dim)[1]

        self.A_s[0][0][2] = A - h
        self.A_s[0][2][0] = -np.conjugate(A - h)

        G_m = self.get_energy_and_gradients(self.A_s, self.n_dim)[1]

        self.A_s[0][0][2] = A
        self.A_s[0][2][0] = -np.conjugate(A)

        G_02_01 = (G_p[0][0][1] - G_m[0][0][1])/(2.0 * abs(h))

        print(G_01_02, G_02_01)


