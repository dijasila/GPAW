from __future__ import division
import numpy as np

from ase.utils.timing import timer

from gpaw import extra_parameters
from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.lfc import BasisFunctions
from gpaw.matrix import Matrix, PAWMatrix
from gpaw.overlap import Overlap
from gpaw.utilities import unpack
from gpaw.utilities.timing import nulltimer
from gpaw.wavefunctions.base import WaveFunctions
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class FDPWWaveFunctions(WaveFunctions):
    """Base class for finite-difference and planewave classes."""
    def __init__(self, initksl, *args, **kwargs):
        WaveFunctions.__init__(self, *args, **kwargs)

        self.initksl = initksl

        self.orthonormalized = False

        self.overlap = self.make_overlap()

        self._M_nn = None  # storage for H, S, ...
        self._work_array_nG = None

    @property
    def work_array_nG(self):
        if self._work_array_nG is None:
            self._work_array_nG = self.empty(self.bd.mynbands)
        return self._work_array_nG

    @property
    def M_nn(self):
        """Get Matrix object for H, S, ..."""
        if self._M_nn is None:
            self._M_nn = Matrix(self.bd.nbands, self.bd.nbands, self.dtype,
                                dist=(self.bd.comm, self.bd.comm.size))
        return self._M_nn

    def set_setups(self, setups):
        WaveFunctions.set_setups(self, setups)

    def set_positions(self, spos_ac, atom_partition=None):
        WaveFunctions.set_positions(self, spos_ac, atom_partition)
        self.orthonormalized = False
        self.pt.set_positions(spos_ac)
        self.allocate_arrays_for_projections(self.pt.my_atom_indices)
        self.positions_set = True

    def make_overlap(self):
        return Overlap(self.timer)

    def initialize(self, density, hamiltonian, spos_ac):
        if self.kpt_u[0].psit_nG is None:
            basis_functions = BasisFunctions(self.gd,
                                             [setup.phit_j
                                              for setup in self.setups],
                                             self.kd, dtype=self.dtype,
                                             cut=True)
            basis_functions.set_positions(spos_ac)
        elif not isinstance(self.kpt_u[0].psit_nG, np.ndarray):
            self.initialize_wave_functions_from_restart_file()

        if self.kpt_u[0].psit_nG is not None:
            density.initialize_from_wavefunctions(self)
        elif density.nt_sG is None:
            density.initialize_from_atomic_densities(basis_functions)
            # Initialize GLLB-potential from basis function orbitals
            if hamiltonian.xc.type == 'GLLB':
                hamiltonian.xc.initialize_from_atomic_orbitals(
                    basis_functions)
        else:  # XXX???
            # We didn't even touch density, but some combinations in paw.set()
            # will make it necessary to do this for some reason.
            density.calculate_normalized_charges_and_mix()
        hamiltonian.update(density)

        if self.kpt_u[0].psit_nG is None:
            self.initialize_wave_functions_from_basis_functions(
                basis_functions, density, hamiltonian, spos_ac)

    def initialize_wave_functions_from_basis_functions(self,
                                                       basis_functions,
                                                       density, hamiltonian,
                                                       spos_ac):
        if self.initksl is None:
            raise RuntimeError('use fewer bands or more basis functions')

        self.timer.start('LCAO initialization')
        lcaoksl, lcaobd = self.initksl, self.initksl.bd
        lcaowfs = LCAOWaveFunctions(lcaoksl, self.gd, self.nvalence,
                                    self.setups, lcaobd, self.dtype,
                                    self.world, self.kd, self.kptband_comm,
                                    nulltimer)
        lcaowfs.basis_functions = basis_functions
        lcaowfs.timer = self.timer
        self.timer.start('Set positions (LCAO WFS)')
        lcaowfs.set_positions(spos_ac, self.atom_partition)
        self.timer.stop('Set positions (LCAO WFS)')

        eigensolver = DirectLCAO()
        eigensolver.initialize(self.gd, self.dtype, self.setups.nao, lcaoksl)

        # XXX when density matrix is properly distributed, be sure to
        # update the density here also
        eigensolver.iterate(hamiltonian, lcaowfs)

        # Transfer coefficients ...
        for kpt, lcaokpt in zip(self.kpt_u, lcaowfs.kpt_u):
            kpt.C_nM = lcaokpt.C_nM

        # and get rid of potentially big arrays early:
        del eigensolver, lcaowfs

        self.timer.start('LCAO to grid')
        self.initialize_from_lcao_coefficients(basis_functions,
                                               lcaobd.mynbands)
        self.timer.stop('LCAO to grid')

        if self.bd.mynbands > lcaobd.mynbands:
            # Add extra states.  If the number of atomic orbitals is
            # less than the desired number of bands, then extra random
            # wave functions are added.
            self.random_wave_functions(lcaobd.mynbands)
        self.timer.stop('LCAO initialization')

    def initialize_wave_functions_from_restart_file(self):
        if isinstance(self.kpt_u[0].psit_nG, np.ndarray):
            return

        # Calculation started from a restart file.  Copy data
        # from the file to memory:
        for kpt in self.kpt_u:
            file_nG = kpt.psit_nG
            kpt.psit_nG = self.empty(self.bd.mynbands, q=kpt.q)
            if extra_parameters.get('sic'):
                kpt.W_nn = np.zeros((self.bd.nbands, self.bd.nbands),
                                    dtype=self.dtype)
            # Read band by band to save memory
            for n, psit_G in enumerate(kpt.psit_nG):
                if self.gd.comm.rank == 0:
                    big_psit_G = file_nG[n][:].astype(psit_G.dtype)
                else:
                    big_psit_G = None
                self.gd.distribute(big_psit_G, psit_G)

    @timer('Orthonormalize')
    def orthonormalize(self, kpt=None):
        if kpt is None:
            for kpt in self.kpt_u:
                self.orthonormalize(kpt)
            self.orthonormalized = True
            return

        self.wrap_wave_function_arrays_in_fancy_objects()

        psit_n = kpt.psit_n
        P_nI = kpt.P_n

        with self.timer('projections'):
            psit_n.project(self.pt, P_nI)

        dS_II = PAWMatrix(self.setups[a].dO_ii for a in kpt.P_ani)

        S_nn = self.M_nn
        dSP_nI = P_nI.new()

        with self.timer('calc_s_matrix'):
            psit_n.matrix_elements(psit_n, S_nn)
            dSP_nI[:] = P_nI * dS_II
            S_nn += P_nI.C * dSP_nI.T

        with self.timer('inverse-cholesky'):
            assert self.bd.comm.size == 1
            S_nn.cholesky()
            S_nn.inv()
            # S_nn now contains the inverse of the Cholesky factorization

        psit2_n = psit_n.new(buf=self.work_array_nG)
        with self.timer('rotate_psi_s'):
            psit2_n[:] = S_nn.T * psit_n
            psit_n[:] = psit2_n
            dSP_nI[:] = S_nn.T * P_nI
            dSP_nI.extract_to(kpt.P_ani)

    def calculate_forces(self, hamiltonian, F_av):
        # Calculate force-contribution from k-points:
        F_av.fill(0.0)
        F_aniv = self.pt.dict(self.bd.mynbands, derivative=True)
        dH_asp = hamiltonian.dH_asp
        for kpt in self.kpt_u:
            self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)
            for a, F_niv in F_aniv.items():
                F_niv = F_niv.conj()
                F_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
                dH_ii = unpack(dH_asp[a][kpt.s])
                P_ni = kpt.P_ani[a]
                F_vii = np.dot(np.dot(F_niv.transpose(), P_ni), dH_ii)
                F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                dO_ii = self.setups[a].dO_ii
                F_vii -= np.dot(np.dot(F_niv.transpose(), P_ni), dO_ii)
                F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

            # Hack used in delta-scf calculations:
            if hasattr(kpt, 'c_on'):
                assert self.bd.comm.size == 1
                self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)  # XXX again
                d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                dtype=complex)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    d_nn += ne * np.outer(c_n.conj(), c_n)
                for a, F_niv in F_aniv.items():
                    F_niv = F_niv.conj()
                    dH_ii = unpack(dH_asp[a][kpt.s])
                    Q_ni = np.dot(d_nn, kpt.P_ani[a])
                    F_vii = np.dot(np.dot(F_niv.transpose(), Q_ni), dH_ii)
                    F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                    dO_ii = self.setups[a].dO_ii
                    F_vii -= np.dot(np.dot(F_niv.transpose(), Q_ni), dO_ii)
                    F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

        self.bd.comm.sum(F_av, 0)

        if self.bd.comm.rank == 0:
            self.kd.comm.sum(F_av, 0)

    def estimate_memory(self, mem):
        gridbytes = self.bytes_per_wave_function()
        n = len(self.kpt_u) * self.bd.mynbands
        mem.subnode('Arrays psit_nG', n * gridbytes)
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'), self)
        ni = sum(dataset.ni for dataset in self.setups) / self.gd.comm.size
        mem.subnode('Projections', n * ni * np.dtype(self.dtype).itemsize)
        self.pt.estimate_memory(mem.subnode('Projectors'))
        self.matrixoperator.estimate_memory(mem.subnode('Overlap op'),
                                            self.dtype)
