from __future__ import division
import numpy as np
from ase.utils.timing import timer

from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.lfc import BasisFunctions
from gpaw.matrix import (Matrix, matrix_matrix_multiply as mmm,
                         suggest_blocking)
#from gpaw.overlap import Overlap
from gpaw.utilities import unpack
from gpaw.utilities.timing import nulltimer
from gpaw.wavefunctions.base import WaveFunctions
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class FDPWWaveFunctions(WaveFunctions):
    """Base class for finite-difference and planewave classes."""
    def __init__(self, parallel, initksl, *args, **kwargs):
        WaveFunctions.__init__(self, *args, **kwargs)

        self.scalapack_parameters = parallel['sl_diagonalize']
        if self.scalapack_parameters == 'auto':
            self.scalapack_parameters = suggest_blocking(self.bd.nbands,
                                                         self.bd.comm.size)
        elif self.scalapack_parameters is None:
            self.scalapack_parameters = (1, 1, None)

        self.initksl = initksl

        self.set_orthonormalized(False)

        #self.overlap = self.make_overlap()

        self._work_matrix_nn = None  # storage for H, S, ...
        self._work_array = None

    @property
    def work_array(self):
        if self._work_array is None:
            self._work_array = self.empty(self.bd.mynbands)
        return self._work_array

    @property
    def work_matrix_nn(self):
        """Get Matrix object for H, S, ..."""
        if self._work_matrix_nn is None:
            self._work_matrix_nn = Matrix(
                self.bd.nbands, self.bd.nbands,
                self.dtype,
                dist=(self.bd.comm, self.bd.comm.size))
        return self._work_matrix_nn

    def __str__(self):
        s = 'Diagonalizer layout: ??????????????????'
        s += 'Orthonormalizer layout: ????????'
        return WaveFunctions.__str__(self) + s

    def set_setups(self, setups):
        WaveFunctions.set_setups(self, setups)

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    def set_positions(self, spos_ac, atom_partition=None):
        WaveFunctions.set_positions(self, spos_ac, atom_partition)
        self.set_orthonormalized(False)
        self.pt.set_positions(spos_ac)
        self.allocate_arrays_for_projections(self.pt.my_atom_indices)
        self.positions_set = True

    def make_overlap(self):
        return Overlap(self.timer)

    def initialize(self, density, hamiltonian, spos_ac):
        """Initialize wave-functions, density and hamiltonian.

        Return (nlcao, nrand) tuple with number of bands intialized from
        LCAO and random numbers, respectively."""

        if self.mykpts[0].psit is None:
            basis_functions = BasisFunctions(self.gd,
                                             [setup.phit_j
                                              for setup in self.setups],
                                             self.kd, dtype=self.dtype,
                                             cut=True)
            basis_functions.set_positions(spos_ac)
        else:
            self.initialize_wave_functions_from_restart_file()

        if self.mykpts[0].psit is not None:
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

        if self.mykpts[0].psit is None:
            nlcao = self.initialize_wave_functions_from_basis_functions(
                basis_functions, density, hamiltonian, spos_ac)
            nrand = self.bd.nbands - nlcao
        else:
            # We got everything from file:
            nlcao = 0
            nrand = 0

        return nlcao, nrand

    def initialize_wave_functions_from_restart_file(self):
        for kpt in self.mykpts:
            if not kpt.psit.in_memory:
                kpt.psit.read_from_file()

    def initialize_wave_functions_from_basis_functions(self,
                                                       basis_functions,
                                                       density, hamiltonian,
                                                       spos_ac):
        # if self.initksl is None:
        #     raise RuntimeError('use fewer bands or more basis functions')

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

        return lcaobd.nbands

    @timer('Orthonormalize')
    def orthonormalize(self, kpt=None):
        if kpt is None:
            for kpt in self.mykpts:
                self.orthonormalize(kpt)
            self.orthonormalized = True
            return

        psit = kpt.psit
        P = kpt.P

        with self.timer('projections'):
            psit.matrix_elements(self.pt, out=P)

        S = self.work_matrix_nn
        P2 = P.new()

        with self.timer('calc_s_matrix'):
            psit.matrix_elements(psit, S, symmetric=True, cc=True)
            self.setups.dS.apply(P, out=P2)
            mmm(1.0, P, 'N', P2, 'C', 1.0, S, symmetric=True)

        with self.timer('inverse-cholesky'):
            S.invcholesky()
            # S_nn now contains the inverse of the Cholesky factorization

        psit2 = psit.new(buf=self.work_array)
        with self.timer('rotate_psi_s'):
            mmm(1.0, S, 'N', psit, 'N', 0.0, psit2)
            mmm(1.0, S, 'N', P, 'N', 0.0, P2)
            psit[:] = psit2
            kpt.P = P2

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
