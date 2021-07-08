import numpy as np
from ase.units import Bohr

from gpaw.lfc import BasisFunctions
from gpaw.utilities import unpack
from gpaw.utilities.tools import tri2full
# from gpaw import debug
# from gpaw.lcao.overlap import NewTwoCenterIntegrals as NewTCI
from gpaw.lcao.tci import TCIExpansions
from gpaw.utilities.blas import gemm, gemmdot
from gpaw.wavefunctions.base import WaveFunctions
from gpaw.lcao.atomic_correction import (DenseAtomicCorrection,
                                         SparseAtomicCorrection)
from gpaw.wavefunctions.mode import Mode


class LCAO(Mode):
    name = 'lcao'

    def __init__(self, atomic_correction=None, interpolation=3,
                 force_complex_dtype=False):
        self.atomic_correction = atomic_correction
        self.interpolation = interpolation
        Mode.__init__(self, force_complex_dtype)

    def __call__(self, *args, **kwargs):
        return LCAOWaveFunctions(*args,
                                 atomic_correction=self.atomic_correction,
                                 **kwargs)

    def __repr__(self):
        return 'LCAO({})'.format(self.todict())

    def todict(self):
        dct = Mode.todict(self)
        dct['interpolation'] = self.interpolation
        return dct


def update_phases(C_unM, q_u, ibzk_qc, spos_ac, oldspos_ac, setups, Mstart):
    """Complex-rotate coefficients compensating discontinuous phase shift.

    This changes the coefficients to counteract the phase discontinuity
    of overlaps when atoms move across a cell boundary."""

    # We don't want to apply any phase shift unless we crossed a cell
    # boundary.  So we round the shift to either 0 or 1.
    #
    # Example: spos_ac goes from 0.01 to 0.99 -- this rounds to 1 and
    # we apply the phase.  If someone moves an atom by half a cell
    # without crossing a boundary, then we are out of luck.  But they
    # should have reinitialized from LCAO anyway.
    phase_qa = np.exp(2j * np.pi *
                      np.dot(ibzk_qc, (spos_ac - oldspos_ac).T.round()))

    for q, C_nM in zip(q_u, C_unM):
        if C_nM is None:
            continue
        for a in range(len(spos_ac)):
            M1 = setups.M_a[a] - Mstart
            M2 = M1 + setups[a].nao
            M1 = max(0, M1)
            C_nM[:, M1:M2] *= phase_qa[q, a]  # (may truncate M2)


# replace by class to make data structure perhaps a bit less confusing
def get_r_and_offsets(nl, spos_ac, cell_cv):
    r_and_offset_aao = {}

    def add(a1, a2, R_c, offset):
        if not (a1, a2) in r_and_offset_aao:
            r_and_offset_aao[(a1, a2)] = []
        r_and_offset_aao[(a1, a2)].append((R_c, offset))

    for a1, spos1_c in enumerate(spos_ac):
        a2_a, offsets = nl.get_neighbors(a1)
        for a2, offset in zip(a2_a, offsets):
            spos2_c = spos_ac[a2] + offset

            R_c = np.dot(spos2_c - spos1_c, cell_cv)
            add(a1, a2, R_c, offset)
            if a1 != a2 or offset.any():
                add(a2, a1, -R_c, -offset)

    return r_and_offset_aao


class LCAOWaveFunctions(WaveFunctions):
    mode = 'lcao'

    def __init__(self, ksl, gd, nvalence, setups, bd,
                 dtype, world, kd, kptband_comm, timer,
                 atomic_correction=None, collinear=True):
        WaveFunctions.__init__(self, gd, nvalence, setups, bd,
                               dtype, collinear, world, kd,
                               kptband_comm, timer)
        self.ksl = ksl
        self.S_qMM = None
        self.T_qMM = None
        self.P_aqMi = None
        self.debug_tci = False

        if atomic_correction is None:
            atomic_correction = 'sparse' if ksl.using_blacs else 'dense'

        if atomic_correction == 'sparse':
            self.atomic_correction_cls = SparseAtomicCorrection
        else:
            assert atomic_correction == 'dense'
            self.atomic_correction_cls = DenseAtomicCorrection

        # self.tci = NewTCI(gd.cell_cv, gd.pbc_c, setups, kd.ibzk_qc, kd.gamma)
        with self.timer('TCI: Evaluate splines'):
            self.tciexpansions = TCIExpansions.new_from_setups(setups)

        self.basis_functions = BasisFunctions(gd,
                                              [setup.phit_j
                                               for setup in setups],
                                              kd,
                                              dtype=dtype,
                                              cut=True)

    def set_orthonormalized(self, o):
        pass

    def empty(self, n=(), global_array=False, realspace=False):
        if realspace:
            return self.gd.empty(n, self.dtype, global_array)
        else:
            if isinstance(n, int):
                n = (n,)
            nao = self.setups.nao
            return np.empty(n + (nao,), self.dtype)

    def __str__(self):
        s = 'Wave functions: LCAO\n'
        s += '  Diagonalizer: %s\n' % self.ksl.get_description()
        s += ('  Atomic Correction: %s\n'
              % self.atomic_correction_cls.description)
        s += '  Datatype: %s\n' % self.dtype.__name__
        return s

    def set_eigensolver(self, eigensolver):
        WaveFunctions.set_eigensolver(self, eigensolver)
        if eigensolver:
            eigensolver.initialize(self.gd, self.dtype, self.setups.nao,
                                   self.ksl)

    def set_positions(self, spos_ac, atom_partition=None, move_wfs=False):
        oldspos_ac = self.spos_ac
        with self.timer('Basic WFS set positions'):
            WaveFunctions.set_positions(self, spos_ac, atom_partition)

        with self.timer('Basis functions set positions'):
            self.basis_functions.set_positions(spos_ac)

        if self.ksl is not None:
            self.basis_functions.set_matrix_distribution(self.ksl.Mstart,
                                                         self.ksl.Mstop)

        nq = len(self.kd.ibzk_qc)
        nao = self.setups.nao
        mynbands = self.bd.mynbands
        Mstop = self.ksl.Mstop
        Mstart = self.ksl.Mstart
        mynao = Mstop - Mstart

        # if self.ksl.using_blacs:  # XXX
        #     S and T have been distributed to a layout with blacs, so
        #     discard them to force reallocation from scratch.
        #
        #     TODO: evaluate S and T when they *are* distributed, thus saving
        #     memory and avoiding this problem
        for kpt in self.kpt_u:
            kpt.S_MM = None
            kpt.T_MM = None

        # Free memory in case of old matrices:
        self.S_qMM = self.T_qMM = self.P_aqMi = None

        if self.dtype == complex and oldspos_ac is not None:
            update_phases([kpt.C_nM for kpt in self.kpt_u],
                          [kpt.q for kpt in self.kpt_u],
                          self.kd.ibzk_qc, spos_ac, oldspos_ac,
                          self.setups, Mstart)

        for kpt in self.kpt_u:
            if kpt.C_nM is None:
                kpt.C_nM = np.empty((mynbands, nao), self.dtype)

        if 0:  # self.debug_tci:
            # if self.ksl.using_blacs:
            #     self.tci.set_matrix_distribution(Mstart, mynao)
            oldS_qMM = np.empty((nq, mynao, nao), self.dtype)
            oldT_qMM = np.empty((nq, mynao, nao), self.dtype)

            oldP_aqMi = {}
            for a in self.basis_functions.my_atom_indices:
                ni = self.setups[a].ni
                oldP_aqMi[a] = np.empty((nq, nao, ni), self.dtype)

            # Calculate lower triangle of S and T matrices:
            self.timer.start('tci calculate')
            # self.tci.calculate(spos_ac, oldS_qMM, oldT_qMM,
            #                   oldP_aqMi)
            self.timer.stop('tci calculate')

        self.timer.start('mktci')
        manytci = self.tciexpansions.get_manytci_calculator(
            self.setups, self.gd, spos_ac, self.kd.ibzk_qc, self.dtype,
            self.timer)
        self.timer.stop('mktci')
        self.manytci = manytci
        self.newtci = manytci.tci

        my_atom_indices = self.basis_functions.my_atom_indices
        self.timer.start('ST tci')
        newS_qMM, newT_qMM = manytci.O_qMM_T_qMM(self.gd.comm,
                                                 Mstart, Mstop,
                                                 self.ksl.using_blacs)
        self.timer.stop('ST tci')
        self.timer.start('P tci')
        P_qIM = manytci.P_qIM(my_atom_indices)
        self.timer.stop('P tci')
        self.P_aqMi = newP_aqMi = manytci.P_aqMi(my_atom_indices)
        self.P_qIM = P_qIM  # XXX atomic correction

        self.atomic_correction = self.atomic_correction_cls.new_from_wfs(self)

        # TODO
        #   OK complex/conj, periodic images
        #   OK scalapack
        #   derivatives/forces
        #   sparse
        #   use symmetry/conj tricks to reduce calculations
        #   enable caching of spherical harmonics

        # if self.atomic_correction.name != 'dense':
        # from gpaw.lcao.newoverlap import newoverlap
        # self.P_neighbors_a, self.P_aaqim = newoverlap(self, spos_ac)

        # if self.atomic_correction.name == 'scipy':
        #    Pold_qIM = self.atomic_correction.Psparse_qIM
        #    for q in range(nq):
        #        maxerr = abs(Pold_qIM[q] - P_qIM[q]).max()
        #        print('sparse maxerr', maxerr)
        #        assert maxerr == 0

        self.atomic_correction.add_overlap_correction(newS_qMM)
        if self.debug_tci:
            self.atomic_correction.add_overlap_correction(oldS_qMM)

        self.allocate_arrays_for_projections(my_atom_indices)

        # S_MM = None  # allow garbage collection of old S_qMM after redist
        if self.debug_tci:
            oldS_qMM = self.ksl.distribute_overlap_matrix(oldS_qMM, root=-1)
            oldT_qMM = self.ksl.distribute_overlap_matrix(oldT_qMM, root=-1)

        newS_qMM = self.ksl.distribute_overlap_matrix(newS_qMM, root=-1)
        newT_qMM = self.ksl.distribute_overlap_matrix(newT_qMM, root=-1)

        # if (debug and self.bd.comm.size == 1 and self.gd.comm.rank == 0 and
        #     nao > 0 and not self.ksl.using_blacs):
        #     S and T are summed only on comm master, so check only there
        #     from numpy.linalg import eigvalsh
        #     self.timer.start('Check positive definiteness')
        #     for S_MM in S_qMM:
        #         tri2full(S_MM, UL='L')
        #         smin = eigvalsh(S_MM).real.min()
        #         if smin < 0:
        #             raise RuntimeError('Overlap matrix has negative '
        #                               'eigenvalue: %e' % smin)
        #     self.timer.stop('Check positive definiteness')
        self.positions_set = True

        if self.debug_tci:
            Serr = np.abs(newS_qMM - oldS_qMM).max()
            Terr = np.abs(newT_qMM - oldT_qMM).max()
            print('S maxerr', Serr)
            print('T maxerr', Terr)
            try:
                assert Terr < 1e-15, Terr
            except AssertionError:
                np.set_printoptions(precision=6)
                if self.world.rank == 0:
                    print(newT_qMM)
                    print(oldT_qMM)
                    print(newT_qMM - oldT_qMM)
                raise
            assert Serr < 1e-15, Serr

            assert len(oldP_aqMi) == len(newP_aqMi)
            for a in oldP_aqMi:
                Perr = np.abs(oldP_aqMi[a] - newP_aqMi[a]).max()
                assert Perr < 1e-15, (a, Perr)

        for kpt in self.kpt_u:
            q = kpt.q
            kpt.S_MM = newS_qMM[q]
            kpt.T_MM = newT_qMM[q]
        self.S_qMM = newS_qMM
        self.T_qMM = newT_qMM

        # Elpa wants to reuse the decomposed form of S_qMM.
        # We need to keep track of the existence of that object here,
        # since this is where we change S_qMM.  Hence, expect this to
        # become arrays after the first diagonalization:
        self.decomposed_S_qMM = [None] * len(self.S_qMM)

    def initialize(self, density, hamiltonian, spos_ac):
        # Note: The above line exists also in set_positions.
        # This is guaranteed to be correct, but we can probably remove one.
        # Of course no human can understand the initialization process,
        # so this will be some other day.
        self.timer.start('LCAO WFS Initialize')
        if density.nt_sG is None:
            if self.kpt_u[0].f_n is None or self.kpt_u[0].C_nM is None:
                density.initialize_from_atomic_densities(self.basis_functions)
            else:
                # We have the info we need for a density matrix, so initialize
                # from that instead of from scratch.  This will be the case
                # after set_positions() during a relaxation
                density.initialize_from_wavefunctions(self)
            # Initialize GLLB-potential from basis function orbitals
            if hamiltonian.xc.type == 'GLLB':
                hamiltonian.xc.initialize_from_atomic_orbitals(
                    self.basis_functions)

        else:
            # After a restart, nt_sg doesn't exist yet, so we'll have to
            # make sure it does.  Of course, this should have been taken care
            # of already by this time, so we should improve the code elsewhere
            density.calculate_normalized_charges_and_mix()

        hamiltonian.update(density)
        self.timer.stop('LCAO WFS Initialize')

        return 0, 0

    def initialize_wave_functions_from_lcao(self):
        """Fill the calc.wfs.kpt_[u].psit_nG arrays with useful data.

        Normally psit_nG is NOT used in lcao mode, but some extensions
        (like ase.dft.wannier) want to have it.
        This code is adapted from fd.py / initialize_from_lcao_coefficients()
        and fills psit_nG with data constructed from the current lcao
        coefficients (kpt.C_nM).

        (This may or may not work in band-parallel case!)
        """
        from gpaw.wavefunctions.arrays import UniformGridWaveFunctions
        bfs = self.basis_functions
        for kpt in self.kpt_u:
            kpt.psit = UniformGridWaveFunctions(
                self.bd.nbands, self.gd, self.dtype, kpt=kpt.q, dist=None,
                spin=kpt.s, collinear=True)
            kpt.psit_nG[:] = 0.0
            bfs.lcao_to_grid(kpt.C_nM, kpt.psit_nG[:self.bd.mynbands], kpt.q)

    def initialize_wave_functions_from_restart_file(self):
        """Dummy function to ensure compatibility to fd mode"""
        self.initialize_wave_functions_from_lcao()

    def add_orbital_density(self, nt_G, kpt, n):
        rank, q = self.kd.get_rank_and_index(kpt.k)
        u = q * self.nspins + kpt.s
        assert rank == self.kd.comm.rank
        assert self.kpt_u[u] is kpt
        psit_G = self._get_wave_function_array(u, n, realspace=True)
        self.add_realspace_orbital_to_density(nt_G, psit_G)

    def calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        self.timer.start('Calculate density matrix')
        rho_MM = self.ksl.calculate_density_matrix(f_n, C_nM, rho_MM)
        self.timer.stop('Calculate density matrix')
        return rho_MM

        if 1:
            # XXX Should not conjugate, but call gemm(..., 'c')
            # Although that requires knowing C_Mn and not C_nM.
            # that also conforms better to the usual conventions in literature
            Cf_Mn = C_nM.T.conj() * f_n
            self.timer.start('gemm')
            gemm(1.0, C_nM, Cf_Mn, 0.0, rho_MM, 'n')
            self.timer.stop('gemm')
            self.timer.start('band comm sum')
            self.bd.comm.sum(rho_MM)
            self.timer.stop('band comm sum')
        else:
            # Alternative suggestion. Might be faster. Someone should test this
            from gpaw.utilities.blas import r2k
            C_Mn = C_nM.T.copy()
            r2k(0.5, C_Mn, f_n * C_Mn, 0.0, rho_MM)
            tri2full(rho_MM)

    def calculate_atomic_density_matrices_with_occupation(self, D_asp, f_un):
        # ac = self.atomic_correction
        # if ac.implements_distributed_projections():
        #     D2_asp = ac.redistribute(self, D_asp, type='asp', op='forth')
        #     WaveFunctions.calculate_atomic_density_matrices_with_occupation(
        #         self, D2_asp, f_un)
        #     D3_asp = ac.redistribute(self, D2_asp, type='asp', op='back')
        #     for a in D_asp:
        #         D_asp[a][:] = D3_asp[a]
        # else:
        WaveFunctions.calculate_atomic_density_matrices_with_occupation(
            self, D_asp, f_un)

    def calculate_density_matrix_delta(self, d_nn, C_nM, rho_MM=None):
        self.timer.start('Calculate density matrix')
        rho_MM = self.ksl.calculate_density_matrix_delta(d_nn, C_nM, rho_MM)
        self.timer.stop('Calculate density matrix')
        return rho_MM

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        # Custom occupations are used in calculation of response potential
        # with GLLB-potential
        if kpt.rho_MM is None:
            rho_MM = self.calculate_density_matrix(f_n, kpt.C_nM)
            if hasattr(kpt, 'c_on'):
                assert self.bd.comm.size == 1
                d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                dtype=kpt.C_nM.dtype)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    assert abs(c_n.imag).max() < 1e-14
                    d_nn += ne * np.outer(c_n.conj(), c_n).real
                rho_MM += self.calculate_density_matrix_delta(d_nn, kpt.C_nM)
        else:
            rho_MM = kpt.rho_MM
        self.timer.start('Construct density')
        self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.q)
        self.timer.stop('Construct density')

    def add_to_kinetic_density_from_k_point(self, taut_G, kpt):
        raise NotImplementedError('Kinetic density calculation for LCAO '
                                  'wavefunctions is not implemented.')

    def calculate_forces(self, hamiltonian, F_av):
        self.timer.start('LCAO forces')

        Fref_av=np.zeros_like(F_av)
        FORCES=LCAO_forces(self.ksl,self.dtype,self.gd,self.kpt_u,
                           self.basis_functions,self.newtci,self.P_aqMi,self.setups,
                           self.manytci,hamiltonian,self.spos_ac,self.timer,Fref_av)

        F_av[:,:]=FORCES.get_forces_sum_GS()

        self.timer.stop('LCAO forces')


    def _get_wave_function_array(self, u, n, realspace=True, periodic=False):
        # XXX Taking kpt is better than taking u
        kpt = self.kpt_u[u]
        C_M = kpt.C_nM[n]

        if realspace:
            psit_G = self.gd.zeros(dtype=self.dtype)
            self.basis_functions.lcao_to_grid(C_M, psit_G, kpt.q)
            if periodic and self.dtype == complex:
                k_c = self.kd.ibzk_kc[kpt.k]
                return self.gd.plane_wave(-k_c) * psit_G
            return psit_G
        else:
            return C_M

    def write(self, writer, write_wave_functions=False):
        WaveFunctions.write(self, writer)
        if write_wave_functions:
            self.write_wave_functions(writer)

    def write_wave_functions(self, writer):
        writer.add_array(
            'coefficients',
            (self.nspins, self.kd.nibzkpts, self.bd.nbands, self.setups.nao),
            dtype=self.dtype)
        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                C_nM = self.collect_array('C_nM', k, s)
                writer.fill(C_nM * Bohr**-1.5)

    def read(self, reader):
        WaveFunctions.read(self, reader)
        r = reader.wave_functions
        if 'coefficients' in r:
            self.read_wave_functions(r)

    def read_wave_functions(self, reader):
        for kpt in self.kpt_u:
            C_nM = reader.proxy('coefficients', kpt.s, kpt.k)
            kpt.C_nM = self.bd.empty(self.setups.nao, dtype=self.dtype)
            for myn, C_M in enumerate(kpt.C_nM):
                n = self.bd.global_index(myn)
                # XXX number of bands could have been rounded up!
                if n >= len(C_nM):
                    break
                C_M[:] = C_nM[n] * Bohr**1.5

    def estimate_memory(self, mem):
        nq = len(self.kd.ibzk_qc)
        nao = self.setups.nao
        ni_total = sum([setup.ni for setup in self.setups])
        itemsize = mem.itemsize[self.dtype]
        mem.subnode('C [qnM]', nq * self.bd.mynbands * nao * itemsize)
        nM1, nM2 = self.ksl.get_overlap_matrix_shape()
        mem.subnode('S, T [2 x qmm]', 2 * nq * nM1 * nM2 * itemsize)
        mem.subnode('P [aqMi]', nq * nao * ni_total // self.gd.comm.size)
        # self.tci.estimate_memory(mem.subnode('TCI'))
        self.basis_functions.estimate_memory(mem.subnode('BasisFunctions'))
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'),
                                         self.dtype)


class LCAO_forces:

    def __init__(self,ksl,dtype,gd,kpt_u,bfs,newtci,P_aqMi,
                 setups,manytci,hamiltonian,spos_ac,
                 timer,Fref_av):
        """    Calculate forces for LCAO       """

        self.ksl=ksl
        self.nao = ksl.nao
        self.mynao = ksl.mynao  
        self.dtype = dtype
        self.newtci = newtci
        self.manytci = manytci
        self.P_aqMi=P_aqMi
        self.gd = gd
        self.kpt_u = kpt_u
        self.bfs = bfs
        self.spos_ac=spos_ac
        self.Mstart = ksl.Mstart
        self.Mstop = ksl.Mstop
        self.setups = setups
        self.hamiltonian=hamiltonian
        self.timer=timer
        self.Fref_av=Fref_av
        self.my_atom_indices = bfs.my_atom_indices
        self.atom_indices = bfs.atom_indices
        self.dH_asp = hamiltonian.dH_asp

        from gpaw.kohnsham_layouts import BlacsOrbitalLayouts
        self.isblacs = isinstance(ksl, BlacsOrbitalLayouts)  # XXX

        self.timer.start('TCI derivative')
        dThetadR_qvMM, dTdR_qvMM = self.manytci.O_qMM_T_qMM(
            gd.comm, self.Mstart, self.Mstop, False, derivative=True)
        dPdR_aqvMi = self.manytci.P_aqMi(
            self.bfs.my_atom_indices, derivative=True)
        gd.comm.sum(dThetadR_qvMM)
        gd.comm.sum(dTdR_qvMM)
        self.timer.stop('TCI derivative')

    def _slices(self,indices):
        for a in indices:
            M1 = self.bfs.M_a[a] - self.Mstart
            M2 = M1 + self.setups[a].nao
            if M2 > 0:
                yield a, max(0, M1), M2
    def slices(self):
        return self._slices(self.atom_indices)
    def my_slices(self):
        return self._slices(self.my_atom_indices)

    def get_forces_sum_GS(self):
        
        self.get_initial()

        F_av=np.zeros_like(self.Fref_av)
        Fkin_av=self.get_kinetic_term()
        Fpot_av=self.get_den_mat_term ()
        Ftheta_av=self.get_pot_term()
        Frho_av=self.get_den_mat_paw_term()
        Fatom_av=self.get_atomic_density_term()
        F_av += Fkin_av + Fpot_av + Ftheta_av + Frho_av + Fatom_av

        return F_av

    def get_initial(self):
        self.timer.start('Initial')
        self.rhoT_uMM = []
        self.ET_uMM = []     
        if self.kpt_u[0].rho_MM is None:
            self.timer.start('Get density matrix')
            for kpt in self.kpt_u:
                rhoT_MM = self.ksl.get_transposed_density_matrix(kpt.f_n,
                                                            kpt.C_nM)
                self.rhoT_uMM.append(rhoT_MM)
                ET_MM = self.ksl.get_transposed_density_matrix(kpt.f_n *
                                                          kpt.eps_n,
                                                          kpt.C_nM)
                self.ET_uMM.append(ET_MM)
                if hasattr(kpt, 'c_on'):
                    # XXX does this work with BLACS/non-BLACS/etc.?
                    assert self.bd.comm.size == 1
                    d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                    dtype=kpt.C_nM.dtype)
                    for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                        d_nn += ne * np.outer(c_n.conj(), c_n)
                    rhoT_MM += ksl.get_transposed_density_matrix_delta(
                        d_nn, kpt.C_nM)
                    ET_MM += ksl.get_transposed_density_matrix_delta(
                        d_nn * kpt.eps_n, kpt.C_nM)
            self.timer.stop('Get density matrix')
        else:
            self.rhoT_uMM = []
            self.ET_uMM = []
            for kpt in self.kpt_u:
                H_MM = self.eigensolver.calculate_hamiltonian_matrix(
                    hamiltonian, self, kpt)
                tri2full(H_MM)
                S_MM = kpt.S_MM.copy()
                tri2full(S_MM)
                ET_MM = np.linalg.solve(S_MM, gemmdot(H_MM,
                                                      kpt.rho_MM)).T.copy()
                del S_MM, H_MM
                rhoT_MM = kpt.rho_MM.T.copy()
                self.rhoT_uMM.append(rhoT_MM)
                self.ET_uMM.append(ET_MM)
        self.timer.stop('Initial')

    def get_kinetic_term(self):
        # requires get_initial
        Fkin_av=np.zeros_like(self.Fref_av)

        if self.kpt_u[0].rho_MM is None:
            self.timer.start('Get density matrix')
            for kpt in self.kpt_u:
                rhoT_MM = self.ksl.get_transposed_density_matrix(kpt.f_n,
                                                            kpt.C_nM)
                self.rhoT_uMM.append(rhoT_MM)
            self.timer.stop('Get density matrix')

        self.timer.start('TCI derivative')
        dThetadR_qvMM, dTdR_qvMM = self.manytci.O_qMM_T_qMM(
            self.gd.comm, self.Mstart, self.Mstop, False, derivative=True)
        # Kinetic energy contribution
        #
        #           ----- d T
        #  a         \       mu nu
        # F += 2 Re   )   -------- rho
        #            /    d R         nu mu
        #           -----    mu nu
        #        mu in a; nu
        #
        Fkin_av = np.zeros_like(Fkin_av)
        for u, kpt in enumerate(self.kpt_u):
            dEdTrhoT_vMM = (dTdR_qvMM[kpt.q] *
                            self.rhoT_uMM[u][np.newaxis]).real
            # XXX load distribution!
            for a, M1, M2 in self.my_slices():
                Fkin_av[a, :] += \
                    2.0 * dEdTrhoT_vMM[:, M1:M2].sum(-1).sum(-1)
        del dEdTrhoT_vMM
        self.timer.stop('TCI derivative')

        return Fkin_av

    def get_den_mat_term (self):

        Ftheta_av=np.zeros_like(self.Fref_av)
        dThetadR_qvMM, dTdR_qvMM = self.manytci.O_qMM_T_qMM(
                self.gd.comm, self.Mstart, self.Mstop, False, derivative=True)

        if self.kpt_u[0].rho_MM is None:
            self.timer.start('Get density matrix')
            for kpt in self.kpt_u:
                ET_MM = self.ksl.get_transposed_density_matrix(kpt.f_n *
                                                          kpt.eps_n,
                                                          kpt.C_nM)
                self.ET_uMM.append(ET_MM)
            self.timer.stop('Get density matrix')
        # Density matrix contribution due to basis overlap
        #
        #            ----- d Theta
        #  a          \           mu nu
        # F  += -2 Re  )   ------------  E
        #             /        d R        nu mu
        #            -----        mu nu
        #         mu in a; nu
        #
        Ftheta_av = np.zeros_like(Ftheta_av)
        for u, kpt in enumerate(self.kpt_u):
            dThetadRE_vMM = (dThetadR_qvMM[kpt.q] *
                             self.ET_uMM[u][np.newaxis]).real
            for a, M1, M2 in self.my_slices():
                Ftheta_av[a, :] += \
                    -2.0 * dThetadRE_vMM[:, M1:M2].sum(-1).sum(-1)
        del dThetadRE_vMM

        return Ftheta_av

    def get_pot_term(self):

        Fpot_av=np.zeros_like(self.Fref_av)
        # Potential contribution
        #
        #           -----      /  d Phi  (r)
        #  a         \        |        mu    ~
        # F += -2 Re  )       |   ---------- v (r)  Phi  (r) dr rho
        #            /        |     d R                nu          nu mu
        #           -----    /         a
        #        mu in a; nu
        #
        self.timer.start('Potential')
        vt_sG = self.hamiltonian.vt_sG
        ksl = self.ksl
        rhoT_uMM = []
        if self.kpt_u[0].rho_MM is None:
            self.timer.start('Get density matrix')
            for kpt in self.kpt_u:
                rhoT_MM = self.ksl.get_transposed_density_matrix(kpt.f_n,
                                                            kpt.C_nM)
                rhoT_uMM.append(rhoT_MM)
            self.timer.stop('Get density matrix')
        Fpot_av = np.zeros_like(Fpot_av)
        for u, kpt in enumerate(self.kpt_u):
            vt_G = vt_sG[kpt.s]
            Fpot_av += self.bfs.calculate_force_contribution(vt_G, rhoT_uMM[u],
                                                        kpt.q)

        self.timer.stop('Potential')
        return Fpot_av        

    def get_den_mat_paw_term (self):
        # Density matrix contribution from PAW correction
        #
        #           -----                        -----
        #  a         \      a                     \     b
        # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
        #            /      mu nu  nu mu          /     mu nu  nu mu
        #           -----                        -----
        #           mu nu                    b; mu in a; nu
        #
        # with
        #                  b*
        #         -----  dP
        #   b      \       i mu    b   b
        #  Z     =  )   -------- dS   P
        #   mu nu  /     dR        ij  j nu
        #         -----    b mu
        #           ij
        #

        Frho_av=np.zeros_like(self.Fref_av)
        dPdR_aqvMi = self.manytci.P_aqMi(self.bfs.my_atom_indices,
                                         derivative=True)
        #self.basis_functions.my_atom_indices, derivative=True)
        ET_uMM = []
        if self.kpt_u[0].rho_MM is None:
            self.timer.start('Get density matrix')
            for kpt in self.kpt_u:
                ET_MM = self.ksl.get_transposed_density_matrix(kpt.f_n *
                                                          kpt.eps_n,
                                                          kpt.C_nM)
                ET_uMM.append(ET_MM)
            self.timer.stop('Get density matrix')

        Frho_av = np.zeros_like(Frho_av)
        self.timer.start('add paw correction')
        for u, kpt in enumerate(self.kpt_u):
            work_MM = np.zeros((self.mynao, self.nao), self.dtype)
            ZE_MM = None
            for b in self.my_atom_indices:
                setup = self.setups[b]
                dO_ii = np.asarray(setup.dO_ii, self.dtype)
                dOP_iM = np.zeros((setup.ni, self.nao), self.dtype)
                gemm(1.0, self.P_aqMi[b][kpt.q], dO_ii, 0.0, dOP_iM, 'c')
                for v in range(3):
                    gemm(1.0, dOP_iM,
                         dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop],
                         0.0, work_MM, 'n')
                    ZE_MM = (work_MM * ET_uMM[u]).real
                    for a, M1, M2 in self.slices():
                        dE = 2 * ZE_MM[M1:M2].sum()
                        Frho_av[a, v] -= dE  # the "b; mu in a; nu" term
                        Frho_av[b, v] += dE  # the "mu nu" term
 

        del work_MM, ZE_MM
        self.timer.stop('add paw correction')
        return Frho_av

        

    def get_atomic_density_term (self):

        Fatom_av=np.zeros_like(self.Fref_av)
        dPdR_aqvMi = self.manytci.P_aqMi(self.bfs.my_atom_indices,
                                         derivative=True)
        #self.basis_functions.my_atom_indices, derivative=True)

        rhoT_uMM = []
        if self.kpt_u[0].rho_MM is None:
            self.timer.start('Get density matrix')
            for kpt in self.kpt_u:
                rhoT_MM = self.ksl.get_transposed_density_matrix(kpt.f_n,
                                                            kpt.C_nM)
                rhoT_uMM.append(rhoT_MM)
            self.timer.stop('Get density matrix')
        # Atomic density contribution
        #            -----                         -----
        #  a          \     a                       \     b
        # F  += -2 Re  )   A      rho       + 2 Re   )   A      rho
        #             /     mu nu    nu mu          /     mu nu    nu mu
        #            -----                         -----
        #            mu nu                     b; mu in a; nu
        #
        #                  b*
        #         ----- d P
        #  b       \       i mu   b   b
        # A     =   )   ------- dH   P
        #  mu nu   /    d R       ij  j nu
        #         -----    b mu
        #           ij
        #

        self.timer.start('Atomic Hamiltonian force')
        Fatom_av = np.zeros_like(Fatom_av)
        for u, kpt in enumerate(self.kpt_u):
            for b in self.my_atom_indices:
                H_ii = np.asarray(unpack(self.dH_asp[b][kpt.s]), self.dtype)
                HP_iM = gemmdot(H_ii,
                                np.ascontiguousarray(
                                    self.P_aqMi[b][kpt.q].T.conj()))
                for v in range(3):
                    dPdR_Mi = dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop]
                    ArhoT_MM = (gemmdot(dPdR_Mi, HP_iM) * rhoT_uMM[u]).real
                    for a, M1, M2 in self.slices():
                        dE = 2 * ArhoT_MM[M1:M2].sum()
                        Fatom_av[a, v] += dE  # the "b; mu in a; nu" term
                        Fatom_av[b, v] -= dE  # the "mu nu" term

        self.timer.stop('Atomic Hamiltonian force')
        return Fatom_av
