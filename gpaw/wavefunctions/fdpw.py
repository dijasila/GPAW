from __future__ import division
import numpy as np

from gpaw import extra_parameters
from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.lfc import BasisFunctions
from gpaw.overlap import Overlap
from gpaw.utilities import unpack
from gpaw.utilities.timing import nulltimer
from gpaw.wavefunctions.base import WaveFunctions
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class NullWfsMover:
    description = 'Wavefunctions reused if atoms move'

    def initialize(self, lcaowfs):
        pass

    def cut_wfs(self, wfs, spos_ac):
        pass


class PseudoPartialWaveWfsMover:
    """Move wavefunctions with atoms according to PAW basis

    Wavefunctions are approximated around atom a as

       ~          --  ~ a      ~a    ~
      psi (r)  ~  >  phi (r) < p  | psi >
         n        --    i       i      n
                  ai

    This quantity is then subtracted and re-added at the new
    positions.
    """
    description = 'Improved wavefunction reuse through dual PAW basis'

    def initialize(self, lcaowfs):
        pass

    def cut_wfs(self, wfs, spos_ac):
        ni_a = {}

        #for a, P_ni in wfs.kpt_u[0].P_ani.items():
        for a in range(len(wfs.setups)):
            setup = wfs.setups[a]
            l_j = [phit.get_angular_momentum_number()
                   for phit in setup.get_actual_atomic_orbitals()]
            assert l_j == setup.l_j[:len(l_j)]  # Relationship to l_orb_j?
            ni_a[a] = sum(2 * l + 1 for l in l_j)

        phit = wfs.get_pseudo_partial_waves()
        phit.set_positions(wfs.spos_ac)

        def add_phit_to_wfs(multiplier):
            for kpt in wfs.kpt_u:
                P_ani = {}
                for a in kpt.P_ani:
                    P_ani[a] =  multiplier * kpt.P_ani[a][:, :ni_a[a]]
                phit.add(kpt.psit_nG, c_axi=P_ani, q=kpt.q)

        add_phit_to_wfs(-1.0)

        def paste():
            phit.set_positions(spos_ac)
            add_phit_to_wfs(1.0)

        return paste


class LCAOWfsMover:
    """Move wavefunctions with atoms according to LCAO basis.

    Approximate wavefunctions as a linear combination of atomic
    orbitals, then subtract that linear combination and re-add
    it after moving the atoms using the same coefficients.

    The coefficients c are determined by the equation

               /    *  _  ^  ~   _   _    --
      X     =  | Phi  (r) S psi (r) dr =  >  S      c
       n mu    /    mu         n          --  mu nu  nu n
                                          nu

    We calculate X directly and then solve for c.
    """
    description = 'Improved wavefunction reuse through full LCAO basis'

    # TODO/FIXME
    # * Get rid of the unnecessary T matrix
    # * Only recalculate S/P when necessary (not first time)
    # * Full parallelization support (ScaLAPACK, check scipy atomic correction)
    #   Also replace np.linalg.solve by parallel/efficient thing
    # * Broken with PW mode because PW mode has very funny P_ani shapes.
    #   Also PW does not use the overlap object; this may be related
    # * Can we use updated S matrix to construct better guess?

    def initialize(self, lcaowfs):
        self.bfs = lcaowfs.basis_functions
        self.tci = lcaowfs.tci
        self.atomic_correction = lcaowfs.atomic_correction
        self.S_qMM = lcaowfs.S_qMM
        self.T_qMM = lcaowfs.T_qMM  # Get rid of this
        self.P_aqMi = lcaowfs.P_aqMi

    def cut_wfs(self, wfs, spos_ac):
        # XXX Must forward vars from LCAO initialization object
        # in order to not need to recalculate them.
        # Also, if we get the vars from the LCAO init object,
        # we can rely on those parallelization settings without danger.
        bfs = self.bfs

        nq = len(wfs.kd.ibzk_qc)
        nao = wfs.setups.nao
        P_aqMi = self.P_aqMi
        S_qMM = self.S_qMM

        # We can inherit S_qMM and P_aqMi from the initialization in the
        # first step, then recalculate them for subsequent steps.
        wfs.timer.start('reuse wfs')
        wfs.timer.start('tci calculate')
        self.tci.calculate(wfs.spos_ac, S_qMM, self.T_qMM, P_aqMi)  # kill T
        wfs.timer.stop('tci calculate')
        self.atomic_correction.initialize(P_aqMi,
                                          wfs.initksl.Mstart, wfs.initksl.Mstop)
        #self.atomic_correction.gobble_data(wfs)
        wfs.timer.start('lcao overlap correction')
        self.atomic_correction.add_overlap_correction(wfs, S_qMM)
        wfs.timer.stop('lcao overlap correction')
        wfs.gd.comm.sum(S_qMM)
        c_unM = []
        for kpt in wfs.kpt_u:
            S_MM = S_qMM[kpt.q]
            X_nM = np.zeros((wfs.bd.mynbands, wfs.setups.nao), wfs.dtype)
            # XXX use some blocksize to reduce memory usage?
            opsit_nG = np.zeros_like(kpt.psit_nG)
            wfs.timer.start('wfs overlap')
            wfs.overlap.apply(kpt.psit_nG, opsit_nG, wfs, kpt,
                              calculate_P_ani=False)
            wfs.timer.stop('wfs overlap')
            wfs.timer.start('bfs integrate')
            bfs.integrate2(opsit_nG, c_xM=X_nM, q=kpt.q)
            wfs.timer.stop('bfs integrate')
            wfs.timer.start('gd comm sum')
            wfs.gd.comm.sum(X_nM)
            wfs.timer.stop('gd comm sum')

            # Mind band parallelization / ScaLAPACK
            # Actually we can probably ignore ScaLAPACK for FD/PW calculations
            # since we never adapted Davidson to those.  Although people
            # may have requested ScaLAPACK for LCAO initialization.
            c_nM = np.linalg.solve(S_MM.T, X_nM.T).T.copy()

            #c_nM *= 0  # This disables the whole mechanism
            wfs.timer.start('lcao to grid')
            bfs.lcao_to_grid(C_xM=-c_nM, psit_xG=kpt.psit_nG, q=kpt.q)
            wfs.timer.stop('lcao to grid')
            c_unM.append(c_nM)

        del opsit_nG

        wfs.timer.start('bfs set pos')
        bfs.set_positions(spos_ac)
        wfs.timer.stop('bfs set pos')

        # Is it possible to recalculate the overlaps and make use of how
        # they have changed here?
        wfs.timer.start('re-add wfs')
        for u, kpt in enumerate(wfs.kpt_u):
            bfs.lcao_to_grid(C_xM=c_unM[u], psit_xG=kpt.psit_nG, q=kpt.q)
        wfs.timer.stop('re-add wfs')
        wfs.timer.stop('reuse wfs')


class FDPWWaveFunctions(WaveFunctions):
    """Base class for finite-difference and planewave classes."""
    def __init__(self, diagksl, orthoksl, initksl, reuse_wfs_method=None,
                 **kwargs):
        WaveFunctions.__init__(self, **kwargs)

        self.diagksl = diagksl
        self.orthoksl = orthoksl
        self.initksl = initksl
        if reuse_wfs_method is None:
            wfs_mover = NullWfsMover()
        elif hasattr(reuse_wfs_method, 'cut_wfs'):
            wfs_mover = reuse_wfs_method
        elif reuse_wfs_method == 'paw':
            wfs_mover = PseudoPartialWaveWfsMover()
        elif reuse_wfs_method == 'lcao':
            wfs_mover = LCAOWfsMover()
        else:
            raise ValueError('Strange way to reuse wfs: {}'
                             .format(reuse_wfs_method))

        self.wfs_mover = wfs_mover

        self.set_orthonormalized(False)

        self.overlap = self.make_overlap()

    def __str__(self):
        if self.diagksl.buffer_size is not None:
            s = ('  MatrixOperator buffer_size (KiB): %d\n' %
                 self.diagksl.buffer_size)
        else:
            s = ('  MatrixOperator buffer_size: default value or \n' +
                 ' %s see value of nblock in input file\n' % (28 * ' '))
        diagonalizer_layout = self.diagksl.get_description()
        s += 'Diagonalizer layout: ' + diagonalizer_layout
        orthonormalizer_layout = self.orthoksl.get_description()
        s += 'Orthonormalizer layout: ' + orthonormalizer_layout
        return WaveFunctions.__str__(self) + s

    def set_setups(self, setups):
        WaveFunctions.set_setups(self, setups)

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    def add_pseudo_partial_waves(self, lfc, spos_ac, multiplier=1.0):
        """Add localized functions times current projections (P_ani).

        Effectively "cut and paste" wavefunctions when moving the atoms.
        This should yield better initial wavefunctions at each step in
        relaxations.  Use multiplier -1 to subtract and +1 to add."""
        # XXX this will only work with 'single-zeta type' basis functions.
        # We should pick the correct ones to use.  In fact it should be
        # strictly phit_j because those are orthonormal to projectors.

        # This is not a real 'basis' functions object because those
        # enumerate their states differently.  We could construct a
        # better guess by using a larger basis for this, but that would
        # require explicit calculation of the coefficients.

        # We want to ignore projectors for fictional unbound states.
        # That means always picking only some of the lowest functions, i.e.
        # a lower number of 'i' indices:
        #ni_a = {}

        #for a, P_ni in self.kpt_u[0].P_ani.items():
        #    setup = self.setups[a]
        #    l_j = [phit.get_angular_momentum_number()
        #           for phit in setup.phit_j]#get_actual_atomic_orbitals()]
        #    assert l_j == setup.l_j[:len(l_j)]  # Relationship to l_orb_j?
        #    ni_a[a] = sum(2 * l + 1 for l in l_j)


        nq = len(self.kd.ibzk_qc)
        nao = self.setups.nao
        S_qMM = np.empty((nq, nao, nao), self.dtype)  # XXX distrib
        T_qMM = np.empty((nq, nao, nao), self.dtype)
        P_aqMi = {}
        for a in lfc.my_atom_indices:
            ni = self.setups[a].ni
            P_aqMi[a] = np.empty((nq, nao, ni), self.dtype)

        from gpaw.lcao.atomic_correction import get_atomic_correction
        corr = get_atomic_correction('dense')
        from gpaw.lcao.overlap import NewTwoCenterIntegrals as NewTCI
        tci = NewTCI(self.gd.cell_cv, self.gd.pbc_c, self.setups,
                     self.kd.ibzk_qc, self.kd.gamma)

        #self.tci.set_matrix_distribution(Mstart, mynao)
        tci.calculate(spos_ac, S_qMM, T_qMM, P_aqMi)
        corr.initialize(P_aqMi, self.initksl.Mstart, self.initksl.Mstop)
        #corr.add_overlap_correction(self, S_qMM)
        assert len(S_qMM) == 1
        print(S_qMM[0])

        #coefs = []
        #for kpt in self.kpt_u:
        kpt = self.kpt_u[0]
        S_MM = S_qMM[0]
        from gpaw.utilities.tools import tri2full
        tri2full(S_MM)
        X_nM = np.zeros((self.bd.mynbands, self.setups.nao), self.dtype)
        lfc.integrate2(kpt.psit_nG, c_xM=X_nM, q=kpt.q)
        #coefs.append(X_nM)

        if multiplier == -1.0:
            self.X_nM = X_nM
        else:
            X_nM = self.X_nM

        #for u, kpt in enumerate(self.kpt_u):
            #if multiplier == -1.0:

            #else:
        #invS_MM = np.linalg.inv(S_MM)
        #c_nM = np.dot(invS_MM.T, X_nM.T).T.copy()
        c_nM = np.linalg.solve(S_MM.T, X_nM.T).T.copy()
        if multiplier == -1.0:
            self.prev_c = c_nM
        else:
            c_nM = self.prev_c
        #c_nM = np.linalg.solve(S_MM.T, X_nM.T).T.copy()
        lfc.lcao_to_grid(C_xM=c_nM * multiplier,
                         psit_xG=kpt.psit_nG, q=kpt.q)


        #if 0:  # Using projections
        #    for kpt in self.kpt_u:
        #        P_ani = {}
        #        for a in kpt.P_ani:
        #            P_ani[a] =  multiplier * kpt.P_ani[a][:, :ni_a[a]]
        #        lfc.add(kpt.psit_nG, c_axi=P_ani, q=kpt.q)

    def set_positions(self, spos_ac, atom_partition=None):
        move_wfs = (self.kpt_u[0].psit_nG is not None
                    and self.spos_ac is not None)

        if move_wfs:
            paste_wfs = self.wfs_mover.cut_wfs(self, spos_ac)

        # This will update the positions -- and transfer, if necessary --
        # the projection matrices which may be necessary for updating
        # the wavefunctions.
        WaveFunctions.set_positions(self, spos_ac, atom_partition)

        if move_wfs and paste_wfs is not None:
            paste_wfs()

        self.set_orthonormalized(False)
        self.pt.set_positions(spos_ac)
        self.allocate_arrays_for_projections(self.pt.my_atom_indices)
        self.positions_set = True

    def make_overlap(self):
        return Overlap(self.orthoksl, self.timer)

    def initialize(self, density, hamiltonian, spos_ac):
        """Initialize wave-functions, density and hamiltonian.

        Return (nlcao, nrand) tuple with number of bands intialized from
        LCAO and random numbers, respectively."""

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
            nlcao = self.initialize_wave_functions_from_basis_functions(
                basis_functions, density, hamiltonian, spos_ac)
            nrand = self.bd.nbands - nlcao
        else:
            # We got everything from file:
            nlcao = 0
            nrand = 0
        return nlcao, nrand

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

        self.wfs_mover.initialize(lcaowfs)

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

    def orthonormalize(self):
        for kpt in self.kpt_u:
            self.overlap.orthonormalize(self, kpt)
        self.set_orthonormalized(True)

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
