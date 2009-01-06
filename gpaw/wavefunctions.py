import numpy as np

from gpaw.lfc import BasisFunctions
from gpaw.utilities.blas import axpy
from gpaw.utilities import pack, unpack2
from gpaw.kpoint import KPoint


class EmptyWaveFunctions:
    def __nonzero__(self):
        return False

    def set_orthonormalized(self, flag):
        pass

class WaveFunctions(EmptyWaveFunctions):
    """
    ``setups``      List of setup objects.
    ``symmetry``    Symmetry object.
    ``kpt_u``       List of **k**-point objects.
    ``nbands``      Number of bands.
    ``nspins``      Number of spins.
    ``dtype``       Data type of wave functions (``float`` or
                    ``complex``).
    ``bzk_kc``      Scaled **k**-points used for sampling the whole
                    Brillouin zone - values scaled to [-0.5, 0.5).
    ``ibzk_kc``     Scaled **k**-points in the irreducible part of the
                    Brillouin zone.
    ``weight_k``    Weights of the **k**-points in the irreducible part
                    of the Brillouin zone (summing up to 1).
    ``kpt_comm``    MPI-communicator for parallelization over
                    **k**-points.
    """
    def __init__(self, gd, nspins, setups, nbands, mynbands, dtype,
                 world, kpt_comm, band_comm,
                 gamma, bzk_kc, ibzk_kc, weight_k, symmetry):
        self.gd = gd
        self.nspins = nspins
        self.nbands = nbands
        self.mynbands = mynbands
        self.dtype = dtype
        self.world = world
        self.kpt_comm = kpt_comm
        self.band_comm = band_comm
        self.gamma = gamma
        self.bzk_kc = bzk_kc
        self.ibzk_kc = ibzk_kc
        self.weight_k = weight_k
        self.symmetry = symmetry
        self.kpt_comm = kpt_comm
        self.rank_a = None

        self.set_setups(setups)

        nibzkpts = len(weight_k)

        # Total number of k-point/spin combinations:
        nks = nibzkpts * nspins

        # Number of k-point/spin combinations on this cpu:
        mynks = nks // kpt_comm.size

        ks0 = kpt_comm.rank * mynks
        k0 = ks0 % nibzkpts
        self.kpt_u = []
        sdisp_cd = gd.domain.sdisp_cd
        for ks in range(ks0, ks0 + mynks):
            s, k = divmod(ks, nibzkpts)
            q = k - k0
            weight = weight_k[k] * 2 / nspins
            if gamma:
                phase_cd = None
            else:
                phase_cd = np.exp(2j * np.pi *
                                  sdisp_cd * ibzk_kc[k, :, np.newaxis])
            self.kpt_u.append(KPoint(weight, s, k, q, phase_cd))

        self.ibzk_qc = ibzk_kc[k0:k + 1]

        self.eigensolver = None
        self.timer = None
        
    def set_setups(self, setups):
        self.setups = setups

    def __nonzero__(self):
        return True

    def calculate_density(self, density):
        """Calculate density from wave functions."""
        nt_sG = density.nt_sG
        nt_sG.fill(0.0)
        for kpt in self.kpt_u:
            self.add_to_density_from_k_point(nt_sG, kpt)
        self.band_comm.sum(nt_sG)
        self.kpt_comm.sum(nt_sG)

        if self.symmetry:
            for nt_G in nt_sG:
                self.symmetry.symmetrize(nt_G, self.gd)

    def calculate_atomic_density_matrices(self, density):
        """Calculate atomic density matrices from projections."""
        D_asp = density.D_asp
        for a, D_sp in D_asp.items():
            ni = self.setups[a].ni
            D_sii = np.zeros((self.nspins, ni, ni))
            for kpt in self.kpt_u:
                P_ni = kpt.P_ani[a]
                D_sii[kpt.s] += np.dot(P_ni.T.conj() * kpt.f_n, P_ni).real

            D_sp[:] = [pack(D_ii) for D_ii in D_sii]
            self.band_comm.sum(D_sp)
            self.kpt_comm.sum(D_sp)

        if self.symmetry:
            all_D_asp = []
            for a, setup in enumerate(self.setups):
                D_sp = D_asp.get(a)
                if D_sp is None:
                    ni = setup.ni
                    D_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
                self.gd.comm.broadcast(D_sp, self.rank_a[a])
                all_D_asp.append(D_sp)

            for s in range(self.nspins):
                D_aii = [unpack2(D_sp[s]) for D_sp in all_D_asp]
                for a, D_sp in D_asp.items():
                    setup = self.setups[a]
                    D_sp[s] = pack(setup.symmetrize(a, D_aii,
                                                    self.symmetry.maps))

    def set_positions(self, spos_ac):
        self.rank_a = self.gd.domain.get_ranks_from_positions(spos_ac)
        if self.symmetry is not None:
            self.symmetry.check(spos_ac)

    def collect_eigenvalues(self, k, s):
        return self.collect_array('eps_n', k, s)
    
    def collect_occupations(self, k, s):
        return self.collect_array('f_n', k, s)
    
    def collect_array(self, name, k, s):
        """Helper method for collect_eigenvalues and collect_occupations.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = divmod(k + len(self.ibzk_kc) * s, len(kpt_u))
        a_n = getattr(kpt_u[u], name)
        if kpt_rank == 0:
            if self.band_comm.size == 1:
                return a_n
            
            if self.band_comm.rank == 0:
                b_n = npy.zeros(self.nbands)
            else:
                b_n = None
            self.band_comm.gather(a_n, 0, b_n)
            return b_n

        if self.kpt_comm.rank == kpt_rank:
            # Domain master send this to the global master
            if self.gd.domain.comm.rank == 0:
                if self.band_comm.size == 1:
                    self.kpt_comm.send(a_n, 0, 1301)
                else:
                    if self.band_comm.rank == 0:
                        b_n = npy.zeros(self.nbands)
                    else:
                        b_n = None
                    self.band_comm.gather(a_n, 0, b_n)
                    if self.band_comm.rank == 0:
                        self.kpt_comm.send(b_n, 0, 1301)

        elif self.world.rank == 0:
            b_n = npy.zeros(self.nbands)
            self.kpt_comm.receive(b_n, kpt_rank, 1301)
            return b_n

        # return something also on the slaves
        # might be nicer to have the correct array everywhere XXXX 
        #return a_n or should it be b_n?  Fix this later XXX


from gpaw.lcao.overlap import TwoCenterIntegrals
class LCAOWaveFunctions(WaveFunctions):
    def __init__(self, *args):
        WaveFunctions.__init__(self, *args)
        self.basis_functions = None
        self.tci = None
        self.S_qMM = None
        self.T_qMM = None
        self.P_aqMi = None

    def set_positions(self, spos_ac):
        WaveFunctions.set_positions(self, spos_ac)
        
        if not self.basis_functions:
            # First time:
            self.basis_functions = BasisFunctions(self.gd,
                                                  [setup.phit_j
                                                   for setup in self.setups],
                                                  self.kpt_comm,
                                                  cut=True)
            if not self.gamma:
                self.basis_functions.set_k_points(self.ibzk_qc)
        self.basis_functions.set_positions(spos_ac)

        nq = len(self.ibzk_qc)
        nao = self.setups.nao
        mynbands = self.mynbands
        
        if not self.tci:
            # First time:
            self.tci = TwoCenterIntegrals(self.gd.domain, self.setups,
                                          self.gamma, self.ibzk_qc)

            self.S_qMM = np.zeros((nq, nao, nao), self.dtype)
            self.T_qMM = np.zeros((nq, nao, nao), self.dtype)
            for kpt in self.kpt_u:
                q = kpt.q
                kpt.S_MM = self.S_qMM[q]
                kpt.T_MM = self.T_qMM[q]
                kpt.C_nM = np.empty((mynbands, nao), self.dtype)

        for kpt in self.kpt_u:
            kpt.P_ani = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            for kpt in self.kpt_u:
                kpt.P_ani[a] = np.empty((mynbands, ni), self.dtype)
            
        self.P_aqMi = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            self.P_aqMi[a] = np.zeros((nq, nao, ni), self.dtype)

        for kpt in self.kpt_u:
            q = kpt.q
            kpt.P_aMi = dict([(a, P_qMi[q])
                              for a, P_qMi in self.P_aqMi.items()])
            
        self.tci.set_positions(spos_ac)
        self.tci.calculate(spos_ac, self.S_qMM, self.T_qMM, self.P_aqMi)
            
    def initialize(self, density, hamiltonian, spos_ac):
        if density.nt_sG is None:
            density.update(self, basis_functions=self.basis_functions)
        hamiltonian.update(density)

    def add_to_density_with_occupation(self, nt_sG, f_un):
        XXX
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        
        for kpt, f_n in zip(self.kpt_u, f_un):
            rho_MM = np.dot(kpt.C_nM.conj().T * f_n, kpt.C_nM)
            self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.k)

    def add_to_density_from_k_point(self, nt_sG, kpt):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        
        rho_MM = np.dot(kpt.C_nM.conj().T * kpt.f_n, kpt.C_nM)
        self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.q)


from gpaw.eigensolvers import get_eigensolver
from gpaw.overlap import Overlap
from gpaw.operators import Laplace
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.utilities import unpack
from gpaw.io.tar import TarFileReference

class GridWaveFunctions(WaveFunctions):
    def __init__(self, stencil, *args):
        WaveFunctions.__init__(self, *args)
        # Kinetic energy operator:
        self.kin = Laplace(self.gd, -0.5, stencil, self.dtype)
        self.set_orthonormalized(False)

    def set_setups(self, setups):
        WaveFunctions.set_setups(self, setups)
        self.pt = LFC(self.gd, [setup.pt_j for setup in setups],
                      self.kpt_comm, dtype=self.dtype, forces=True)
        if not self.gamma:
            self.pt.set_k_points(self.ibzk_qc)

        self.overlap = None

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    def set_positions(self, spos_ac):
        WaveFunctions.set_positions(self, spos_ac)
        self.set_orthonormalized(False)
        self.pt.set_positions(spos_ac)

        mynbands = self.mynbands
        for kpt in self.kpt_u:
            kpt.P_ani = {}
        for a in self.pt.my_atom_indices:
            ni = self.setups[a].ni
            for kpt in self.kpt_u:
                kpt.P_ani[a] = np.empty((mynbands, ni), self.dtype)

        if not self.overlap:
            self.overlap = Overlap(self)

    def initialize(self, density, hamiltonian, spos_ac):
        if self.kpt_u[0].psit_nG is None:
            basis_functions = BasisFunctions(self.gd,
                                             [setup.phit_j
                                              for setup in self.setups],
                                             cut=True)
            if not self.gamma:
                basis_functions.set_k_points(self.ibzk_qc)
            basis_functions.set_positions(spos_ac)

        if density.nt_sG is None:
            if self.kpt_u[0].psit_nG is None:
                density.update(self, basis_functions=basis_functions)
            else:
                if density.D_asp is None:
                    for kpt in self.kpt_u:
                        self.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
                    density.update(self, normalize_density=True)
                else:
                    density.update(self,
                                   calculate_atomic_density_matrices=False)

        hamiltonian.update(density)

        if self.kpt_u[0].psit_nG is None:
            self.initialize_wave_functions_from_basis_functions(
                basis_functions, density, hamiltonian, spos_ac)
        elif isinstance(self.kpt_u[0].psit_nG, TarFileReference):
            self.initialize_wave_functions_from_restart_file()

    def initialize_wave_functions_from_basis_functions(self,
                                                       basis_functions,
                                                       density, hamiltonian,
                                                       spos_ac):
        if self.nbands < self.setups.nao:
            lcaonbands = self.nbands
            lcaomynbands = self.mynbands
        else:
            lcaonbands = self.setups.nao
            lcaomynbands = self.setups.nao
            assert self.band_comm.size == 1

        lcaowfs = LCAOWaveFunctions(self.gd, self.nspins, self.setups,
                                    lcaonbands,
                                    lcaomynbands, self.dtype,
                                    self.world, self.kpt_comm,
                                    self.band_comm,
                                    self.gamma, self.bzk_kc, self.ibzk_kc,
                                    self.weight_k, self.symmetry)
        lcaowfs.basis_functions = basis_functions
        lcaowfs.timer = self.timer
        lcaowfs.set_positions(spos_ac)
        hamiltonian.update(density)
        eigensolver = get_eigensolver('lcao', 'lcao')
        eigensolver.iterate(hamiltonian, lcaowfs)

        # Transfer coefficients ...
        for kpt, lcaokpt in zip(self.kpt_u, lcaowfs.kpt_u):
            kpt.C_nM = lcaokpt.C_nM

        # and rid of potentially big arrays early:
        del eigensolver, lcaowfs

        for kpt in self.kpt_u:
            kpt.psit_nG = self.gd.zeros(self.mynbands, self.dtype)
            basis_functions.lcao_to_grid(kpt.C_nM, 
                                         kpt.psit_nG[:lcaomynbands], kpt.q)
            kpt.C_nM = None

            if self.mynbands > lcaomynbands:
                assert not True
                # Add extra states.  If the number of atomic
                # orbitals is less than the desired number of
                # bands, then extra random wave functions are
                # added.

                eps_n = np.empty(nbands)
                f_n = np.empty(nbands)
                eps_n[:nao] = kpt.eps_n[:nao]
                eps_n[nao:] = kpt.eps_n[nao - 1] + 0.5
                f_n[nao:] = 0.0
                f_n[:nao] = kpt.f_n[:nao]
                kpt.eps_n = eps_n
                kpt.f_n = f_n
                kpt.random_wave_functions(nao)

    def initialize_wave_functions_from_restart_file(self):
        # Calculation started from a restart file.  Copy data
        # from the file to memory:
        for kpt in self.kpt_u:
            file_nG = kpt.psit_nG
            kpt.psit_nG = self.gd.empty(self.mynbands, self.dtype)
            # Read band by band to save memory
            for n, psit_G in enumerate(kpt.psit_nG):
                if self.world.rank == 0:
                    big_psit_G = file_nG[n][:]
                else:
                    big_psit_G = None
                self.gd.distribute(big_psit_G, psit_G)
        
    def random_wave_functions(self, paw):
        dsfglksjdfglksjdghlskdjfhglsdfkjghsdljgkhsdljghdsfjghdlfgjkhdsfgjhsdfjgkhdsfgjkhdsfgkjhdsflkjhg

        self.allocate_bands(0)
        for kpt in self.kpt_u:
            kpt.psit_nG = self.gd.zeros(self.mynbands,
                                        dtype=self.dtype)
        if not self.density.starting_density_initialized:
            self.density.initialize_from_atomic_density(self.wfs)

    def add_to_density_from_k_point(self, nt_sG, kpt):
        nt_G = nt_sG[kpt.s]
        if self.dtype == float:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                axpy(f, psit_G**2, nt_G)
        else:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                nt_G += f * (psit_G * psit_G.conj()).real

    def add_to_density_with_occupation(self, nt_sG, f_un):
        XXX
        for kpt in self.kpt_u:
            nt_G = nt_sG[kpt.s]
            f_n = f_un[kpt.u]
            if kpt.dtype == float:
                for f, psit_G in zip(f_n, kpt.psit_nG):
                    axpy(f, psit_G**2, nt_G)  # nt_G += f * psit_G**2
            else:
                for f, psit_G in zip(f_n, kpt.psit_nG):
                    nt_G += f * (psit_G * np.conjugate(psit_G)).real

            # Hack used in delta-scf calculations:
            if hasattr(kpt, 'ft_omn'):
                for ft_mn in kpt.ft_omn:
                    for ft_n, psi_m in zip(ft_mn, kpt.psit_nG):
                        for ft, psi_n in zip(ft_n, kpt.psit_nG):
                            if abs(ft) > 1.e-12:
                                nt_G += (np.conjugate(psi_m) *
                                         ft * psi_n).real

    def orthonormalize(self):
        for kpt in self.kpt_u:
            self.overlap.orthonormalize(self, kpt)
        self.set_orthonormalized(True)

    def initialize2(self, paw):
        khjgkjhgkhjg
        hamiltonian = paw.hamiltonian
        density = paw.density
        eigensolver = paw.eigensolver
        assert not eigensolver.lcao
        self.overlap = paw.overlap
        if not eigensolver.initialized:
            eigensolver.initialize(paw)
        if not self.initialized:
            if self.kpt_u[0].psit_nG is None:
                paw.text('Atomic orbitals used for initialization:', paw.nao)
                if paw.nbands > paw.nao:
                    paw.text('Random orbitals used for initialization:',
                             paw.nbands - paw.nao)
            
                # Now we should find out whether init'ing from file or
                # something else
                self.initialize_wave_functions_from_atomic_orbitals(paw)

            else:
                self.initialize_wave_functions_from_restart_file(paw)

    def get_wave_function_array(self, n, k, s):
        """Return pseudo-wave-function array.
        
        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        nk = len(self.ibzk_kc)
        mynu = len(self.kpt_u)
        kpt_rank, u = divmod(k + nk * s, mynu)
        nn, band_rank = divmod(n, self.band_comm.size)

        psit_nG = self.kpt_u[u].psit_nG
        if psit_nG is None:
            raise RuntimeError('This calculator has no wave functions!')

        size = self.world.size
        rank = self.world.rank
        if size == 1:
            return psit_nG[nn][:]

        if self.kpt_comm.rank == kpt_rank:
            if self.band_comm.rank == band_rank:
                psit_G = self.gd.collect(psit_nG[nn][:])

                if kpt_rank == 0 and band_rank == 0:
                    if rank == 0:
                        return psit_G

                # Domain master send this to the global master
                if self.domain.comm.rank == 0:
                    self.world.send(psit_G, 0, 1398)

        if rank == 0:
            # allocate full wavefunction and receive
            psit_G = self.gd.empty(dtype=self.dtype, global_array=True)
            world_rank = (kpt_rank * self.domain.comm.size *
                          self.band_comm.size +
                          band_rank * self.domain.comm.size)
            self.world.receive(psit_G, world_rank, 1398)
            return psit_G

    def calculate_forces(self, hamiltonian, F_av):
        # Calculate force-contribution from k-points:
        F_aniv = self.pt.dict(self.nbands, derivative=True)
        for kpt in self.kpt_u:
            self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)
            for a, F_niv in F_aniv.items():
                F_niv = F_niv.conj()
                F_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                P_ni = kpt.P_ani[a]
                F_vii = np.dot(np.dot(F_niv.transpose(), P_ni), dH_ii)
                F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                dO_ii = hamiltonian.setups[a].O_ii
                F_vii -= np.dot(np.dot(F_niv.transpose(), P_ni), dO_ii)
                F_av[a] += 2 * F_vii.real.trace(0, 1, 2)
