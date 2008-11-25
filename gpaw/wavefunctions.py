import numpy as np

from gpaw.lfc import BasisFunctions
from gpaw.utilities.blas import axpy
from gpaw.utilities import pack
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
        self.setups = setups
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

        nibzkpts = len(weight_k)

        # Total number of k-point/spin combinations:
        nks = nibzkpts * nspins

        # Number of k-point/spin combinations on this cpu:
        mynks = nks // kpt_comm.size

        ks0 = kpt_comm.rank * mynks
        k0 = ks0 % nibzkpts
        self.kpt_u = []
        for ks in range(ks0, ks0 + mynks):
            s, k = divmod(ks, nibzkpts)
            q = k - k0
            weight = weight_k[k] * 2 / nspins
            self.kpt_u.append(KPoint(weight, s, k, q))

        self.ibzk_qc = ibzk_kc[k0:k + 1]

        self.eigensolver = None
        self.timer = None
        
    def add_to_density(self, nt_sG, D_asp):
        """Add contribution to pseudo electron-density."""
        for kpt in self.kpt_u:
            self.add_to_density_from_k_point(nt_sG, kpt)
        self.band_comm.sum(nt_sG)
        self.kpt_comm.sum(nt_sG)

        for a, D_sp in D_asp.items():
            ni = self.setups[a].ni
            D_sii = np.zeros((self.nspins, ni, ni))
            for kpt in self.kpt_u:
                P_ni = kpt.P_ani[a]
                D_sii[kpt.s] += np.dot(P_ni.T.conj() * kpt.f_n, P_ni).real

            D_sp[:] = [pack(D_ii) for D_ii in D_sii]
            self.band_comm.sum(D_sp)
            self.kpt_comm.sum(D_sp)

        if self.symmetry is not None:
            for nt_G in self.nt_sG:
                self.symmetry.symmetrize(nt_G, self.gd)

            all_D_asp = []
            for a, setup in enumerate(self.setups):
                D_sp = D_asp.get(a)
                if D_sp is None:
                    ni = setup.ni
                    D_sp = np.empty((self.nspins, ni * (ni + 1) / 2))
                self.gd.comm.broadcast(D_sp, 0)
                assert self.gd.comm.size == 1
                all_D_asp.append(D_sp)

            for s in range(self.nspins):
                D_aii = [unpack2(D_sp[s]) for D_sp in all_D_asp]
                for a, D_sp in D_asp.items():
                    D_sp[s] = pack(self.setups[a].symmetrize(a, D_aii,
                                                             symmetry.maps))

    def set_positions(self, spos_ac):
        if self.symmetry is not None:
            self.symmetry.check(spos_ac)

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
                basis_functions.set_k_points(self.ibzk_qc)
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
                kpt.eps_n = np.empty(mynbands)
                kpt.f_n = np.empty(mynbands)

        self.P_aqMi = {}
        self.P_aqni = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            self.P_aqMi[a] = np.zeros((nq, nao, ni), self.dtype)
            self.P_aqni[a] = np.empty((nq, mynbands, ni), self.dtype)
            
        for kpt in self.kpt_u:
            q = kpt.q
            kpt.P_aMi = dict([(a, P_qMi[q])
                              for a, P_qMi in self.P_aqMi.items()])
            kpt.P_ani = dict([(a, P_qni[q])
                              for a, P_qni in self.P_aqni.items()])
            
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
        self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.k)


from gpaw.eigensolvers import get_eigensolver
from gpaw.overlap import Overlap
class GridWaveFunctions(WaveFunctions):
    def __init__(self, stencil, *args):
        WaveFunctions.__init__(self, *args)
        # Kinetic energy operator:
        self.kin = Laplace(self.gd, -0.5, stencil, self.dtype)
        self.set_orthonormalized(False)
        self.pt = LFC(gd, [setup.pt_j for setup in setups], self.kpt_comm)
        if not self.gamma:
            self.pt.set_k_points(self.ibzk_qc)

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    def set_positions(self, spoc_ac):
        WaveFunctions.set_positions(self, spos_ac)

        self.set_orthonormalized(False)

        self.pt.set_positions(spos_ac)

        self.overlap = Overlap(self) # XXX grid specific, should live in wfs

        nq = len(self.ibzk_qc)
        mynbands = self.mynbands

        self.P_aqni = {}
        for a in self.pt.my_atom_indices:
            ni = self.setups[a].ni
            self.P_aqni[a] = np.empty((nq, mynbands, ni), self.dtype)
            
        for kpt in self.kpt_u:
            q = kpt.q
            kpt.P_ani = dict([(a, P_qni[q])
                              for a, P_qni in self.P_aqni.items()])

    def initialize(self, density, hamiltonian, spos_ac):
        if density.nt_sG is None or self.kpt_u[0].psit_nG is None:
            basis_functions = BasisFunctions(self.gd,
                                             [setup.phit_j
                                              for setup in self.setups],
                                             cut=True)
            if not self.gamma:
                basis_functions.set_k_points(self.ibzk_qc)
            basis_functions.set_positions(spos_ac)

        if density.nt_sG is None:
            density.update(self, basis_functions=basis_functions)

        hamiltonian.update(density)

        if self.kpt_u[0].psit_nG is None:
            nlcaobands = min(self.nbands, self.setups.nao)
            lcao = LCAOWaveFunctions(self.gd, self.nspins, lcaonbands,
                                     self.mynbands, self.dtype,
                                     self.kpt_comm, self.band_comm,
                                     self.gamma, self.bzk_kc, self.ibzk_kc,
                                     self.weight_k, self.symmetry)
            lcao.basis_functions = basis_functions
            hamiltonian.update(density)
            eigensolver = get_eigensolver('lcao', 'lcao')
            eigensolver.iterate(hamiltonian, lcao)

            for kpt, lcaokpt in zip(self.kpt_u, lcao.kpt_u):
                kpt.psit_nG = self.gd.zeros(self.mynbands,
                                            self.dtype)
                basis_functions.lcao_to_grid(lcaokpt.C_nM, 
                                             kpt.psit_nG[:nlcaobands], kpt.k)
                lcaokpt.C_nM = None

                if 0:#nbands > nao:
                    # Add extra states.
                    # If the number of atomic orbitals is less than the desired
                    # number of bands, then extra random wave functions are added.

                    eps_n = np.empty(nbands)
                    f_n = np.empty(nbands)
                    eps_n[:nao] = kpt.eps_n[:nao]
                    eps_n[nao:] = kpt.eps_n[nao - 1] + 0.5
                    f_n[nao:] = 0.0
                    f_n[:nao] = kpt.f_n[:nao]
                    kpt.eps_n = eps_n
                    kpt.f_n = f_n
                    kpt.random_wave_functions(nao)

    def initialize_wave_functions_from_restart_file(self, paw):
        # Calculation started from a restart file.  Copy data
        # from the file to memory:
        if paw.world.size > 1:
            i = paw.gd.get_slice()
            for kpt in self.kpt_u:
                refs = kpt.psit_nG
                kpt.psit_nG = paw.gd.empty(paw.mynbands, self.dtype)
                # Read band by band to save memory
                for n, psit_G in enumerate(kpt.psit_nG):
                    full = refs[n][:]
                    psit_G[:] = full[i]
        else:
            for kpt in self.kpt_u:
                kpt.psit_nG = kpt.psit_nG[:]
        self.initialized = True
        
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
        f_n = f_un[kpt.u]
        if self.dtype == float:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G**2, nt_G)
        else:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                nt_G += f * (psit_G * psit_G.conj()).real

    def add_to_density_with_occupation(self, nt_sG, f_un):
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
            self.overlap.orthonormalize(kpt)
        self.set_orthonormalized(True)

    def initialize2(self, paw):
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
