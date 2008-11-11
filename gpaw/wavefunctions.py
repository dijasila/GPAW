import numpy as npy

from gpaw.lfc import BasisFunctions
from gpaw.eigensolvers import get_eigensolver
from gpaw.utilities.blas import axpy

class WaveFunctions:
    def __init__(self, kpoints):
        self.kpoints = kpoints
        self.eigensolver = None
        self.orthonormalized = False
        self.wfs_initialized = False

    def add_to_density(self, nt_sG):
        """Add contribution to pseudo electron-density."""
        self.add_to_density_with_occupation(nt_sG, self.kpoints.f_un)

    def add_to_density_with_occupation(self, nt_sG, f_un):
        raise NotImplementedError

    def orthonormalize(self):
        raise NotImplementedError

    def is_orthonormalized(self):
        return self.orthonormalized

    def is_initialized(self):
        return self.wfs_initialized

    def initialize(self, paw, hamiltonian, density, eigensolver):
        raise NotImplementedError

    def atomic_movement(self):
        self.orthonormalized = False

class EmptyWaveFunctions(WaveFunctions):
    def __init__(self):
        WaveFunctions.__init__(self, None)

class LCAOmatic(WaveFunctions):
    # Contains C_nm, BasisFunctions object, P_kmi, S_kmm, T_kmm
    def __init__(self, kpoints):
        WaveFunctions.__init__(self, kpoints)
        self.C_unM = None
        self.P_kMI = None # I ~ a, i, i.e. projectors across all atoms
        self.S_kMM = None
        self.T_kMM = None
        self.basis_functions = None
        self.nao = None
        self.lcao_initialized = False
        #self.lcao_hamiltonian = None # class to be removed

    def initialize(self, paw, hamiltonian, density, eigensolver):
        assert eigensolver.lcao
        self.basis_functions = BasisFunctions(paw.gd, [n.setup.phit_j
                                                       for n in paw.nuclei])
        if not paw.gamma:
            self.basis_functions.set_k_points(self.kpoints.ibzk_kc)
            
        self.basis_functions.set_positions([n.spos_c for n in paw.nuclei])

        if not self.lcao_initialized: # XXX
            from gpaw.lcao.hamiltonian import LCAOHamiltonian
            lcao_hamiltonian = LCAOHamiltonian(self.basis_functions,
                                               hamiltonian)
            lcao_hamiltonian.initialize(paw)
            lcao_hamiltonian.initialize_lcao()
            self.nao = lcao_hamiltonian.nao
            nao = lcao_hamiltonian.nao
            self.nao = nao
            self.T_kMM = lcao_hamiltonian.T_kmm
            self.S_kMM = lcao_hamiltonian.S_kmm
            del lcao_hamiltonian
            #self.P_kMI = XXXXXX

        if not eigensolver.initialized:
            eigensolver.initialize(paw, self)
        if not self.wfs_initialized:
            kpts = self.kpoints
            kpts.allocate(paw.nmybands)
            self.C_unM = npy.empty((kpts.nmyu, kpts.nmybands, nao), kpts.dtype)
            self.wfs_initialized = True
        
        if not density.starting_density_initialized:
            density.initialize_from_atomic_density(self)
            #paw.wave_functions_initialized = True

        # TODO: initialize P_kMI, ...
        self.wfs_initialized = True

    def orthonormalize(self):
        self.orthonormalized = True

    def add_to_density_with_occupation(self, nt_sG, f_un):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        
        for f_n, C_nM, kpt in zip(f_un, self.C_unM, self.kpoints.kpt_u):
            rho_MM = npy.dot(C_nM.conj().T * f_n, C_nM)
            self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.k)


class Gridmotron(WaveFunctions):
    def __init__(self, kpoints):
        WaveFunctions.__init__(self, kpoints)
        # XXX
        #self.psit_unG = None
        #self.projectors = None

    def initialize_wave_functions_from_restart_file(self, paw):
        # Calculation started from a restart file.  Copy data
        # from the file to memory:
        if paw.world.size > 1:
            i = paw.gd.get_slice()
            for kpt in self.kpoints.kpt_u:
                refs = kpt.psit_nG
                kpt.psit_nG = paw.gd.empty(paw.nmybands, self.kpoints.dtype)
                # Read band by band to save memory
                for n, psit_G in enumerate(kpt.psit_nG):
                    full = refs[n][:]
                    psit_G[:] = full[i]
        else:
            for kpt in self.kpoints.kpt_u:
                kpt.psit_nG = kpt.psit_nG[:]
        self.wfs_initialized = True
        
    def initialize_wave_functions_from_atomic_orbitals(self, paw):
        lcaowfs = LCAOmatic(self.kpoints)
        eigensolver = get_eigensolver('lcao')

        original_eigensolver = paw.eigensolver
        original_nbands = paw.nbands
        original_nmybands = paw.nmybands
        

        lcaowfs.initialize(paw, paw.hamiltonian, paw.density,
                           eigensolver)

        paw.nbands = min(paw.nbands, lcaowfs.nao)
        paw.nmybands = min(paw.nmybands, lcaowfs.nao)
        paw.density.lcao = True
        paw.eigensolver = eigensolver
        paw.wfs = lcaowfs

        if paw.band_comm.size == 1:
            paw.nmybands = paw.nbands # XXX how can this be right?
        for nucleus in paw.my_nuclei:
            nucleus.reallocate(paw.nmybands)
        paw.hamiltonian.update(paw.density)
        eigensolver.iterate(paw.hamiltonian, lcaowfs)
        
        xcfunc = paw.hamiltonian.xc.xcfunc
        paw.Enlxc = xcfunc.get_non_local_energy()
        paw.Enlkin = xcfunc.get_non_local_kinetic_corrections()
        paw.occupation.calculate(self.kpoints.kpt_u)
        paw.add_up_energies()
        paw.check_convergence()
        paw.print_iteration()
        paw.fixdensity = 2 # XXX

        paw.wfs = self
        paw.eigensolver = original_eigensolver
        paw.nbands = original_nbands
        paw.nmybands = original_nmybands
        paw.density.lcao = False

        for kpt in self.kpoints.kpt_u:
            kpt.psit_nG = kpt.gd.zeros(self.kpoints.nmybands)
        # XXX might number of lcao bands be different from grid bands?
        part_psit_unG = [kpt.psit_nG[:self.kpoints.nmybands]
                         for kpt in self.kpoints.kpt_u]
        for C_nM, psit_nG, kpt in zip(lcaowfs.C_unM, part_psit_unG,
                                      self.kpoints.kpt_u):
            lcaowfs.basis_functions.lcao_to_grid(C_nM, psit_nG, kpt.k)
        # XXX init remaining wave functions randomly
        for nucleus in paw.nuclei:
            del nucleus.P_kmi
        for nucleus in paw.my_nuclei:
            nucleus.reallocate(paw.nmybands)
                
        paw.density.scale()
        paw.density.interpolate_pseudo_density()
        self.orthonormalize()

        if paw.xcfunc.is_gllb():
            paw.xcfunc.xc.eigensolver = paw.eigensolver

    def random_wave_functions(self, paw):
        self.kpoints.allocate(0)
        for kpt in self.kpoints.kpt_u:
            kpt.psit_nG = self.gd.zeros(self.nmybands,
                                        dtype=self.dtype)
        if not self.density.starting_density_initialized:
            self.density.initialize_from_atomic_density(self.wfs)

    def add_to_density_with_occupation(self, nt_sG, f_un):
        for kpt in self.kpoints.kpt_u:
            nt_G = nt_sG[kpt.s]
            f_n = f_un[kpt.u]
            if kpt.dtype == float:
                for f, psit_G in zip(f_n, kpt.psit_nG):
                    axpy(f, psit_G**2, nt_G)  # nt_G += f * psit_G**2
            else:
                for f, psit_G in zip(f_n, kpt.psit_nG):
                    nt_G += f * (psit_G * npy.conjugate(psit_G)).real

            # Hack used in delta-scf calculations:
            if hasattr(kpt, 'ft_omn'):
                for ft_mn in kpt.ft_omn:
                    for ft_n, psi_m in zip(ft_mn, kpt.psit_nG):
                        for ft, psi_n in zip(ft_n, kpt.psit_nG):
                            if abs(ft) > 1.e-12:
                                nt_G += (npy.conjugate(psi_m) *
                                         ft * psi_n).real

    def orthonormalize(self):
        for kpt in self.kpoints.kpt_u:
            self.overlap.orthonormalize(kpt)
        self.orthonormalized = True

    def initialize(self, paw, hamiltonian, density, eigensolver):
        assert not eigensolver.lcao
        self.overlap = paw.overlap
        if not eigensolver.initialized:
            eigensolver.initialize(paw)
        if not self.wfs_initialized:
            paw.text('Atomic orbitals used for initialization:', paw.nao)
            if paw.nbands > paw.nao:
                paw.text('Random orbitals used for initialization:',
                         paw.nbands - paw.nao)
            
            # Now we should find out whether init'ing from file or
            # something else
            self.initialize_wave_functions_from_atomic_orbitals(paw)
            #self.initialize_wave_functions() # XXX
