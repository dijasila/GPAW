import numpy as npy

from gpaw.lfc import BasisFunctions
from gpaw.eigensolvers import get_eigensolver

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

    def initialize(self, paw, atoms, hamiltonian, density, eigensolver):
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

    def initialize(self, paw, atoms, hamiltonian, density, eigensolver):
        assert eigensolver.lcao
        self.basis_functions = BasisFunctions(paw.gd,
                                              [n.setup.phit_j
                                               for n in paw.nuclei])
        if not paw.gamma:
            self.basis_functions.set_j_points(self.kpoints.ibzk_kc)
            
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
        self.psit_unG = None
        self.projectors = None

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
        paw.density.lcao = True
        lcaowfs.initialize(paw, paw.atoms, paw.hamiltonian, paw.density,
                           eigensolver)
        nbands = min(paw.nbands, lcaowfs.nao)

        #if paw.band_comm.size == 1:
        #    nmybands = nbands

        #for nucleus in paw.my_nuclei:
        #    nucleus.reallocate(nmybands)
        old_eigensolver = paw.eigensolver
        paw.eigensolver = eigensolver
        paw.wfs = lcaowfs
        
        paw.hamiltonian.update(paw.density)
        eigensolver.iterate(paw.hamiltonian, lcaowfs)
        
        xcfunc = paw.hamiltonian.xc.xcfunc
        paw.Enlxc = xcfunc.get_non_local_energy()
        paw.Enlkin = xcfunc.get_non_local_kinetic_corrections()
        paw.occupation.calculate(self.kpoints.kpt_u)        
        paw.add_up_energies()
        paw.check_convergence()
        paw.print_iteration()

        paw.eigensolver = old_eigensolver
        paw.wfs = self

        #raise RuntimeError('sdfkjsdfkj')
        
        #paw.hamiltonian.update(paw.density)
        #paw.density.update(self, paw.symmetry)
        #paw.update_kinetic()
        #paw.hamiltonian.update(paw.density)
        #paw.eigensolver.iterate(paw.hamiltonian, self)

        #original_eigensolver = self.eigensolver
        #original_nbands = self.nbands
        #original_nmybands = self.nmybands
        #original_maxiter = self.maxiter

        #self.maxiter = 0


        #try:
        #    self.find_ground_state(self.atoms, write=False)
        #except KohnShamConvergenceError:
         #   pass

        #self.maxiter = original_maxiter
        #self.nbands = original_nbands
        #self.nmybands = original_nmybands
        for kpt in self.kpt_u:
            kpt.calculate_wave_functions_from_lcao_coefficients(
                self.nmybands, self.hamiltonian.basis_functions)
            # Delete basis-set expansion coefficients:
            #kpt.C_nm = None

        for nucleus in self.nuclei:
            del nucleus.P_kmi

        for nucleus in self.my_nuclei:
            nucleus.reallocate(self.nmybands)

        self.eigensolver = original_eigensolver
        if self.xcfunc.is_gllb():
            self.xcfunc.xc.eigensolver = self.eigensolver
        #self.density.mixer.reset(self.my_nuclei)
        self.density.lcao = False

    def random_wave_functions(self, paw):
        self.kpoints.allocate(0)
        for kpt in self.kpoints.kpt_u:
            kpt.psit_nG = self.gd.zeros(self.nmybands,
                                        dtype=self.dtype)
        if not self.density.starting_density_initialized: # XXX???
            self.density.initialize_from_atomic_density(self.wfs)
            # Should use preinitialized LCAO wave function object?
            # This makes little sense, why is this code called
            # from a section concerning only random wfs?
            #
            # Isn't this completely broken?

    def add_to_density_with_occupation(self, nt_G, f_n):
        for kpt in self.kpoints.kpt_u:
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

    def initialize(self, paw, atoms, hamiltonian, density, eigensolver):
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
