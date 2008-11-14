import numpy as npy

from gpaw.lfc import BasisFunctions
from gpaw.eigensolvers import get_eigensolver
from gpaw.utilities.blas import axpy
from gpaw.kpoint import KPointCollection

class WaveFunctions(KPointCollection):
    def __init__(self, gd, eigensolver, dtype):
        KPointCollection.__init__(self, gd, dtype)
        #self.kpoints = kpoints
        self.eigensolver = eigensolver
        #self.gd = gd
        #self.dtype = dtype
        self.orthonormalized = False
        self.initialized = False

    def add_to_density(self, nt_sG):
        """Add contribution to pseudo electron-density."""
        f_un = [kpt.f_n for kpt in self.kpt_u]
        self.add_to_density_with_occupation(nt_sG, f_un)

    def add_to_density_with_occupation(self, nt_sG, f_un):
        raise NotImplementedError

    def orthonormalize(self):
        raise NotImplementedError

    def initialize(self, paw):
        raise NotImplementedError

    def set_kpoints(self, weight_k, ibzk_kc, nkpts, nmyu, myuoffset):
        KPointCollection.initialize(self, weight_k, ibzk_kc, nkpts, nmyu,
                                    myuoffset)

class EmptyWaveFunctions(WaveFunctions):
    def __init__(self):
        WaveFunctions.__init__(self, None, None, None)

class LCAOWaveFunctions(WaveFunctions):
    # Contains C_nm, BasisFunctions object, P_kmi, S_kmm, T_kmm
    def __init__(self, gd, eigensolver, dtype):
        WaveFunctions.__init__(self, gd, eigensolver, dtype)
        #self.C_unM = None
        #self.P_kMI = None # I ~ a, i, i.e. projectors across all atoms
        self.S_kMM = None
        self.T_kMM = None
        self.basis_functions = None
        self.nao = None
        self.lcao_initialized = False

    def initialize(self, paw):
        assert self.gd is not None, 'Must set kpoints before initialize'
        hamiltonian = paw.hamiltonian
        density = paw.density
        eigensolver = paw.eigensolver
        assert eigensolver.lcao
        self.basis_functions = BasisFunctions(paw.gd, [n.setup.phit_j
                                                       for n in paw.nuclei],
                                              cut=True)
        if not paw.gamma:
            self.basis_functions.set_k_points(self.ibzk_kc)
            
        self.basis_functions.set_positions([n.spos_c for n in paw.nuclei])

        if not self.lcao_initialized: # XXX
            from gpaw.lcao.hamiltonian import LCAOHamiltonian
            lcao_hamiltonian = LCAOHamiltonian(self.basis_functions,
                                               hamiltonian)
            lcao_hamiltonian.initialize(paw)
            lcao_hamiltonian.initialize_lcao()
            self.nao = lcao_hamiltonian.nao
            nao = lcao_hamiltonian.nao
            #self.nao = nao
            self.T_kMM = lcao_hamiltonian.T_kmm
            self.S_kMM = lcao_hamiltonian.S_kmm
            del lcao_hamiltonian
            self.lcao_initialized = True
            #self.P_kMI = XXXXXX

        if not eigensolver.initialized:
            eigensolver.initialize(paw, self)
        if not self.initialized:
            #kpts = self.kpoints
            self.allocate_bands(paw.nmybands)
            for kpt in self.kpt_u:
                kpt.C_nM = npy.empty((self.nmybands, nao), self.dtype)
            self.initialized = True
        
        if not density.starting_density_initialized:
            density.initialize_from_atomic_density(self)
            #paw.wave_functions_initialized = True

        # TODO: initialize P_kMI, ...
        self.initialized = True

    def orthonormalize(self):
        self.orthonormalized = True

    def add_to_density_with_occupation(self, nt_sG, f_un):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        
        for f_n, C_nM, kpt in zip(f_un, self.C_unM, self.kpt_u):
            rho_MM = npy.dot(C_nM.conj().T * f_n, C_nM)
            self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.k)


class GridWaveFunctions(WaveFunctions):
    def __init__(self, gd, eigensolver, dtype):
        WaveFunctions.__init__(self, gd, eigensolver, dtype)
        # XXX
        #self.psit_unG = None
        #self.projectors = None

    def initialize_wave_functions_from_restart_file(self, paw):
        # Calculation started from a restart file.  Copy data
        # from the file to memory:
        if paw.world.size > 1:
            i = paw.gd.get_slice()
            for kpt in self.kpt_u:
                refs = kpt.psit_nG
                kpt.psit_nG = paw.gd.empty(paw.nmybands, self.dtype)
                # Read band by band to save memory
                for n, psit_G in enumerate(kpt.psit_nG):
                    full = refs[n][:]
                    psit_G[:] = full[i]
        else:
            for kpt in self.kpt_u:
                kpt.psit_nG = kpt.psit_nG[:]
        self.initialized = True
        
    def initialize_wave_functions_from_atomic_orbitals(self, paw):
        eigensolver = get_eigensolver('lcao')
        self.allocate_bands(paw.nmybands)
        lcaowfs = LCAOWaveFunctions(self.gd, eigensolver, self.dtype)
        lcaowfs.set_kpoints(self.weight_k, self.ibzk_kc, self.nkpts,
                            self.nmyu, self.myuoffset)
        lcaowfs.kpt_u = self.kpt_u

        original_eigensolver = paw.eigensolver
        original_nbands = paw.nbands
        original_nmybands = paw.nmybands

        paw.density.lcao = True
        paw.eigensolver = eigensolver
        paw.wfs = lcaowfs

        lcaowfs.initialize(paw)

        paw.nbands = min(paw.nbands, lcaowfs.nao)
        paw.nmybands = min(paw.nmybands, lcaowfs.nao)

        if paw.band_comm.size == 1:
            paw.nmybands = paw.nbands # XXX how can this be right?
        for nucleus in paw.my_nuclei:
            nucleus.reallocate(paw.nmybands)
        paw.hamiltonian.update(paw.density)
        eigensolver.iterate(paw.hamiltonian, lcaowfs)
        
        xcfunc = paw.hamiltonian.xc.xcfunc
        paw.Enlxc = xcfunc.get_non_local_energy()
        paw.Enlkin = xcfunc.get_non_local_kinetic_corrections()
        paw.occupation.calculate(lcaowfs.kpt_u)
        paw.add_up_energies()
        paw.check_convergence() # XXX should not be here
        paw.print_iteration() # XXX will fail if convergence not checked
        paw.fixdensity = max(2, paw.fixdensity) # XXX

        paw.wfs = self
        paw.eigensolver = original_eigensolver
        paw.nbands = original_nbands
        paw.nmybands = original_nmybands
        paw.density.lcao = False

        #self.allocate_bands(paw.nmybands)

        #self.kpt_u = lcaowfs.kpt_u

        # XXX might number of lcao bands be different from grid bands?
        for kpt in self.kpt_u:
            kpt.psit_nG = self.gd.zeros(self.nmybands,
                                        self.dtype)
            part_psit_nG = kpt.psit_nG[:self.nmybands]
            lcaowfs.basis_functions.lcao_to_grid(kpt.C_nM, part_psit_nG, kpt.k)
            kpt.C_nM = None

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
        self.allocate_bands(0)
        for kpt in self.kpt_u:
            kpt.psit_nG = self.gd.zeros(self.nmybands,
                                        dtype=self.dtype)
        if not self.density.starting_density_initialized:
            self.density.initialize_from_atomic_density(self.wfs)

    def add_to_density_with_occupation(self, nt_sG, f_un):
        for kpt in self.kpt_u:
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
        for kpt in self.kpt_u:
            self.overlap.orthonormalize(kpt)
        self.orthonormalized = True

    def initialize(self, paw):
        hamiltonian = paw.hamiltonian
        density = paw.density
        eigensolver = paw.eigensolver
        assert not eigensolver.lcao
        self.overlap = paw.overlap
        if not eigensolver.initialized:
            eigensolver.initialize(paw)
        if not self.initialized:
            paw.text('Atomic orbitals used for initialization:', paw.nao)
            if paw.nbands > paw.nao:
                paw.text('Random orbitals used for initialization:',
                         paw.nbands - paw.nao)
            
            # Now we should find out whether init'ing from file or
            # something else
            self.initialize_wave_functions_from_atomic_orbitals(paw)
            #self.initialize_wave_functions() # XXX
