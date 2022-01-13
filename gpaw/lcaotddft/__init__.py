import numpy as np
from typing import Optional
from gpaw.typing import ArrayLike

from ase.units import Bohr, Hartree

from gpaw.calculator import GPAW
from gpaw.external import ExternalPotential, ConstantElectricField
from gpaw.lcaotddft.hamiltonian import TimeDependentHamiltonian
from gpaw.lcaotddft.logger import TDDFTLogger
from gpaw.lcaotddft.propagators import create_propagator
from gpaw.tddft.units import attosec_to_autime
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.tddft.tdopers import TimeDependentDensity
from gpaw.utilities.scalapack import scalapack_zero
from scipy.linalg import schur, eigvals

from gpaw.blacs import Redistributor


class LCAOTDDFT(GPAW):
    """Real-time time-propagation TDDFT calculator with LCAO basis.

    Parameters
    ----------
    filename
        File containing ground state or time-dependent state to propagate
    propagator
        Time propagator for the Kohn-Sham wavefunctions
    td_potential
        External time-dependent potential
    rremission
        Radiation-reaction potential for Self-consistent Light-Matter coupling
    fxc
        Exchange-correlation functional used for
        the dynamic part of Hamiltonian
    scale
        Experimental option (use carefully).
        Scaling factor for the dynamic part of Hamiltonian
    parallel
        Parallelization options
    communicator
        MPI communicator
    txt
        Text output
    """
    def __init__(self, filename: str, *,
                 propagator: dict = None,
                 td_potential: dict = None,
                 rremission: object = None,
                 fxc: str = None,
                 scale: float = None,
                 parallel: dict = None,
                 communicator: object = None,
                 txt: str = '-',
                 PP_flag: bool = False,
                 ED_F: bool = False,
                 S_flag: bool = False,
                 calculate_energy: bool = True):
        """"""
        assert filename is not None
        self.time = 0.0
        self.niter = 0
        # TODO: deprecate kick keywords (and store them as td_potential)
        self.kick_strength = np.zeros(3)
        self.kick_ext: Optional[ExternalPotential] = None
        self.tddft_initialized = False
        self.action = ''
        tdh = TimeDependentHamiltonian(fxc=fxc, td_potential=td_potential,
                                       scale=scale, rremission=rremission)
        self.td_hamiltonian = tdh

        self.propagator_set = propagator is not None
        self.propagator = create_propagator(propagator)
        self.default_parameters = GPAW.default_parameters.copy()
        self.default_parameters['symmetry'] = {'point_group': False}
        GPAW.__init__(self, filename, parallel=parallel,
                      communicator=communicator, txt=txt)
        self.set_positions()
        self.td_density = TimeDependentDensity(self)
        self.calculate_energy = calculate_energy
        self.PP_flag = PP_flag
        self.ED_F = ED_F
        self.S_flag = S_flag
        # Save old overlap S_MM_old which is necessary for propagating C_MM
        for kpt in self.wfs.kpt_u:
            kpt.S_MM_old = kpt.S_MM.copy()

    def write(self, filename, mode=''):
        # This function is included here in order to generate
        # documentation for LCAOTDDFT.write() with autoclass in sphinx
        GPAW.write(self, filename, mode=mode)

    def _write(self, writer, mode):
        GPAW._write(self, writer, mode)
        if self.tddft_initialized:
            w = writer.child('tddft')
            w.write(time=self.time,
                    niter=self.niter,
                    kick_strength=self.kick_strength,
                    propagator=self.propagator.todict())
            self.td_hamiltonian.write(w.child('td_hamiltonian'))

    def read(self, filename):
        reader = GPAW.read(self, filename)
        if 'tddft' in reader:
            r = reader.tddft
            self.time = r.time
            self.niter = r.niter
            self.kick_strength = r.kick_strength
            if not self.propagator_set:
                self.propagator = create_propagator(r.propagator)
            else:
                self.log('Note! Propagator possibly changed!')
            self.td_hamiltonian.wfs = self.wfs
            self.td_hamiltonian.read(r.td_hamiltonian)

    def tddft_init(self):
        if self.tddft_initialized:
            return

        self.log('-----------------------------------')
        self.log('Initializing time-propagation TDDFT')
        self.log('-----------------------------------')
        self.log()

        assert self.wfs.dtype == complex

        self.timer.start('Initialize TDDFT')

        # Initialize Hamiltonian
        self.td_hamiltonian.initialize(self)

        # Initialize propagator
        self.propagator.initialize(self)
        
        self.log('Propagator:')
        self.log(self.propagator.get_description())
        self.log()

        # Add logger
        TDDFTLogger(self)

        # Call observers before propagation
        self.action = 'init'
        self.call_observers(self.niter)

        self.tddft_initialized = True
        self.timer.stop('Initialize TDDFT')

    def absorption_kick(self, kick_strength: ArrayLike):
        """Kick with a weak electric field.

        Parameters
        ----------
        kick_strength
            Strength of the kick in atomic units
        """
        self.tddft_init()

        self.timer.start('Kick')

        self.kick_strength = np.array(kick_strength, dtype=float)
        magnitude = np.sqrt(np.sum(self.kick_strength**2))
        direction = self.kick_strength / magnitude

        self.log('----  Applying absorption kick')
        self.log('----  Magnitude: %.8f Hartree/Bohr' % magnitude)
        self.log('----  Direction: %.4f %.4f %.4f' % tuple(direction))

        # Create hamiltonian object for absorption kick
        cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)

        # Propagate kick
        self.propagator.kick(cef, self.time)

        # Call observers after kick
        self.action = 'kick'
        self.call_observers(self.niter)
        self.niter += 1
        self.timer.stop('Kick')

    def kick(self, ext):
        """Kick with any external potential.

        Parameters
        ----------
        ext
            External potential
        """
        self.tddft_init()

        self.timer.start('Kick')

        self.log('----  Applying kick')
        self.log('----  %s' % ext)

        self.kick_ext = ext

        # Propagate kick
        self.propagator.kick(ext, self.time)

        # Call observers after kick
        self.action = 'kick'
        self.call_observers(self.niter)
        self.niter += 1
        self.timer.stop('Kick')

    def propagate(self, time_step: float = 10.0, iterations: int = 2000):
        """Propagate the electronic system.

        Parameters
        ----------
        time_step
            Time step in attoseconds
        iterations
            Number of propagation steps
        """
        self.tddft_init()

        time_step *= attosec_to_autime
        self.maxiter = self.niter + iterations

        self.log('----  About to do %d propagation steps' % iterations)

        self.timer.start('Propagate')
        while self.niter < self.maxiter:
            # Propagate one step
            self.time = self.propagator.propagate(self.time, time_step)

            # Call registered callback functions
            self.action = 'propagate'
            self.call_observers(self.niter)

            self.niter += 1
        self.timer.stop('Propagate')

    def replay(self, **kwargs):
        # TODO: Consider deprecating this function?
        self.propagator = create_propagator(**kwargs)
        self.tddft_init()
        self.propagator.control_paw(self)

    def get_td_energy(self):
 
        """Calculate the time-dependent total energy"""
        if not self.calculate_energy:
            self.Etot = 0.0

        # self.td_overlap.update(self.wfs)
        # self.td_density.update()
        # self.td_hamiltonian.update('density')
        # self.td_hamiltonian.update(self.td_density.get_density(),self.time)
        self.td_hamiltonian.update()
        self.update_eigenvalues()
        return self.Etot

    def update_eigenvalues(self):
        # Calculate eigenvalue by non scf hamiltonian diagonalization
        for kpt in self.wfs.kpt_u:
            eig = self.wfs.eigensolver
            H_MM = eig.calculate_hamiltonian_matrix(self.hamiltonian,
                                                    self.wfs, kpt)
            C_nM = kpt.C_nM.copy()
            eig.iterate_one_k_point(self.hamiltonian, self.wfs, kpt, C_nM=C_nM)
            kpt.C_nM = C_nM.copy()

        # Calculate eigenvalue by rho_uMM * H_MM
        dmat = DensityMatrix(self)
        
        self.e_band_rhoH = 0.0
        self.e_band = 0.0
        rho_uMM = dmat.get_density_matrix((self.niter, self.action))
        get_H_MM = self.td_hamiltonian.get_hamiltonian_matrix
        ksl = self.wfs.ksl
        for u, kpt in enumerate(self.wfs.kpt_u):
            rho_MM = rho_uMM[u]

            # H_MM = get_H_MM(kpt, paw.time)
            H_MM = get_H_MM(kpt, self.time, addfxc=False, addpot=False)

            if ksl.using_blacs:
                # rhoH_MM = (rho_MM * H_MM).real  # General case
                rhoH_MM = rho_MM.real * H_MM.real  # Hamiltonian is real
                # Hamiltonian has correct values only in lower half, so
                # 1. Add lower half and diagonal twice
                scalapack_zero(ksl.mmdescriptor, rhoH_MM, 'U')
                e = 2 * np.sum(rhoH_MM)
                # 2. Reduce the extra diagonal
                scalapack_zero(ksl.mmdescriptor, rhoH_MM, 'L')
                e -= np.sum(rhoH_MM)
                # Sum over all ranks
                e = ksl.block_comm.sum(e)
            else:
                e = np.sum(rho_MM * H_MM).real
            self.e_band_rhoH += e

        H = self.td_hamiltonian.hamiltonian

        # PAW
        self.e_band = self.wfs.calculate_band_energy()
        self.Ekin = H.e_kinetic0 + self.e_band
        self.e_coulomb = H.e_coulomb
        self.Eext = H.e_external
        self.Ebar = H.e_zero
        self.Exc = H.e_xc
        self.Etot = self.Ekin + self.e_coulomb + self.Ebar + self.Exc

    def save_old_S_MM(self):
        """Save overlap function from previous MD step"""
        for kpt in self.wfs.kpt_u:
            kpt.S_MM_old = kpt.S_MM.copy()

    def propagate_using_S12(self, time, time_step):
        """PROPAGATE C USING S : S(R+dR)^(1/2) PSI(R+dr) = S(R)^(1/2) PSI(R)"""
        using_blacs = self.wfs.ksl.using_blacs

        if using_blacs is True:
            nao = self.wfs.ksl.nao
            MM_descriptor = self.wfs.ksl.blockgrid.new_descriptor(nao, nao,
                                                                  nao, nao)
            mm_block_descriptor = self.wfs.ksl.mmdescriptor
            mm2MM = Redistributor(self.wfs.ksl.block_comm,
                                  mm_block_descriptor,
                                  MM_descriptor)
            # MM2mm = Redistributor(self.wfs.ksl.block_comm,
            #                       MM_descriptor,
            #                       mm_block_descriptor)

            for kpt in self.wfs.kpt_u:
                S_MM = (kpt.S_MM.copy())
                S_MM_full = MM_descriptor.empty(dtype=S_MM.dtype)
                mm2MM.redistribute(S_MM, S_MM_full)

                S_MM_old_full = MM_descriptor.empty(dtype=kpt.S_MM_old.dtype)
                mm2MM.redistribute(kpt.S_MM_old, S_MM_old_full)

                # if self.world.rank == 0:
                if self.density.gd.comm.rank == 0:
                    T1, Seig_v = schur(S_MM_full, output='real')
                    Seig = eigvals(T1)
        
                    Seig_dm12 = np.diag(1 / np.sqrt(Seig))
                    # Seig_dp12 = np.diag(Seig**0.5)
        
                    # S^1/2
                    Sm12 = np.matmul(Seig_v, np.matmul((Seig_dm12),
                                     np.conj(Seig_v).T))
                    # Sp12 = np.matmul(Seig_v, np.matmul((Seig_dp12),
                    #                 np.conj(Seig_v).T))

                    # OLD OVERLAP S^-1/2
                    T2_o, Seig_v_o = schur(S_MM_old_full, output='real')
                    Seig_o = eigvals(T2_o)

                    # Seig_dm12_o = np.diag(Seig_o**-0.5)
                    Seig_dp12_o = np.diag(Seig_o**0.5)
                    # Sm12_o = np.matmul(Seig_v_o, np.matmul(Seig_dm12_o,
                    #                    np.conj(Seig_v_o).T))
                    Sp12_o = np.matmul(Seig_v_o, np.matmul(Seig_dp12_o,
                                       np.conj(Seig_v_o).T))
                    C_nM_temp = kpt.C_nM.copy()

                    # propagate PSI(R+dr) = S(R+dR)^(-1/2)S(R)^(1/2) PSI(R))
                    Sp12xC_nM = np.matmul(Sp12_o, np.transpose(C_nM_temp))
                    Sm12xSp12xC_nM = np.matmul(Sm12, Sp12xC_nM)
                    t_Sm12xSp12xC_nM = np.transpose(Sm12xSp12xC_nM)
                    kpt.C_nM = t_Sm12xSp12xC_nM.copy()
        #            kpt.C_nM[:,:]=t_Sm12xSp12xC_nM[:,:]
                self.density.gd.comm.broadcast(kpt.C_nM, 0)
                self.td_hamiltonian.update()
        #            self.call_observers(self.niter)
        else:
            for kpt in self.wfs.kpt_u:
                S_MM = (kpt.S_MM.copy())
                T1, Seig_v = schur(S_MM, output='real')
                Seig = eigvals(T1)

                Seig_dm12 = np.diag(1 / np.sqrt(Seig))
                # Seig_dp12 = np.diag(Seig**0.5)
    
                # S^1/2
                Sm12 = np.matmul(Seig_v, np.matmul((Seig_dm12),
                                 np.conj(Seig_v).T))
                # Sp12 = np.matmul(Seig_v, np.matmul((Seig_dp12),
                #                  np.conj(Seig_v).T))
     
                # OLD OVERLAP S^-1/2
                T2_o, Seig_v_o = schur(kpt.S_MM_old, output='real')
                Seig_o = eigvals(T2_o)

                # Seig_dm12_o = np.diag(Seig_o**-0.5)
                Seig_dp12_o = np.diag(Seig_o**0.5)
                # Sm12_o = np.matmul(Seig_v_o, np.matmul(Seig_dm12_o,
                #                    np.conj(Seig_v_o).T))
                Sp12_o = np.matmul(Seig_v_o, np.matmul(Seig_dp12_o,
                                   np.conj(Seig_v_o).T))
                C_nM_temp = kpt.C_nM.copy()

                # propagate PSI(R+dr) = S(R+dR)^(-1/2)S(R)^(1/2) PSI(R))
                Sp12xC_nM = np.matmul(Sp12_o, np.transpose(C_nM_temp))
                Sm12xSp12xC_nM = np.matmul(Sm12, Sp12xC_nM)
                t_Sm12xSp12xC_nM = np.transpose(Sm12xSp12xC_nM)
                kpt.C_nM = t_Sm12xSp12xC_nM.copy()

    #            kpt.C_nM[:,:]=t_Sm12xSp12xC_nM[:,:]

                self.td_hamiltonian.update()
    #            self.call_observers(self.niter)
        return time + time_step
