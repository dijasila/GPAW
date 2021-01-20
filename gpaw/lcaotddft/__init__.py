import numpy as np

from ase.units import Bohr, Hartree

from gpaw import GPAW
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft.hamiltonian import TimeDependentHamiltonian
from gpaw.lcaotddft.logger import TDDFTLogger
from gpaw.lcaotddft.propagators import create_propagator
from gpaw.tddft.units import attosec_to_autime

from gpaw.tddft.tdopers import  TimeDependentDensity



class LCAOTDDFT(GPAW):
    def __init__(self, filename=None, propagator=None, scale=None,calculate_energy=True,
                 fxc=None, td_potential=None, **kwargs):
        self.time = 0.0
        self.niter = 0
        # TODO: deprecate kick keywords (and store them as td_potential)
        self.kick_strength = np.zeros(3)
        self.kick_ext = None
        self.tddft_initialized = False
        self.action = None
        tdh = TimeDependentHamiltonian(fxc=fxc, td_potential=td_potential,
                                       scale=scale)
        self.td_hamiltonian = tdh

        self.propagator = propagator
        if filename is None:
            kwargs['mode'] = kwargs.get('mode', 'lcao')
        self.default_parameters = GPAW.default_parameters.copy()
        self.default_parameters['symmetry'] = {'point_group': False}
        GPAW.__init__(self, filename, **kwargs)

        # Restarting from a file
        if filename is not None:
#            self.initialize()
            self.set_positions()

        self.td_density = TimeDependentDensity(self)
        self.calculate_energy = calculate_energy


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
            if self.propagator is None:
                self.propagator = r.propagator
            else:
                self.log('Note! Propagator changed!')
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
        self.propagator = create_propagator(self.propagator)
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
        
    def absorption_kick(self, kick_strength):
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

    def propagate(self, time_step=10, iterations=2000):
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
        self.propagator = create_propagator(**kwargs)
        self.tddft_init()
        self.propagator.control_paw(self)

    def get_td_energy(self):
        ''' THIS NEEDS TO BE MODIFIED FOR LCAO '''
        """Calculate the time-dependent total energy"""

        if not self.calculate_energy:
           self.Etot = 0.0
           return 0.0
#        print ('=====',type(self),dir(self))
#        self.td_overlap.update(self.wfs)
#        self.td_density.update()
#        self.td_hamiltonian.update('density')
#        self.td_hamiltonian.update(self.td_density.get_density(),self.time)
        self.td_hamiltonian.update()
        self.update_eigenvalues()

#        return self.Etot

    def update_eigenvalues(self):
#        print ('in update eig',type(self),type(self.hamiltonian),type(self.wfs))
        self.wfs.eigensolver.iterate(self.hamiltonian, self.wfs) # Vlad ED
        ''' THIS NEEDS TO BE MODIFIED FOR LCAO '''
        '''
        kpt_u = self.wfs.kpt_u
        if self.hpsit is None:
            self.hpsit = self.wfs.gd.zeros(len(kpt_u[0].psit_nG),
                                           dtype=complex)
        if self.eps_tmp is None:
            self.eps_tmp = np.zeros(len(kpt_u[0].eps_n),
                                    dtype=complex)

        # self.Eband = sum_i <psi_i|H|psi_j>
        for kpt in kpt_u:
            self.td_hamiltonian.apply(kpt, kpt.psit_nG, self.hpsit,
                                      calculate_P_ani=False)
            self.mblas.multi_zdotc(self.eps_tmp, kpt.psit_nG,
                                   self.hpsit, len(kpt_u[0].psit_nG))
            self.eps_tmp *= self.wfs.gd.dv
            kpt.eps_n[:] = self.eps_tmp.real

        H = self.td_hamiltonian.hamiltonian

        # Nonlocal
        self.Enlkin = H.xc.get_kinetic_energy_correction()

        # PAW
        e_band = self.wfs.calculate_band_energy()
        self.Ekin = H.e_kinetic0 + e_band + self.Enlkin
        self.e_coulomb = H.e_coulomb
        self.Eext = H.e_external
        self.Ebar = H.e_zero
        self.Exc = H.e_xc
        self.Etot = self.Ekin + self.e_coulomb + self.Ebar + self.Exc
        '''
