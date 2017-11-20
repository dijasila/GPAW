import numpy as np
from ase.units import Bohr, Hartree

from gpaw import GPAW
from gpaw.external import ConstantElectricField
from gpaw.tddft.units import attosec_to_autime
from gpaw.lcaotddft.logger import TDDFTLogger
from gpaw.lcaotddft.propagators import create_propagator
from gpaw.lcaotddft.hamiltonian import TimeDependentHamiltonian
from gpaw.lcaotddft.hamiltonian import KickHamiltonian


class LCAOTDDFT(GPAW):
    def __init__(self, filename=None,
                 propagator='sicn', fxc=None, **kwargs):
        self.time = 0.0
        self.niter = 0
        self.kick_strength = np.zeros(3)
        self.tddft_initialized = False
        self.action = None
        self.td_hamiltonian = TimeDependentHamiltonian(fxc=fxc)

        self.propagator = create_propagator(propagator)
        if filename is None:
            kwargs['mode'] = kwargs.get('mode', 'lcao')
        GPAW.__init__(self, filename, **kwargs)

        # Restarting from a file
        if filename is not None:
            # self.initialize()
            self.set_positions()

    def _write(self, writer, mode):
        GPAW._write(self, writer, mode)
        w = writer.child('tddft')
        w.write(time=self.time,
                niter=self.niter,
                kick_strength=self.kick_strength)
        self.td_hamiltonian.write(w.child('td_hamiltonian'))

    def read(self, filename):
        reader = GPAW.read(self, filename)
        if 'tddft' in reader:
            r = reader.tddft
            self.time = r.time
            self.niter = r.niter
            self.kick_strength = r.kick_strength
            if 'td_hamiltonian' in r:
                self.td_hamiltonian.wfs = self.wfs
                self.td_hamiltonian.read(r.td_hamiltonian)

    def absorption_kick(self, kick_strength):
        self.tddft_init()
        self.timer.start('Kick')
        self.kick_strength = np.array(kick_strength, dtype=float)

        # magnitude
        magnitude = np.sqrt(np.sum(self.kick_strength**2))

        # normalize
        direction = self.kick_strength / magnitude

        self.log('----  Applying absorption kick')
        self.log('----  Magnitude: %.8f hartree/bohr' % magnitude)
        self.log('----  Direction: %.4f %.4f %.4f' % tuple(direction))

        # Create hamiltonian object for absorption kick
        cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)
        kick_hamiltonian = KickHamiltonian(self, cef)

        # Use explicit propagator for the kick
        kick_propagator = create_propagator(name='ecn')
        kick_propagator.initialize(self, kick_hamiltonian)
        for i in range(10):
            kick_propagator.propagate(self.time, 0.1)

        # Update density and Hamiltonian
        self.td_hamiltonian.update()

        # Call observers after kick
        self.action = 'kick'
        self.call_observers(self.niter)

        self.niter += 1
        self.timer.stop('Kick')

    def tddft_init(self):
        if self.tddft_initialized:
            return

        self.timer.start('Initialize TDDFT')
        assert self.wfs.dtype == complex
        assert len(self.wfs.kpt_u) == 1

        # Initialize Hamiltonian
        self.td_hamiltonian.initialize(self)

        # Initialize propagator
        self.propagator.initialize(self)

        # Add logger
        TDDFTLogger(self)

        # Call observers before propagation
        self.action = 'init'
        self.call_observers(self.niter)

        self.tddft_initialized = True
        self.timer.stop('Initialize TDDFT')

    def propagate(self, time_step=10, iterations=2000):
        self.tddft_init()

        time_step *= attosec_to_autime
        maxiter = self.niter + iterations

        self.log('----  About to do %d propagation steps' % iterations)

        self.timer.start('Propagate')
        while self.niter < maxiter:
            # Propagate one step
            self.time = self.propagator.propagate(self.time, time_step)

            # Call registered callback functions
            self.action = 'propagate'
            self.call_observers(self.niter)

            self.niter += 1

        self.timer.stop('Propagate')
