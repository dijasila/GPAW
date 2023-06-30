import os

import numpy as np
from typing import Optional
from gpaw.typing import Any, Vector

from ase.units import Bohr, Hartree

from gpaw.calculator import GPAW
from gpaw.external import ExternalPotential, ConstantElectricField
from gpaw.lcaotddft.hamiltonian import TimeDependentHamiltonian
from gpaw.lcaotddft.logger import TDDFTLogger
from gpaw.lcaotddft.propagators import create_propagator
from gpaw.tddft.units import attosec_to_autime


def LCAOTDDFT(filename: str, **kwargs) -> Any:
    if os.environ.get('GPAW_NEW'):
        from gpaw.new.rttddft import RTTDDFT
        assert kwargs.get('propagator', None) in [None, 'ecn'], \
            'Not implemented yet'
        assert kwargs.get('rremisison', None) in [None], 'Not implemented yet'
        assert kwargs.get('fxc', None) in [None], 'Not implemented yet'
        assert kwargs.get('scale', None) in [None], 'Not implemented yet'
        assert kwargs.get('parallel', None) in [None], 'Not implemented yet'
        assert kwargs.get('communicator', None) in [None], \
            'Not implemented yet'
        new_tddft = RTTDDFT.from_dft_file(filename)
        return new_tddft
    return OldLCAOTDDFT(filename, **kwargs)


class OldLCAOTDDFT(GPAW):
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
    keep_static_external_potential
        Experimental option (use carefully).
        Whether to keep the external potential included in the DFT calculation
        (default: True).
        Setting this to False corresponds to sudden switching off of
        the external potential.
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
                 keep_static_external_potential: bool = None,
                 parallel: dict = None,
                 communicator: object = None,
                 txt: str = '-'):
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
        self.keep_static_external_potential = keep_static_external_potential
        self.default_parameters = GPAW.default_parameters.copy()
        self.default_parameters['symmetry'] = {'point_group': False}

        GPAW.__init__(self, filename, parallel=parallel,
                      communicator=communicator, txt=txt)
        # Note! Here happens implicitly self.read() that
        #       GPAW.__init__() triggers!

        if self.keep_static_external_potential is None:
            # Set the default value if parameter is not set or read
            self.keep_static_external_potential = True
        if not self.keep_static_external_potential:
            # Discard the external potential but keep it in
            # the parameter listing
            self.hamiltonian.vext = None

        self.set_positions()

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
                    propagator=self.propagator.todict(),
                    keep_static_external_potential=self.keep_static_external_potential,  # noqa: E501
                    )
            self.td_hamiltonian.write(w.child('td_hamiltonian'))

    def read(self, filename):
        reader = GPAW.read(self, filename)
        if 'tddft' in reader:
            r = reader.tddft
            self.time = r.time
            self.niter = r.niter
            self.kick_strength = r.kick_strength
            if self.keep_static_external_potential is None:
                # Read this parameter only if it was not reset in __init__()
                self.keep_static_external_potential = \
                    r.get('keep_static_external_potential', None)
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

    def absorption_kick(self, kick_strength: Vector):
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
