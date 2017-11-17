from __future__ import print_function

import numpy as np
from ase.units import Bohr, Hartree

from gpaw import GPAW
from gpaw.external import ConstantElectricField
from gpaw.utilities.blas import gemm
from gpaw.mixer import DummyMixer
from gpaw.tddft.units import attosec_to_autime
from gpaw.xc import XC
from gpaw.lcaotddft.logger import TDDFTLogger
from gpaw.lcaotddft.propagators import create_propagator


class KickHamiltonian:
    def __init__(self, calc, ext):
        ham = calc.hamiltonian
        dens = calc.density
        vext_g = ext.get_potential(ham.finegd)
        self.vt_sG = [ham.restrict_and_collect(vext_g)]
        self.dH_asp = ham.setups.empty_atomic_matrix(1, ham.atom_partition)

        W_aL = dens.ghat.dict()
        dens.ghat.integrate(vext_g, W_aL)
        # XXX this is a quick hack to get the distribution right
        dHtmp_asp = ham.atomdist.to_aux(self.dH_asp)
        for a, W_L in W_aL.items():
            setup = dens.setups[a]
            dHtmp_asp[a] = np.dot(setup.Delta_pL, W_L).reshape((1, -1))
        self.dH_asp = ham.atomdist.from_aux(dHtmp_asp)


class LCAOTDDFT(GPAW):
    def __init__(self, filename=None,
                 propagator='sicn', fxc=None, **kwargs):
        self.time = 0.0
        self.niter = 0
        self.kick_strength = np.zeros(3)
        self.tddft_initialized = False
        self.action = None
        self.fxc = fxc
        self.propagator = create_propagator(propagator)
        if filename is None:
            kwargs['mode'] = kwargs.get('mode', 'lcao')
        GPAW.__init__(self, filename, **kwargs)

        # Restarting from a file
        if filename is not None:
            # self.initialize()
            self.set_positions()

    def read(self, filename):
        reader = GPAW.read(self, filename)
        if 'tddft' in reader:
            self.time = reader.tddft.time
            self.niter = reader.tddft.niter
            self.kick_strength = reader.tddft.kick_strength
        # TODO: read fxc

    def _write(self, writer, mode):
        GPAW._write(self, writer, mode)
        writer.child('tddft').write(time=self.time,
                                    niter=self.niter,
                                    kick_strength=self.kick_strength)
        # TODO: write fxc

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
        self.update_hamiltonian()

        # Call observers after kick
        self.action = 'kick'
        self.call_observers(self.niter)
        self.timer.stop('Kick')

    def tddft_init(self):
        if self.tddft_initialized:
            return

        self.timer.start('Initialize TDDFT')
        assert self.wfs.dtype == complex
        assert len(self.wfs.kpt_u) == 1

        # Initialize propagator
        self.propagator.initialize(self)

        # Reset the density mixer
        self.density.set_mixer(DummyMixer())

        # Update density and Hamiltonian
        self.update_hamiltonian()

        # Add logger
        TDDFTLogger(self)

        # Call observers before propagation
        self.action = 'init'
        self.call_observers(self.niter)

        if self.niter == 0:
            # Initialize fxc
            if self.fxc is not None:
                for k, kpt in enumerate(self.wfs.kpt_u):
                    kpt.deltaXC_H_MM = self.get_hamiltonian(kpt)
                    self.hamiltonian.xc = XC(self.fxc)
                    self.update_hamiltonian()
                    assert len(self.wfs.kpt_u) == 1  # TODO: Why?
                    kpt.deltaXC_H_MM -= self.get_hamiltonian(kpt)

        self.tddft_initialized = True
        self.timer.stop('Initialize TDDFT')

    def update_projectors(self):  # TODO: move to propagator?
        self.timer.start('LCAO update projectors')
        # Loop over all k-points
        for k, kpt in enumerate(self.wfs.kpt_u):
            for a, P_ni in kpt.P_ani.items():
                P_ni.fill(117)
                gemm(1.0, self.wfs.P_aqMi[a][kpt.q], kpt.C_nM, 0.0, P_ni, 'n')
        self.timer.stop('LCAO update projectors')

    def get_hamiltonian(self, kpt):  # TODO: move to propagator?
        eig = self.wfs.eigensolver
        H_MM = eig.calculate_hamiltonian_matrix(self.hamiltonian, self.wfs,
                                                kpt, root=-1)
        return H_MM

    def update_hamiltonian(self):  # TODO: move to propagator?
        self.update_projectors()
        self.density.update(self.wfs)
        self.hamiltonian.update(self.density)

    def propagate(self, time_step=10, iterations=2000):
        self.tddft_init()

        time_step *= attosec_to_autime
        maxiter = self.niter + iterations

        if self.niter == 0:
            self.log('----  About to do %d propagation steps' % iterations)
        else:
            self.log('----  About to continue from iteration %d and '
                     'do %d more propagation steps' % (self.niter, iterations))

        self.timer.start('Propagate')
        while self.niter < maxiter:
            # Propagate one step
            self.time = self.propagator.propagate(self.time, time_step)

            # Call registered callback functions
            self.action = 'propagate'
            self.call_observers(self.niter)

            self.niter += 1

        self.timer.stop('Propagate')
