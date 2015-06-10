# This makes sure that division works as in Python3
from __future__ import division, print_function

import sys
from math import pi

import numpy as np

from ase.utils import opencew, devnull
from ase.utils.timing import timer, Timer
from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw.response.selfenergy import GWSelfEnergy
from gpaw.xc.exx import EXX
from gpaw.xc.tools import vxc


class GWQuasiParticleCalculator:

    def __init__(self, calc, kpts=None, bandrange=None,
                 filename=None, txt=sys.stdout, savew=False,
                 ecut=150., nbands=None,
                 qptint=None, truncation='wigner-seitz',
                 nblocks=1, world=mpi.world):

        # Create output buffer
        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt

        self.timer = Timer()

        with self.timer('Read ground state'):
            if isinstance(calc, str):
                print('Reading ground state calculation:\n  %s' % calc,
                      file=self.fd)
                if not calc.split('.')[-1] == 'gpw':
                    calc = calc + '.gpw'
                self.reader = io.Reader(calc, comm=mpi.serial_comm)
                calc = GPAW(calc, txt=None, communicator=mpi.serial_comm,
                            read_projections=False)
            else:
                self.reader = None
                assert calc.wfs.world.size == 1

        assert calc.wfs.kd.symmetry.symmorphic, \
          'Can only handle symmorhpic symmetries at the moment'
        self.calc = calc

        if kpts is None:
            kpts = range(len(calc.get_ibz_k_points()))
        self.kpts = kpts

        self.ecut = ecut / Hartree

        if nbands is None:
            vol = abs(np.linalg.det(self.calc.wfs.gd.cell_cv))
            nbands = min(calc.get_number_of_bands(),
                         int(vol * self.ecut**1.5 * 2**0.5 / 3 / pi**2))
        self.nbands = nbands

        if bandrange is None:
            bandrange = (0, nbands)
        self.bandrange = bandrange

        self.nspins = calc.get_number_of_spins()

        self.shape = (self.nspins, len(kpts), bandrange[1] - bandrange[0])

        self.filename = filename

        eps_sin = np.array([[self.calc.get_eigenvalues(kpt=k, spin=s)
                             for k in range(len(calc.get_ibz_k_points()))]
                             for s in range(self.nspins)])
        omegamax = np.amax(eps_sin) - np.amin(eps_sin) + 10.0

        self.selfenergy = GWSelfEnergy(calc, kpts=kpts, bandrange=bandrange,
                                       filename=filename, txt=txt, savew=savew,
                                       ecut=ecut, nbands=nbands,
                                       omegamax=omegamax,
                                       qptint=qptint, truncation=truncation,
                                       nblocks=nblocks, world=world)

        self.initialize()
        self.reset()

    def initialize(self):
        """Do various initialization tasks like getting the original Kohn-Sham
        eigenvalues from the calculator."""
        # Get Kohn-Sham eigenvalues from the calculator in the beginning - they
        # can be overwritten later.
        na, nb = self.bandrange
        self.eps_sin = np.array(
            [[self.calc.get_eigenvalues(kpt=k, spin=s)[na:nb]
              for k in self.kpts]
              for s in range(self.nspins)]) / Hartree

    def reset(self):
        """Resets the iterations and sets the quasiparticle energies to the
        Kohn-Sham values."""
        self.iter = 0
        self.qp_sin = self.eps_sin.copy()

    @timer('Quasi-particle equation')
    def calculate(self, niter=1, mixing=1.0, updatew=False):
        """Calculate the quasiparticle energies after a number of iterations
        of the quasiparticle equation."""

        if self.iter == 0:
            self.calculate_ks_xc_contribution()
            # The exchange contribution does not depend on the quasiparticle
            # energies - only the wavefunctions so we only have to do this
            # once. In full scQPGW this has to be recalculated at every
            # iteration.
            self.calculate_exchange_contribution()

        for i in range(niter):
            print('Starting iteration %d' % (self.iter + 1), file=self.fd)
            self.update_energies(mixing=mixing)
            
            self.calculate_correlation_contribution(updatew=updatew)

            self.qp_sin = self.qp_sin + self.Z_sin * (self.eps_sin -
                            self.vxc_sin - self.qp_sin + self.exx_sin +
                            self.sigma_sin)
            self.iter += 1
            print('Iteration %d done. Quasiparticle energies:' % self.iter,
                  file=self.fd)
            print(self.qp_sin * Hartree, file=self.fd)
            yield self.qp_sin * Hartree
    
    def update_energies(self, mixing):
        """Updates the energies of the calculator with the quasi-particle
        energies."""
        shifts_sn = np.mean(self.qp_sin - self.eps_sin, axis=1)
        na, nb = self.bandrange
        for kpt in self.calc.wfs.kpt_u:
            kpt.eps_n[na:nb] = self.mixer(kpt.eps_n[na:nb],
                                          self.qp_sin[kpt.s, kpt.k],
                                          mixing)
            # Should we do something smart with the bands outside the interval?
            # Here we shift the unoccupied bands not included by the average
            # change of the top-most band and the occupied by the bottom-most
            # band included
            kpt.eps_n[:na] += shifts_sn[kpt.s, 0]
            kpt.eps_n[nb:] += shifts_sn[kpt.s, -1]

    def mixer(self, e0_sin, e1_sin, mixing=1.0):
        """Mix energies."""
        return e0_sin + mixing * (e1_sin - e0_sin)

    @timer('Kohn-Sham XC-contribution')
    def calculate_ks_xc_contribution(self):
        name = self.filename + '.vxc.npy'
        fd = opencew(name)
        if fd is None:
            print('Reading Kohn-Sham XC contribution from file:', name,
                  file=self.fd)
            with open(name) as fd:
                self.vxc_sin = np.load(fd)
            assert self.vxc_sin.shape == self.shape, self.vxc_sin.shape
            return
            
        print('Calculating Kohn-Sham XC contribution', file=self.fd)
        if self.reader is not None:
            self.calc.wfs.read_projections(self.reader)
        vxc_skn = vxc(self.calc, self.calc.hamiltonian.xc) / Hartree
        n1, n2 = self.bandrange
        self.vxc_sin = vxc_skn[:, self.kpts, n1:n2]
        np.save(fd, self.vxc_sin)
        
    @timer('Exchange self-energy')
    def calculate_exchange_contribution(self):
        """Calculate the exchange self-energy = Fock-potential/exact exchange
        contribution."""
        name = self.filename + '.exx.npy'
        fd = opencew(name)
        if fd is None:
            print('Reading EXX contribution from file:', name, file=self.fd)
            with open(name) as fd:
                self.exx_sin = np.load(fd)
            assert self.exx_sin.shape == self.shape, self.exx_sin.shape
            return
        
        print('Calculating EXX contribution', file=self.fd)
        exx = EXX(self.calc, kpts=self.kpts, bands=self.bandrange,
                  txt=self.filename + '.exx.txt', timer=self.timer)
        exx.calculate()
        self.exx_sin = exx.get_eigenvalue_contributions() / Hartree
        np.save(fd, self.exx_sin)

    @timer('Correlation self-energy')
    def calculate_correlation_contribution(self, updatew=False):
        """Calculate the correlation self-energy."""
        self.selfenergy.calculate(readw=not updatew)
        self.sigma_sin = self.selfenergy.sigma_sin
        self.dsigma_sin = self.selfenergy.dsigma_sin
        self.Z_sin = 1. / (1 - self.dsigma_sin)
