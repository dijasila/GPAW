# This makes sure that division works as in Python3
from __future__ import division, print_function

import sys
import os
import functools
from math import pi

import numpy as np

from ase.utils import opencew, devnull
from ase.utils.timing import timer, Timer
from ase.units import Hartree
from ase.parallel import paropen

import gpaw
from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.response.selfenergy import GWSelfEnergy
from gpaw.xc.exx import EXX
from gpaw.xc.tools import vxc
import gpaw.io.tar as io

class GWQuasiParticleCalculator:
    """Class for calculating quasiparticle energies using the GW approximation
    for the self-energy. This is done by solving the quasi-particle equation
    using the wavefunctions and eigenvalues from a DFT calculation as an
    initial guess and estimating the solutions using the Newton-Raphson root
    finding method. The calculation can be iterated updating the energies in
    the calculation of the the Green's function and the screened interaction.
    Calculation of the offdiagonal components of the self-energy and thereby
    calculation of the self-consistent quasiparticle wavefunctions are
    currently not implemented."""
    def __init__(self, calc, kpts=None, bandrange=None,
                 filename=None, txt=sys.stdout, savechi0=False, temp=False,
                 ecut=150., nbands=None, domega0=0.025, omega2=10.,
                 qptint=None, truncation='3D',
                 nblocks=1, world=mpi.world):
        """Creates a new quasiparticle calculator.

        Parameters:
        calc: str or PAW object
            GPAW calculator object or filename of saved calculator object.
        kpts: list
            List of indices of the IBZ k-points to calculate the quasi-particle
            energies for.
        bandrange: tuple
            Range of band indices, like (n1, n2+1), to calculate the quasi-
            particle energies for. Note that the second band index is not
            included.
        filename: str
            Base filename for output files. If None no output files are saved.
        txt: str or file object
            Filename or file object of main output file. By default it used the
            system standard output.
        scratch: str
            Path to directory where large temporary files should be stored.
        savechi0: bool
            Determines whether the response function matrices should be stored.
            This makes calculations with extrapolation to infinite planewave
            cutoff much faster but takes up more disk space. The large files
            will be stored in the directory given by the 'scratch' parameter.
        ecut: float or list
            Planewave expansion cut-off energy in eV. If a single number, an
            automatic extrapolation of the selfenergy to infinite cutoff will
            be performed. If a list of several numbers is given, the
            extrapolation will be done using those numbers. If a list with only
            one number is given no extrapolation will be done.
        nbands: int
            Number of bands to use in the calculation. If :None: the number
            will be determined from :ecut: to yield a number close to the
            number of plane waves used.
        domega0: float
            Minimum frequency step (in eV) used in the generation of the non-
            linear frequency grid.
        omega2: float
            Control parameter for the non-linear frequency grid, equal to the
            frequency where the grid spacing has doubled in size.
        qptint: QPointIntegration object
            Object for performing the q-point integration. By default it uses
            a standard Gauss-Fourier quadrature method.
        truncation: str or CoulombKernel object
            Specifies which Coulomb potential to use. Can be '3D', '2D' or
            'wigner-seitz'.
        nblocks: int
            Number of G-vector blocks to parallelize. If > 1 the matrices are
            distributed over :nblocks: cpus to lower the memory requirement on
            each cpu.
        """
        # Create output buffer
        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w', 1)
        self.fd = txt

        self.timer = Timer()

        with self.timer('Read ground state'):
            if isinstance(calc, str):
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

        if bandrange is None:
            bandrange = (0, calc.get_number_of_bands())
        self.bandrange = bandrange

        self.nspins = calc.get_number_of_spins()

        self.shape = (self.nspins, len(kpts), bandrange[1] - bandrange[0])

        self.filename = filename
        #self.scratch = scratch
        self.savechi0 = savechi0
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
        self.qptint = qptint
        self.truncation = truncation

        self.nblocks = nblocks
        self.world = world

        na, nb = self.bandrange
        self.eps_sin = np.array(
            [[self.calc.get_eigenvalues(kpt=k, spin=s)[na:nb]
              for k in self.kpts]
              for s in range(self.nspins)]) / Hartree
        self.f_sin = np.array(
            [[self.calc.get_occupation_numbers(kpt=k, spin=s)[na:nb]
              for k in self.kpts]
             for s in range(self.nspins)])

        print(self.eps_sin.shape)

        omegamax = np.amax(self.eps_sin) - np.amin(self.eps_sin) + 5.0

        self.print_header()

        self.selfenergy = GWSelfEnergy(self.calc,
                                       kpts=self.kpts,
                                       bandrange=self.bandrange,
                                       filename=self.filename,
                                       txt=self.filename + '.sigmac.txt',
                                       temp=temp,
                                       ecut=ecut,
                                       nbands=nbands,
                                       domega0=self.domega0 * Hartree,
                                       omega2=self.omega2 * Hartree,
                                       omegamax=omegamax,
                                       qptint=self.qptint,
                                       truncation=self.truncation,
                                       nblocks=self.nblocks,
                                       world=self.world,
                                       timer=self.timer)
        
        self.selfenergy.addEventHandler('progress',
                                        self.print_selfenergy_progress)
        self.print_parameters()
        self.reset()

    def reset(self):
        """Resets the iterations and sets the quasiparticle energies to the
        Kohn-Sham values."""
        self.iter = 0
        self.qp_sin = self.eps_sin.copy()

    @timer('Quasi-particle equation')
    def calculate(self, niter=1, mixing=1.0, overwrite=False, updatew=False):
        """Calculate the quasiparticle energies after a number of iterations
        of the quasiparticle equation."""

        if self.iter == 0:
            self.calculate_ks_xc_contribution(overwrite=overwrite)
            print('', file=self.fd)
            # The exchange contribution does not depend on the quasiparticle
            # energies - only the wavefunctions so we only have to do this
            # once. In full scQPGW this has to be recalculated at every
            # iteration.

        for i in range(niter):
            print('', file=self.fd)
            print('--------------------------------------', file=self.fd)
            print('Iteration {0:d}'.format(self.iter + 1), file=self.fd)
            print('--------------------------------------', file=self.fd)
            self.update_energies(mixing=mixing)
            
            self.calculate_exchange_contribution(overwrite=overwrite)
            print('', file=self.fd)
            self.calculate_correlation_contribution(updatew=updatew)
            print('', file=self.fd)

            self.qp_sin = self.qp_sin + self.Z_sin * (self.eps_sin -
                            self.vxc_sin - self.qp_sin + self.exx_sin +
                            self.sigma_sin)
            self.iter += 1
            print('Iteration %d done.' % self.iter,
                  file=self.fd)
            #print(self.qp_sin * Hartree, file=self.fd)
            self.print_qp_energies()
        
        self.timer.write(self.fd)
        return self.qp_sin * Hartree
    
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
    def calculate_ks_xc_contribution(self, overwrite=False):
        name = self.filename + '.vxc.npy'
        if not overwrite:
            try:
                fd = open(name, 'rb')
            except IOError:
                pass
            else:
                print('Reading Kohn-Sham XC contribution from file: ' + name,
                  file=self.fd)
                self.vxc_sin = np.load(fd)
                assert self.vxc_sin.shape == self.shape, self.vxc_sin.shape
                return
        
        print('Calculating Kohn-Sham XC contribution', file=self.fd)
        if self.reader is not None:
            self.calc.wfs.read_projections(self.reader)
        vxc_skn = vxc(self.calc, self.calc.hamiltonian.xc) / Hartree
        n1, n2 = self.bandrange
        self.vxc_sin = vxc_skn[:, self.kpts, n1:n2]

        with paropen(name, 'wb') as fd:
            np.save(fd, self.vxc_sin)
        
    @timer('Exchange self-energy')
    def calculate_exchange_contribution(self, overwrite=False):
        """Calculate the exchange self-energy = Fock-potential/exact exchange
        contribution."""
        name = self.filename + '.exx.npy'
        if not overwrite:
            try:
                fd = open(name, 'rb')
            except IOError:
                pass
            else:
                print('Reading exchange self-energy contribution from file:',
                      name, file=self.fd)
                self.exx_sin = np.load(fd)
                assert self.exx_sin.shape == self.shape, self.exx_sin.shape
                return
        
        print('Calculating exchange self-energy contribution', file=self.fd)
        exx = EXX(self.calc, kpts=self.kpts, bands=self.bandrange,
                  txt=self.filename + '.exx.txt', timer=self.timer)
        exx.calculate()
        self.exx_sin = exx.get_eigenvalue_contributions() / Hartree
        with paropen(name, 'wb') as fd:
            np.save(fd, self.exx_sin)

    @timer('Correlation self-energy')
    def calculate_correlation_contribution(self, updatew=False):
        """Calculate the correlation self-energy."""
        print('Calculating correlation self-energy contribution')
        self.selfenergy.calculate(readw=not updatew)
        self.sigma_sin = self.selfenergy.sigma_skn
        self.dsigma_sin = self.selfenergy.dsigma_skn
        self.Z_sin = 1. / (1 - self.dsigma_sin)

    def print_header(self):
        from gpaw.version import version
        import platform
        import time
        p = functools.partial(print, file=self.fd)
        p('  ___  _ _ _ ')
        p(' |   || | | |')
        p(' | | || | | |')
        p(' |__ ||_____|')
        p(' |___|        GPAW {0:s}'.format(version))
        p()

        uname = platform.uname()
        p('User:  ' + os.getenv('USER', '???') + '@' + uname[1])
        p('Date:  {0}'.format(time.asctime()))
        p('Arch:  {0}'.format(uname[4]))
        p('Pid:   {0}'.format(os.getpid()))
        p('gpaw:  {0}'.format(os.path.dirname(gpaw.__file__)))
        p('units: Angstrom and eV')
        p('cores: {0:d}'.format(self.world.size))
        if gpaw.extra_parameters:
            p('Extra parameters: {0}'.format(gpaw.extra_parameters))
        p('----------------------------------')
        p()

    def print_parameters(self):
        p = functools.partial(print, file=self.fd)
        p('Calculating quasi-particle energies for the states:')
        p('    IBZ k-points: ' + ', '.join(['{0:d}'.format(k) for k in self.kpts]))
        p('    Band range:   ({0:d}, {1:d})'.format(self.bandrange[0],
                                                    self.bandrange[1]))
        p('    Spins:        {0:d}'.format(self.nspins))
        p()
        p('Parameters for the correlation self-energy:')
        p('    PW cut-off:        {0:g} eV'.format(self.selfenergy.ecut * Hartree))
        p('    Number of bands:   {0:d}'.format(self.selfenergy.nbands))
        p('    Coulomb potential: {0}'.format(self.truncation))
        p()

    def print_selfenergy_progress(self, progress):
        print('%.f %%' % (100.0 * progress), file=self.fd)

    def print_qp_energies(self):
        p = functools.partial(print, file=self.fd)
        p('Quasi-particle energies:')
        
        if len(self.kpts) > 10:
            p('Warning: Showing only the first 10 k-points')
            print_range = 10
        else:
            print_range = len(self.kpts)
        
        if self.nspins > 1:
            p('                      Up                     Down')
            p(' Kpt  Band     Energy    Occupancy     Energy    Occupancy')
        else:
            p(' Kpt  Band     Energy    Occupancy')
        
        for k, kpt in enumerate(self.kpts):
            for n, band in enumerate(range(*self.bandrange)):
                line = ' {0:2d}   {1:3d}'.format(kpt, band)
                for s in range(self.nspins):
                    line += '  {0:11.5f}  {1:9.5f}'.format(
                        self.qp_sin[s, k, n] * Hartree, self.f_sin[s, k, n])
                p(line)
            p()
                                                         
