# This makes sure that division works as in Python3
from __future__ import division, print_function

import sys
import os
import functools
from math import pi
import pickle

import numpy as np

from ase.utils import opencew, devnull
from ase.utils.timing import timer, Timer
from ase.units import Hartree
from ase.parallel import paropen

import gpaw
from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.response.selfenergy import GWSelfEnergy
from gpaw.utilities.progressbar import ProgressBar
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
    def __init__(self, filename=None, txt=sys.stdout,
                 calc=None, kpts=None, bandrange=None,
                 maxiter=1, convergence=0.05, mixing=0.2,
                 savechi0=False, temp=False, savepair=False, updatew=False,
                 ecut=150., nbands=None, domega0=0.025, omega2=10.,
                 qptint=None, truncation='3D',
                 nblocks=1, world=mpi.world):
        """Creates a new quasiparticle calculator.

        Parameters:
        filename: str
            Base filename for output files. If None no output files are saved.
        txt: str or file object
            Filename or file object of main output file. By default it used the
            system standard output.
        calc: str or PAW object
            GPAW calculator object or filename of saved calculator object.
        kpts: list
            List of indices of the IBZ k-points to calculate the quasi-particle
            energies for.
        bandrange: tuple
            Range of band indices, like (n1, n2+1), to calculate the quasi-
            particle energies for. Note that the second band index is not
            included.
        maxiter: int
            Maximum number of self-consistent iterations. Default is 1 which is
            equal to G0W0
        convergence: float
            Convergence criterion for self-consistent loop. If the maximum
            change for all qp energies is less than this threshold the
            calculation will be considered converged and will stop.
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

        self.filename = filename
        self.world = world

        # Initialize variables
        self.eps_skn = None
        self.qp_skn = None
        self.vxc_skn = None
        self.exx_skn = None
        self.sigma_skn = None
        self.Z_skn = None
        self.f_skn = None

        self.print_header()

        self.calc_file = None
        
        # We start by seeing if there is a previous calculation
        if self.load_restart_file():
            print('State loaded from file: %s' % (self.filename + '.pckl'),
                  file=self.fd)
            print('Calculation done for the states:', file=self.fd)
            self.print_states()
            #print('', file=self.fd)
            #self.print_parameters()
            self.print_qp_energies()

        if calc is None and self.calc_file is not None:
            # Use previous calculator file
            calc = self.calc_file

        with self.timer('Read ground state'):
            if isinstance(calc, str):
                if not calc.split('.')[-1] == 'gpw':
                    calc = calc + '.gpw'
                    self.calc_file = calc
                self.reader = io.Reader(calc, comm=mpi.serial_comm)
                print('Reading ground state calculation from file: %s' % calc,
                      file=self.fd)
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

        self.convergence = convergence / Hartree
        self.maxiter = maxiter
        self.mixing = mixing

        self.ecut = ecut
        self.nbands = nbands
        
        #self.scratch = scratch
        self.savechi0 = savechi0
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
        self.qptint = qptint
        self.truncation = truncation

        self.nblocks = nblocks

        # Get data from ground state
        self.ibzk_kc = calc.get_ibz_k_points()
        na, nb = self.bandrange
        self.eps_skn = np.array(
            [[self.calc.get_eigenvalues(kpt=k, spin=s)[na:nb]
              for k in self.kpts]
              for s in range(self.nspins)]) / Hartree
        self.f_skn = np.array(
            [[self.calc.get_occupation_numbers(kpt=k, spin=s)[na:nb]
              for k in self.kpts]
             for s in range(self.nspins)])
        
        # Save list of original DFT eigenvalues so that the calculation can be
        # reset later. The reason is that we overwrite the eigenvalues of the
        # calculator object with the quasiparticle energies, so after an
        # iteration we may have lost the original KS values.
        nibzk = len(self.ibzk_kc)
        self.eps0_skn = np.array([[self.calc.get_eigenvalues(kpt=k, spin=s)
                                   for k in range(nibzk)]
                                  for s in range(self.nspins)]) / Hartree

        # Prepare for a progress bar
        self.pb = None
        
        self.selfenergy = GWSelfEnergy(self.calc,
                                       kpts=self.kpts,
                                       bandrange=self.bandrange,
                                       filename=self.filename,
                                       txt=self.filename + '.sigmac.txt',
                                       temp=temp,
                                       savechi0=savechi0,
                                       savepair=savepair,
                                       ecut=ecut,
                                       nbands=nbands,
                                       domega0=self.domega0 * Hartree,
                                       omega2=self.omega2 * Hartree,
                                       qptint=self.qptint,
                                       truncation=self.truncation,
                                       nblocks=self.nblocks,
                                       world=self.world,
                                       timer=self.timer)
        
        # Track progress events from the self energy
        self.selfenergy.addEventHandler('progress',
                                        self.print_selfenergy_progress)

    def reset(self):
        """Resets the iterations and sets the quasiparticle energies to the
        Kohn-Sham values."""
        self.iter = 0
        self.qp_skn = self.eps_skn.copy()
        self.qp_iskn = np.array([self.qp_skn])

    @timer('Quasi-particle equation')
    def calculate(self):
        """Calculate the quasiparticle energies after a number of iterations
        of the quasiparticle equation."""
        # Reset calculation if previous calculation has not been loaded
        if self.qp_skn is None:
            self.reset()
        
        print('Calculating quasiparticle energies for the states:',
              file=self.fd)
        self.print_states()
        print('', file=self.fd)
        #self.print_parameters()

        self.calculate_ks_xc_contribution(overwrite=False)
        print('', file=self.fd)
        # The exchange contribution does not depend on the quasiparticle
        # energies - only the wavefunctions so we only have to do this
        # once. In full scQPGW this has to be recalculated at every
        # iteration.
        if len(self.qp_iskn) > 1:
            dqp_skn = self.qp_iskn[-1] - self.qp_iskn[-2]
        else:
            dqp_skn = np.zeros(self.shape)
        
        # Do self-consistent loop until convergence or maxiter is reached
        while ((np.amax(np.absolute(dqp_skn)) > self.convergence or
                self.iter == 0) and self.iter < self.maxiter):
            print('', file=self.fd)
            print('--------------------------------------', file=self.fd)
            print('Iteration {0:d}'.format(self.iter + 1), file=self.fd)
            print('--------------------------------------', file=self.fd)
            self.update_energies(mixing=self.mixing)
            
            self.calculate_exchange_contribution(overwrite=False)
            print('', file=self.fd)
            self.calculate_correlation_contribution(updatew=False)
            print('', file=self.fd)

            # Solve the QP equation using the Newton-Raphson root finding
            # method
            qp_skn = self.qp_skn + self.Z_skn * (self.eps_skn -
                        self.vxc_skn - self.qp_skn + self.exx_skn +
                        self.sigma_skn)
            
            # Calculate change in QP energies for convergence check
            dqp_skn = qp_skn - self.qp_skn
            self.qp_skn = qp_skn
            
            self.qp_iskn = np.concatenate((self.qp_iskn,
                                           np.array([self.qp_skn])))
            
            self.save_restart_file()
            print('Iteration {0:d} done.'.format(self.iter + 1), file=self.fd)
            self.print_qp_energies()
            self.iter += 1
        
        self.timer.write(self.fd)

    def get_qp_energies(self):
        if self.qp_skn is None:
            self.calculate()
        return self.qp_skn * Hartree
    
    def update_energies(self, mixing):
        """Updates the energies of the calculator with the quasi-particle
        energies."""
        shifts_skn = np.zeros(self.shape)
        na, nb = self.bandrange
        i1 = len(self.qp_iskn) - 2
        i2 = i1 + 1
        if i1 < 0:
            i1 = 0
        #print('eps0_skn=%s' % self.eps0_skn[:, :, na:nb])
        for kpt in self.calc.wfs.kpt_u:
            s = kpt.s
            if kpt.k in self.kpts:
                ik = self.kpts.index(kpt.k)
                eps1_n = self.mixer(self.qp_iskn[i1, s, ik],
                                    self.qp_iskn[i2, s, ik],
                                    mixing)
                kpt.eps_n[na:nb] = eps1_n
                # Should we do something smart with the bands outside the interval?
                # Here we shift the unoccupied bands not included by the average
                # change of the top-most band and the occupied by the bottom-most
                # band included
                shifts_skn[s, ik] = (eps1_n - self.eps0_skn[s, kpt.k, na:nb])
        
        for kpt in self.calc.wfs.kpt_u:
            s = kpt.s
            if kpt.k in self.kpts:
                ik = self.kpts.index(kpt.k)
                kpt.eps_n[:na] = (self.eps0_skn[s, kpt.k, :na] +
                                  np.mean(shifts_skn[s, :, 0]))
                kpt.eps_n[nb:] = (self.eps0_skn[s, kpt.k, nb:] +
                                  np.mean(shifts_skn[s, :, -1]))
            else:
                """
                kpt.eps_n[:na] = (self.eps0_skn[s, kpt.k, :na] +
                                  np.mean(shifts_skn[s, :, 0]))
                kpt.eps_n[na:nb] = (self.eps0_skn[s, kpt.k, na:nb] +
                                    np.mean(shifts_skn[s, :], axis=0))
                kpt.eps_n[nb:] = (self.eps0_skn[s, kpt.k, nb:] +
                                  np.mean(shifts_skn[s, :, -1]))
                """
                pass

    def mixer(self, e0_skn, e1_skn, mixing=1.0):
        """Mix energies."""
        return e0_skn + mixing * (e1_skn - e0_skn)

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
                self.vxc_skn = np.load(fd)
                assert self.vxc_skn.shape == self.shape, self.vxc_skn.shape
                return
        
        print('Calculating Kohn-Sham XC contribution', file=self.fd)
        if self.reader is not None:
            self.calc.wfs.read_projections(self.reader)
        vxc_skn = vxc(self.calc, self.calc.hamiltonian.xc) / Hartree
        n1, n2 = self.bandrange
        self.vxc_skn = vxc_skn[:, self.kpts, n1:n2]

        with paropen(name, 'wb') as fd:
            np.save(fd, self.vxc_skn)
        
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
                self.exx_skn = np.load(fd)
                assert self.exx_skn.shape == self.shape, self.exx_skn.shape
                return
        
        print('Calculating exchange self-energy contribution', file=self.fd)
        exx = EXX(self.calc, kpts=self.kpts, bands=self.bandrange,
                  txt=self.filename + '.exx.txt', timer=self.timer)
        exx.calculate()
        self.exx_skn = exx.get_eigenvalue_contributions() / Hartree
        with paropen(name, 'wb') as fd:
            np.save(fd, self.exx_skn)

    @timer('Correlation self-energy')
    def calculate_correlation_contribution(self, updatew=False):
        """Calculate the correlation self-energy."""
        print('Calculating correlation self-energy contribution', file=self.fd)
        self.pb = ProgressBar(self.fd)
        self.pb.update(0.0)
        self.selfenergy.calculate(readw=not updatew)
        self.sigma_skn = self.selfenergy.sigma_skn.real
        self.dsigma_skn = self.selfenergy.dsigma_skn.real
        self.Z_skn = 1. / (1 - self.dsigma_skn)
        self.pb.finish()

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

    def print_states(self):
        p = functools.partial(print, file=self.fd)
        p('    IBZ k-points: ' + ', '.join(['{0:d}'.format(k) for k in self.kpts]))
        p('    Band range:   ({0:d}, {1:d})'.format(self.bandrange[0],
                                                    self.bandrange[1]))
        p('    Spins:        {0:d}'.format(self.nspins))
        p()

    def print_parameters(self):
        print(self.nbands)
        p = functools.partial(print, file=self.fd)
        p('Parameters for the self-consistency cycle:')
        p('    Energy convergence threshold: {0:g} eV'.format(self.convergence))
        p('    Maximum number of iterations: {0:d}'.format(self.maxiter))
        p('    Energy mixing:                {0:g}'.format(self.mixing))
        p()
        p('Parameters for the correlation self-energy:')
        p('    PW cut-off:        {0:g} eV'.format(self.ecut * Hartree))
        p('    Number of bands:   {0:d}'.format(self.nbands))
        p('    Coulomb potential: {0}'.format(self.truncation))
        p()

    def print_selfenergy_progress(self, progress):
        self.pb.update(progress)
        #print('%.f %%' % (100.0 * progress), file=self.fd)

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
        
        maxk = min(10, len(self.kpts))
        for k, kpt in enumerate(self.kpts[:maxk]):
            k_c = self.ibzk_kc[kpt]
            p(' {0:2d}: ({1:.3f}, {2:.3f}, {3:.3f})'.format(
                kpt, *tuple(k_c)))
            for n, band in enumerate(range(*self.bandrange)):
                line = '      {1:3d}'.format(kpt, band)
                for s in range(self.nspins):
                    line += '  {0:11.5f}  {1:9.5f}'.format(
                        self.qp_skn[s, k, n] * Hartree, self.f_skn[s, k, n])
                p(line)
            p()

    
    def save_iteration(self):
        """Save quasi-particle energies to a file so that they can be loaded
        later to restart a calculation or continue iterations."""
        data = {'kpts': self.kpts,
                'bandrange': self.bandrange,
                'iter': self.iter,
                'qp_iskn': self.qp_iskn}
        if self.filename:
            pickle.dump(data, paropen(self.filename + '.pckl', 'wb'))
    
    
    def load_iteration(self):
        """Load quasi-particle energies from saved file. Returns True/False
        whether saved quasi-particle energies were successfully loaded or
        not."""
        if self.filename:
            pckl = self.filename + '.pckl'
            try:
                data = pickle.load(open(pckl, 'rb'))
                self.calculate_ks_xc_contribution()
                self.calculate_exchange_contribution()
                self.sigma_skn = self.selfenergy.sigma_skn.real
                self.dsigma_skn = self.selfenergy.dsigma_skn.real
                self.Z_skn = 1. / (1 - self.dsigma_skn)
            except IOError:
                return False
            else:
                self.kpts = data['kpts']
                self.bandrange = data['bandrange']
                self.iter = data['iter']
                self.qp_iskn = data['qp_iskn']
                self.qp_skn = self.qp_iskn[-1]
                self.nspins = self.qp_skn.shape[0]
                return True
        else:
            return False
                
    def load_restart_file(self):
        """Loads data from previous calculation so that it can be restarted.
        If there is no file or it cannot be read False is returned. True is
        returned if the data is read successfully."""
        if self.filename:
            pckl = self.filename + '.pckl'
            try:
                # Open on all nodes so that they all have the same parameters
                data = pickle.load(open(pckl, 'rb'))
            except IOError:
                return False
            else:
                if 'mixing' not in data:
                    # Old format
                    return False
                
                # Get parameters
                self.calc_file = data['calc_file']
                self.kpts = data['kpts']
                self.bandrange = data['bandrange']
                self.nspins = data['nspins']
                self.ibzk_kc = data['ibzk_kc']
                self.ecut = data['ecut']
                self.nbands = data['nbands']
                self.domega0 = data['domega0']
                self.omega2 = data['omega2']
                self.maxiter = data['maxiter']
                self.convergence = data['convergence']
                self.mixing = data['mixing']
                # Get state
                self.iter = data['iter']
                self.qp_iskn = data['qp_iskn']
                self.qp_skn = data['qp_skn']
                self.f_skn = data['f_skn']
                self.vxc_skn = data['vxc_skn']
                self.exx_skn = data['exx_skn']
                self.sigma_skn = data['sigma_skn']
                self.dsigma_skn = data['dsigma_skn']
                self.Z_skn = 1. / (1 - self.dsigma_skn)
                
                return True
        else:
            return False

    def save_restart_file(self):
        """Save parameters and quasi-particle energies to a file so that they
        can be loaded later to restart a calculation or continue iterations."""
        if self.filename:
            data = {'calc_file': self.calc_file,
                    'kpts': self.kpts,
                    'bandrange': self.bandrange,
                    'nspins': self.nspins,
                    'ibzk_kc': self.ibzk_kc,
                    'ecut': self.ecut,
                    'nbands': self.nbands,
                    'domega0': self.domega0,
                    'omega2': self.omega2,
                    'maxiter': self.maxiter,
                    'convergence': self.convergence,
                    'mixing': self.mixing,
                    'iter': self.iter,
                    'qp_iskn': self.qp_iskn,
                    'qp_skn': self.qp_skn,
                    'f_skn': self.f_skn,
                    'vxc_skn': self.vxc_skn,
                    'exx_skn': self.exx_skn,
                    'sigma_skn': self.sigma_skn,
                    'dsigma_skn': self.dsigma_skn}
            
            pickle.dump(data, paropen(self.filename + '.pckl', 'wb'))
