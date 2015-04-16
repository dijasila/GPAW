from __future__ import print_function, division

import sys
from time import ctime

import numpy as np
from ase.units import Hartree
from ase.utils.timing import timer, Timer

import gpaw.mpi as mpi
from gpaw import extra_parameters
from gpaw.blacs import (BlacsGrid, BlacsDescriptor, Redistributor,
                        DryRunBlacsGrid)
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.occupations import FermiDirac
from gpaw.response.pair import PairDensity
from gpaw.utilities.memory import maxrss
from gpaw.utilities.blas import gemm, rk, czher, mmm
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.response.pair import PWSymmetryAnalyzer
from gpaw.response.integrators import (TetrahedronIntegrator,
                                       BroadeningIntegrator)
from functools import partial


def frequency_grid(domega0, omega2, omegamax):
    beta = (2**0.5 - 1) * domega0 / omega2
    wmax = int(omegamax / (domega0 + beta * omegamax)) + 2
    w = np.arange(wmax)
    omega_w = w * domega0 / (1 - beta * w)
    return omega_w


class Chi0:
    def __init__(self, calc,
                 frequencies=None, domega0=0.1, omega2=10.0, omegamax=None,
                 ecut=50, hilbert=True, nbands=None,
                 timeordered=False, eta=0.2, ftol=1e-6, threshold=1,
                 real_space_derivatives=False, intraband=True,
                 world=mpi.world, txt=sys.stdout, timer=None,
                 nblocks=1, no_optical_limit=False,
                 keep_occupied_states=False, gate_voltage=None,
                 disable_point_group=False, disable_time_reversal=False,
                 use_more_memory=1, unsymmetrized=True):

        self.pair = PairDensity(calc, ecut, ftol, threshold,
                                real_space_derivatives, world, txt, timer,
                                nblocks=nblocks, gate_voltage=gate_voltage)

        calc = self.pair.calc
        self.calc = calc

        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt

        self.vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))

        self.world = world

        if nblocks == 1:
            self.blockcomm = self.world.new_communicator([world.rank])
            self.kncomm = world
        else:
            assert world.size % nblocks == 0, world.size
            rank1 = world.rank // nblocks * nblocks
            rank2 = rank1 + nblocks
            self.blockcomm = self.world.new_communicator(range(rank1, rank2))
            ranks = range(world.rank % nblocks, world.size, nblocks)
            self.kncomm = self.world.new_communicator(ranks)

        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt

        if ecut is not None:
            ecut /= Hartree

        if gate_voltage is not None:
            gate_voltage /= Hartree

        self.ecut = ecut

        self.eta = eta / Hartree
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
        self.omegamax = None if omegamax is None else omegamax / Hartree
        self.nbands = nbands or self.calc.wfs.bd.nbands
        self.keep_occupied_states = keep_occupied_states

        omax = self.find_maximum_frequency()

        if frequencies is None:
            if self.omegamax is None:
                self.omegamax = omax
            print('Using nonlinear frequency grid from 0 to %.3f eV' %
                  (self.omegamax * Hartree), file=self.fd)
            self.omega_w = frequency_grid(self.domega0, self.omega2,
                                          self.omegamax)
        else:
            self.omega_w = np.asarray(frequencies) / Hartree
            assert not hilbert
        self.hilbert = hilbert
        self.timeordered = bool(timeordered)

        if self.eta == 0.0:
            assert not hilbert
            assert not timeordered
            assert not self.omega_w.real.any()

        # Occupied states:
        wfs = self.calc.wfs

        self.nocc1 = self.pair.nocc1  # number of completely filled bands
        self.nocc2 = self.pair.nocc2  # number of non-empty bands

        self.timer = timer or Timer()

        self.Q_aGii = None

        self.prefactor = 1.0

    def find_maximum_frequency(self):
        self.epsmin = 10000.0
        self.epsmax = -10000.0
        for kpt in self.calc.wfs.kpt_u:
            self.epsmin = min(self.epsmin, kpt.eps_n[0])
            self.epsmax = max(self.epsmax, kpt.eps_n[self.nbands - 1])

        print('Minimum eigenvalue: %10.3f eV' % (self.epsmin * Hartree),
              file=self.fd)
        print('Maximum eigenvalue: %10.3f eV' % (self.epsmax * Hartree),
              file=self.fd)

        return self.epsmax - self.epsmin

    def calculate(self, q_c, spin='all', A_x=None):
        wfs = self.calc.wfs

        if spin == 'all':
            spins = range(wfs.nspins)
        else:
            assert spin in range(wfs.nspins)
            spins = [spin]

        q_c = np.asarray(q_c, dtype=float)
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, wfs.gd, complex, qd)

        self.print_chi(pd)

        if extra_parameters.get('df_dry_run'):
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        nG = pd.ngmax
        nw = len(self.omega_w)
        mynG = (nG + self.blockcomm.size - 1) // self.blockcomm.size
        self.Ga = self.blockcomm.rank * mynG
        self.Gb = min(self.Ga + mynG, nG)
        assert mynG * (self.blockcomm.size - 1) < nG
        
        if A_x is not None:
            nx = nw * (self.Gb - self.Ga) * nG
            chi0_wGG = A_x[:nx].reshape((nw, self.Gb - self.Ga, nG))
            chi0_wGG[:] = 0.0
        else:
            chi0_wGG = np.zeros((nw, self.Gb - self.Ga, nG), complex)

        if np.allclose(q_c, 0.0):
            chi0_wxvG = np.zeros((len(self.omega_w), 2, 3, nG), complex)
            chi0_wvv = np.zeros((len(self.omega_w), 3, 3), complex)
            self.chi0_vv = np.zeros((3, 3), complex)
        else:
            chi0_wxvG = None
            chi0_wvv = None

        # Do all empty bands:
        m1 = self.nocc1
        m2 = self.nbands
        
        self._calculate(pd, chi0_wGG, chi0_wxvG, chi0_wvv, m1, m2, spins)
        
        return pd, chi0_wGG, chi0_wxvG, chi0_wvv

    @timer('Calculate CHI_0')
    def _calculate(self, pd, chi0_wGG, chi0_wxvG, chi0_wvv, m1, m2, spins):
        # The method of integration is determined
        # by the choice of integrator class
        if False:
            integrator = BroadeningIntegrator(self.eta, self.calc.wfs.kd, 0, 
                                              self.nocc2, m1, m2, spins,
                                              comm=self.kncomm)
        else:
            integrator = TetrahedronIntegrator(self.calc.wfs.kd, self.calc.wfs.gd, 0, 
                                               self.nocc2, m1, m2, spins,
                                               comm=self.kncomm)

        # The kind of integral we want to make
        kind = 'response_function'

        # The type of integral that we make determines
        # the function that we use
        integrate = integrator.get_integration_function(kind=kind)

        # The function to be integrated has to have a
        # specific input to work with the integrator
        integrand = partial(self.integrand, pd)

        # Integrate function
        if False:
            integrate(integrand, self.omega_w,
                      hilbert=self.hilbert,
                      timeordered=self.timeordered,
                      hermitian=(self.eta == 0),
                      out_wxx=chi0_wGG)
        else:
            integrate(integrand, self.omega_w, out_wxx=chi0_wGG)

        # Remember the prefactor
        chi0_wGG *= self.prefactor

        return pd, chi0_wGG, chi0_wxvG, chi0_wvv

    def integrand(self, pd, s, k_c, n1, n2, m1, m2):
        """A function that can be integrated.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""

        wfs = self.calc.wfs
        K1 = wfs.kd.where_is_q(k_c, wfs.kd.bzk_kc)
        q_c = pd.kd.bzk_kc[0]
        if self.Q_aGii is None:
            self.Q_aGii = self.pair.initialize_paw_corrections(pd)
            
        kptpair = self.pair.get_kpoint_pair(pd, s, K1, n1, n2, m1, m2)
        kpt1 = kptpair.get_k1()
        kpt2 = kptpair.get_k2()
        n_n = range(n2 - n1)
        eps_n = kpt1.eps_n - self.pair.fermi_level
        eps_m = kpt2.eps_n - self.pair.fermi_level
        f_n = kpt1.f_n
        f_m = kpt2.f_n

        m_m = range(0, kpt2.n2 - kpt2.n1)
        deps_nm = kptpair.get_transition_energies(n_n, m_m)
        n_nmG, _, _ = self.pair.get_pair_density(pd, kptpair, n_n, m_m,
                                                 Q_aGii=self.Q_aGii)
        n_nmG[deps_nm >= 0.0] = 0.0

        return n_nmG, eps_n, eps_m, f_n, f_m
                
    @timer('redist')
    def redistribute(self, in_wGG, out_x=None):
        """Redistribute array.
        
        Switch between two kinds of parallel distributions:
            
        1) parallel over G-vectors (second dimension of in_wGG)
        2) parallel over frequency (first dimension of in_wGG)

        Returns new array using the memory in the 1-d array out_x.
        """
        
        comm = self.blockcomm
        
        if comm.size == 1:
            return in_wGG
            
        nw = len(self.omega_w)
        nG = in_wGG.shape[2]
        mynw = (nw + comm.size - 1) // comm.size
        mynG = (nG + comm.size - 1) // comm.size
        
        bg1 = BlacsGrid(comm, comm.size, 1)
        bg2 = BlacsGrid(comm, 1, comm.size)
        md1 = BlacsDescriptor(bg1, nw, nG**2, mynw, nG**2)
        md2 = BlacsDescriptor(bg2, nw, nG**2, nw, mynG * nG)
        
        if len(in_wGG) == nw:
            mdin = md2
            mdout = md1
        else:
            mdin = md1
            mdout = md2
            
        r = Redistributor(comm, mdin, mdout)
        
        outshape = (mdout.shape[0], mdout.shape[1] // nG, nG)
        if out_x is None:
            out_wGG = np.empty(outshape, complex)
        else:
            out_wGG = out_x[:np.product(outshape)].reshape(outshape)
            
        r.redistribute(in_wGG.reshape(mdin.shape),
                       out_wGG.reshape(mdout.shape))
        
        return out_wGG

    @timer('dist freq')
    def distribute_frequencies(self, chi0_wGG):
        """Distribute frequencies to all cores."""
        
        world = self.world
        comm = self.blockcomm
        
        if world.size == 1:
            return chi0_wGG
            
        nw = len(self.omega_w)
        nG = chi0_wGG.shape[2]
        mynw = (nw + world.size - 1) // world.size
        mynG = (nG + comm.size - 1) // comm.size
  
        wa = min(world.rank * mynw, nw)
        wb = min(wa + mynw, nw)

        if self.blockcomm.size == 1:
            return chi0_wGG[wa:wb].copy()

        if self.kncomm.rank == 0:
            bg1 = BlacsGrid(comm, 1, comm.size)
            in_wGG = chi0_wGG.reshape((nw, -1))
        else:
            bg1 = DryRunBlacsGrid(mpi.serial_comm, 1, 1)
            in_wGG = np.zeros((0, 0), complex)
        md1 = BlacsDescriptor(bg1, nw, nG**2, nw, mynG * nG)
        
        bg2 = BlacsGrid(world, world.size, 1)
        md2 = BlacsDescriptor(bg2, nw, nG**2, mynw, nG**2)
        
        r = Redistributor(world, md1, md2)
        shape = (wb - wa, nG, nG)
        out_wGG = np.empty(shape, complex)
        r.redistribute(in_wGG, out_wGG.reshape((wb - wa, nG**2)))
        
        return out_wGG

    def print_chi(self, pd):
        calc = self.calc
        gd = calc.wfs.gd

        if extra_parameters.get('df_dry_run'):
            from gpaw.mpi import DryRunCommunicator
            size = extra_parameters['df_dry_run']
            world = DryRunCommunicator(size)
        else:
            world = self.world

        q_c = pd.kd.bzk_kc[0]
        nw = len(self.omega_w)
        ecut = self.ecut * Hartree
        ns = calc.wfs.nspins
        nbands = self.nbands
        nk = calc.wfs.kd.nbzkpts
        nik = calc.wfs.kd.nibzkpts
        ngmax = pd.ngmax        
        eta = self.eta * Hartree
        wsize = world.size
        knsize = self.kncomm.size
        nocc = self.nocc1
        npocc = self.nocc2
        keep = self.keep_occupied_states
        chisize = nw * pd.ngmax**2 * 16. / 1024**2
        ngridpoints = gd.N_c[0] * gd.N_c[1] * gd.N_c[2]
        if self.keep_occupied_states:
            nstat = (ns * nk * npocc + world.size - 1) // world.size
        else:
            nstat = (ns * npocc + world.size - 1) // world.size
        occsize = nstat * ngridpoints * 16. / 1024**2
        bsize = self.blockcomm.size

        p = partial(print, file=self.fd)

        p('%s' % ctime())
        p('Called response.chi0.calculate with')
        p('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]))
        p('    Number of frequency points: %d' % nw)
        p('    Planewave cutoff: %f' % ecut)
        p('    Number of spins: %d' % ns)
        p('    Number of bands: %d' % nbands)
        p('    Number of kpoints: %d' % nk)
        p('    Number of irredicible kpoints: %d' % nik)
        p('    Number of planewaves: %d' % ngmax)
        p('    Broadening (eta): %f' % eta)
        p('    world.size: %d' % wsize)
        p('    kncomm.size: %d' % knsize)
        p('    blockcomm.size: %d' % bsize)
        p('    Number of completely occupied states: %d' % nocc)
        p('    Number of partially occupied states: %d' % npocc)
        p('    Keep occupied states: %s' % keep)
        p()
        p('    Memory estimate of potentially large arrays:')
        p('        chi0_wGG: %f M / cpu' % chisize)
        p('        Occupied states: %f M / cpu' % occsize)
        p('        Memory usage before allocation: %f M / cpu' % (maxrss() / 1024**2))
        p()

