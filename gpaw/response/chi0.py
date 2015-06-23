from __future__ import print_function, division

import sys
from time import ctime

import numpy as np
from ase.units import Hartree
from ase.utils import devnull
from ase.utils.timing import timer, Timer

import gpaw.mpi as mpi
from gpaw import extra_parameters
from gpaw.blacs import (BlacsGrid, BlacsDescriptor, Redistributor,
                        DryRunBlacsGrid)
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.pair import PairDensity
from gpaw.utilities.memory import maxrss
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.response.pair import PWSymmetryAnalyzer
from gpaw.response.integrators import (TetrahedronIntegrator, HilbertTransform)

from functools import partial


class ArrayDescriptor:
    """Describes a single dimensional array."""
    def __init__(self, data_x):
        self.data_x = data_x
        self._data_len = len(data_x)

    def __len__(self):
        return self._data_len

    def get_data(self):
        return self.data_x

    def get_closest_index(self, scalar):
        """Get closest index.
        
        Get closest index approximating scalar from below."""
        diff_x = scalar - self.data_x
        return np.argmin(diff_x)

    def get_index_range(self, lim1, lim2):
        """Get index range. """
        i_x = np.nonzero(np.logical_and(lim1 <= self.data_x,
                                        self.data_x <= lim2))[0]
        return i_x


class FrequencyDescriptor(ArrayDescriptor):
    def __init__(self, domega0, omega2, omegamax):
        beta = (2**0.5 - 1) * domega0 / omega2
        wmax = int(omegamax / (domega0 + beta * omegamax)) + 2
        w = np.arange(wmax)
        omega_w = w * domega0 / (1 - beta * w)

        ArrayDescriptor.__init__(self, omega_w)

        self.domega0 = domega0
        self.omega2 = omega2
        self.omegamax = omegamax
        self.omegamin = 0

        self.beta = beta
        self.wmax = wmax
        self.omega_w = omega_w
        self.nw = len(omega_w)

    def get_closest_index(self, omega):
        beta = self.beta
        o_m = omega
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        return w_m

    def get_index_range(self, omega1, omega2):
        if omega1 < 0 or omega2 < 0:
            return np.array((), int)

        w1 = self.get_closest_index(omega1)
        w2 = self.get_closest_index(omega2)
        o1 = self.omega_w[w1]
        o2 = self.omega_w[w2]
        if o1 < omega1:
            w1 += 1
        if o2 > omega2:
            w2 -= 1
        
        return np.arange(w1, w2 + 1)


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

        self.timer = timer or Timer()

        self.pair = PairDensity(calc, ecut, ftol, threshold,
                                real_space_derivatives, world, txt,
                                self.timer,
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
            self.wd = FrequencyDescriptor(self.domega0, self.omega2,
                                          self.omegamax)
            self.omega_w = self.wd.get_data()
        else:
            self.omega_w = np.asarray(frequencies) / Hartree
            assert not hilbert
        self.hilbert = hilbert
        self.timeordered = bool(timeordered)

        if self.eta == 0.0:
            assert not hilbert
            assert not timeordered
            assert not self.omega_w.real.any()

        self.nocc1 = self.pair.nocc1  # number of completely filled bands
        self.nocc2 = self.pair.nocc2  # number of non-empty bands

        self.Q_aGii = None

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
        optical_limit = np.allclose(q_c, 0.0)
        
        pd = self.get_PWDescriptor(q_c)

        self.print_chi(pd)

        if extra_parameters.get('df_dry_run'):
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        nG = pd.ngmax + 2 * optical_limit
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

        if optical_limit:
            chi0_wxvG = np.zeros((len(self.omega_w), 2, 3, nG), complex)
            chi0_wvv = np.zeros((len(self.omega_w), 3, 3), complex)
            self.plasmafreq_vv = np.zeros((3, 3), complex)
        else:
            chi0_wxvG = None
            chi0_wvv = None
            self.plasmafreq_vv = None

        # Do all empty bands:
        m1 = self.nocc1
        m2 = self.nbands
        
        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self._calculate(pd,
                                                            chi0_wGG,
                                                            chi0_wxvG,
                                                            chi0_wvv,
                                                            m1, m2, spins)
        
        return pd, chi0_wGG, chi0_wxvG, chi0_wvv

    @timer('Calculate CHI_0')
    def _calculate(self, pd, chi0_wGG, chi0_wxvG, chi0_wvv, m1, m2, spins):
        bzk_kv, PWSA = self.get_kpoints(pd)

        # Initialize integrator
        integrator = TetrahedronIntegrator(comm=self.kncomm,
                                           timer=self.timer)
        td = integrator.tesselate(bzk_kv)

        # Integrate interband response
        kd2 = self.calc.wfs.kd
        mat_kwargs = {'kd': kd2, 'pd': pd, 'n1': 0,
                      'n2': self.nocc2, 'm1': m1,
                      'm2': m2, 'symmetry': PWSA}
        eig_kwargs = {'kd': kd2, 'm1': m1, 'm2': m2, 'n1': 0,
                      'n2': self.nocc2, 'pd': pd}
        domain = (td, spins)
        integrator.integrate('response_function', domain,
                             (self.get_matrix_element,
                              self.get_eigenvalues),
                             self.wd,
                             kwargs=(mat_kwargs, eig_kwargs),
                             out_wxx=chi0_wGG)

        with self.timer('Hilbert transform'):
            omega_w = self.wd.get_data()
            ht = HilbertTransform(np.array(omega_w), 0.0001 / Hartree)
            ht(chi0_wGG)

        if chi0_wxvG is not None:
            if self.nocc1 != self.nocc2:
                # Determine plasma frequency
                mat_kwargs = {'kd': kd2, 'symmetry': PWSA,
                              'n1': self.nocc1, 'n2': self.nocc2,
                              'pd': pd}

                eig_kwargs = {'kd': kd2, 'n1': self.nocc1,
                              'n2': self.nocc2, 'pd': pd}
                domain = (td, spins)
                fermi_level = self.pair.fermi_level
                plasmafreq_wvv = np.zeros((1, 3, 3), complex)
                integrator.integrate('response_function', domain,
                                     (self.get_intraband_response,
                                      self.get_intraband_eigenvalue),
                                     ArrayDescriptor([fermi_level]),
                                     kwargs=(mat_kwargs, eig_kwargs),
                                     out_wxx=plasmafreq_wvv)
                self.plasmafreq_vv = plasmafreq_wvv[0]
                chi0_wGG[:, 0:3, 0:3] += (self.plasmafreq_vv[np.newaxis] /
                                          (self.omega_w[:, np.newaxis,
                                                        np.newaxis]**2 +
                                           1e-10))

        # Symmetrize chi the results
        chi0_wGG *= (2 * PWSA.how_many_symmetries() /
                     (len(spins) * (2 * np.pi)**3))  # Remember prefactor

        tmpchi0_wGG = self.redistribute(chi0_wGG)
        PWSA.symmetrize_wxx(tmpchi0_wGG)
        self.redistribute(tmpchi0_wGG, chi0_wGG)

        chi0_wxvG[:, 0] = np.transpose(chi0_wGG[:, :, 0:3], (0, 2, 1))
        chi0_wxvG[:, 1] = chi0_wGG[:, 0:3]
        chi0_wxvG = chi0_wxvG[..., 2:]
        chi0_wvv = chi0_wGG[:, 0:3, 0:3]
        chi0_wGG = chi0_wGG[:, 2:, 2:]

        return pd, chi0_wGG, chi0_wxvG, chi0_wvv

    def get_PWDescriptor(self, q_c):
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, self.calc.wfs.gd,
                          complex, qd)
        return pd

    @timer('Get kpoints')
    def get_kpoints(self, pd):
        # Use symmetries
        PWSA = PWSymmetryAnalyzer
        PWSA = PWSA(self.calc.wfs.kd, pd,
                    timer=self.timer, txt=self.fd)
        bzk_kc = PWSA.get_reduced_kd().bzk_kc
        bzk_kv = np.dot(bzk_kc, pd.gd.icell_cv) * 2 * np.pi

        return bzk_kv, PWSA

    @timer('Get matrix element')
    def get_matrix_element(self, k_v, s, n1=None, n2=None,
                           m1=None, m2=None,
                           pd=None, kd=None,
                           symmetry=None):
        """A function that returns pair-densities.

        A pair density is defined as::

         <snk| e^(-i (q + G) r) |smk+q>,

        where s is spin, n and m are band indices, k is
        the kpoint and q is the momentum transfer."""
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        nG = pd.ngmax
        
        if self.Q_aGii is None:
            self.Q_aGii = self.pair.initialize_paw_corrections(pd)

        kptpair = self.pair.get_kpoint_pair(pd, s, k_c, n1, n2,
                                            m1, m2)
        m_m = np.arange(m1, m2)
        n_n = np.arange(n1, n2)
        n_nmG = self.pair.get_pair_density(pd, kptpair, n_n, m_m,
                                           Q_aGii=self.Q_aGii)
        df_nm = kptpair.get_occupation_differences(n_n, m_m)
        df_nm[df_nm <= 1e-20] = 0.0
        n_nmG *= df_nm[..., np.newaxis]
        return n_nmG.reshape((-1, nG + 2))

    @timer('Get eigenvalues')
    def get_eigenvalues(self, k_v, s, n1=None, n2=None,
                        m1=None, m2=None,
                        kd=None, pd=None, wfs=None):
        """A function that can return the eigenvalues.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""

        if wfs is None:
            wfs = self.calc.wfs

        kd = wfs.kd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = kd.where_is_q(k_c, kd.bzk_kc)
        ik = kd.bz2ibz_k[K1]
        kpt1 = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]
        K2 = kd.where_is_q(k_c + pd.kd.bzk_kc[0], kd.bzk_kc)
        ik = kd.bz2ibz_k[K2]
        kpt2 = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]
        deps_nm = (kpt2.eps_n[m1:m2][np.newaxis]
                   - kpt1.eps_n[n1:n2][:, np.newaxis])

        return deps_nm.reshape(-1)

    def get_intraband_response(self, k_v, s, n1=None, n2=None,
                               kd=None, symmetry=None, pd=None):
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        kpt1 = self.pair.get_k_point(s, k_c, n1, n2)
        n_n = range(n1, n2)

        vel_nv = self.pair.intraband_pair_density(kpt1, n_n)
        return vel_nv

    def get_intraband_eigenvalue(self, k_v, s,
                                 n1=None, n2=None, kd=None, pd=None):
        """A function that can return the eigenvalues.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        wfs = self.calc.wfs
        kd = wfs.kd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = kd.where_is_q(k_c, kd.bzk_kc)
        ik = kd.bz2ibz_k[K1]
        kpt1 = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]

        return kpt1.eps_n[n1:n2]

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

