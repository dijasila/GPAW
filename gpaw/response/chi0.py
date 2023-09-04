from __future__ import annotations

import warnings
from time import ctime
from typing import Union

import numpy as np
from ase.units import Ha

import gpaw
import gpaw.mpi as mpi
from gpaw.bztools import convex_hull_volume
from gpaw.response.chi0_data import (Chi0Data, Chi0BodyData,
                                     Chi0OpticalExtensionData)
from gpaw.response.frequencies import (FrequencyDescriptor,
                                       NonLinearFrequencyDescriptor)
from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.response.hilbert import HilbertTransform
from gpaw.response.integrators import (
    Integrand, PointIntegrator, TetrahedronIntegrator)
from gpaw.response import timer
from gpaw.response.pair import KPointPairFactory
from gpaw.response.pw_parallelization import PlaneWaveBlockDistributor
from gpaw.response.symmetry import PWSymmetryAnalyzer
from gpaw.typing import Array1D
from gpaw.utilities.memory import maxrss
from gpaw.response.integrators import (
    HermitianOpticalLimit, HilbertOpticalLimit, OpticalLimit,
    HilbertOpticalLimitTetrahedron,
    Hermitian, Hilbert, HilbertTetrahedron, GenericUpdate)


def find_maximum_frequency(kpt_u, context, nbands=0):
    """Determine the maximum electron-hole pair transition energy."""
    epsmin = 10000.0
    epsmax = -10000.0
    for kpt in kpt_u:
        epsmin = min(epsmin, kpt.eps_n[0])
        epsmax = max(epsmax, kpt.eps_n[nbands - 1])

    context.print('Minimum eigenvalue: %10.3f eV' % (epsmin * Ha),
                  flush=False)
    context.print('Maximum eigenvalue: %10.3f eV' % (epsmax * Ha))

    return epsmax - epsmin


class Chi0Integrand(Integrand):
    def __init__(self, chi0calc, optical, qpd, analyzer, m1, m2):
        self._chi0calc = chi0calc

        # In a normal response calculation, we include transitions from all
        # completely and partially unoccupied bands to range(m1, m2)
        self.n1 = 0
        self.n2 = chi0calc.nocc2
        assert m1 <= m2
        self.m1 = m1
        self.m2 = m2

        self.context = chi0calc.context
        self.kptpair_factory = chi0calc.kptpair_factory
        self.gs = chi0calc.gs

        self.qpd = qpd
        self.analyzer = analyzer
        self.integrationmode = chi0calc.integrationmode
        self.optical = optical

    @timer('Get matrix element')
    def matrix_element(self, k_v, s):
        """Return pair density matrix element for integration.

        A pair density is defined as::

         <snk| e^(-i (q + G) r) |s'mk+q>,

        where s and s' are spins, n and m are band indices, k is
        the kpoint and q is the momentum transfer. For dielectric
        response s'=s, for the transverse magnetic response
        s' is flipped with respect to s.

        Parameters
        ----------
        k_v : ndarray
            Kpoint coordinate in cartesian coordinates.
        s : int
            Spin index.

        If self.optical, then return optical pair densities, that is, the
        head and wings matrix elements indexed by:
        # P = (x, y, v, G1, G2, ...).

        Return
        ------
        n_nmG : ndarray
            Pair densities.
        """

        if self.optical:
            target_method = self._chi0calc.pair_calc.get_optical_pair_density
            out_ngmax = self.qpd.ngmax + 2
        else:
            target_method = self._chi0calc.pair_calc.get_pair_density
            out_ngmax = self.qpd.ngmax

        return self._get_any_matrix_element(
            k_v, s, block=not self.optical,
            target_method=target_method,
        ).reshape(-1, out_ngmax)

    def _get_any_matrix_element(self, k_v, s, block, target_method):
        qpd = self.qpd

        k_c = np.dot(qpd.gd.cell_cv, k_v) / (2 * np.pi)

        weight = np.sqrt(self.analyzer.get_kpoint_weight(k_c) /
                         self.analyzer.how_many_symmetries())

        # Here we're again setting pawcorr willy-nilly
        if self._chi0calc.pawcorr is None:
            pairden_paw_corr = self.gs.pair_density_paw_corrections
            self._chi0calc.pawcorr = pairden_paw_corr(qpd)

        kptpair = self.kptpair_factory.get_kpoint_pair(
            qpd, s, k_c, self.n1, self.n2,
            self.m1, self.m2, block=block)

        m_m = np.arange(self.m1, self.m2)
        n_n = np.arange(self.n1, self.n2)
        n_nmG = target_method(qpd, kptpair, n_n, m_m,
                              pawcorr=self._chi0calc.pawcorr,
                              block=block)

        if self.integrationmode is None:
            n_nmG *= weight

        df_nm = kptpair.get_occupation_differences(n_n, m_m)
        df_nm[df_nm <= 1e-20] = 0.0
        n_nmG *= df_nm[..., np.newaxis]**0.5

        return n_nmG

    @timer('Get eigenvalues')
    def eigenvalues(self, k_v, s):
        """A function that can return the eigenvalues.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""

        qpd = self.qpd
        gs = self.gs
        kd = gs.kd

        k_c = np.dot(qpd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = self.kptpair_factory.find_kpoint(k_c)
        K2 = self.kptpair_factory.find_kpoint(k_c + qpd.q_c)

        ik1 = kd.bz2ibz_k[K1]
        ik2 = kd.bz2ibz_k[K2]
        kpt1 = gs.kpt_qs[ik1][s]
        assert kd.comm.size == 1
        kpt2 = gs.kpt_qs[ik2][s]
        deps_nm = np.subtract(kpt1.eps_n[self.n1:self.n2][:, np.newaxis],
                              kpt2.eps_n[self.m1:self.m2])
        return deps_nm.reshape(-1)


class Chi0Calculator:
    def __init__(self, *args,
                 eshift=0.0,
                 intraband=True,
                 rate=0.0,
                 **kwargs):
        self.chi0_body_calc = Chi0BodyCalculator(
            *args, eshift=eshift, **kwargs)
        self.chi0_opt_ext_calc = Chi0OpticalExtensionCalculator(
            *args, intraband=intraband, rate=rate, **kwargs)

        # Attributes groped by other classes...
        # Oh the horror, there are many of these...
        # Remove these in the future XXX
        self.gs = self.chi0_body_calc.gs
        self.context = self.chi0_body_calc.context
        self.wd = self.chi0_body_calc.wd
        self.nbands = self.chi0_body_calc.nbands
        self.nocc1 = self.chi0_body_calc.nocc1
        self.nocc2 = self.chi0_body_calc.nocc2
        self.ecut = self.chi0_body_calc.ecut
        self.kptpair_factory = self.chi0_body_calc.kptpair_factory
        self.integrator = self.chi0_body_calc.integrator

    @property
    def nblocks(self):
        return self.kptpair_factory.nblocks

    @property
    def pair_calc(self):
        return self.kptpair_factory.pair_calculator()

    def base_ini(self, kptpair_factory,
                 context=None,
                 disable_point_group=False,
                 disable_time_reversal=False,
                 integrationmode=None):
        """Set up attributes common to all response function calculators."""

        if context is None:
            context = kptpair_factory.context

        # TODO: More refactoring to avoid non-orthogonal inputs.
        assert kptpair_factory.context.comm is context.comm
        self.context = context

        self.kptpair_factory = kptpair_factory
        self.gs = kptpair_factory.gs

        # Number of completely filled bands and number of non-empty bands.
        self.nocc1, self.nocc2 = self.gs.count_occupied_bands()

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal

        # Set up integrator
        self.integrationmode = integrationmode
        self.integrator = self.construct_integrator()

    def tmp_init(self, wd, *args,
                 hilbert=True,
                 nbands=None,
                 timeordered=False,
                 ecut=None,
                 eta=0.2,
                 **kwargs):
        """Set up attributes to calculate the chi0 body and optical extensions.
        """
        self.base_ini(*args, **kwargs)

        if ecut is None:
            ecut = 50.0
        ecut /= Ha
        self.ecut = ecut
        self.nbands = nbands or self.gs.bd.nbands

        self.wd = wd
        self.context.print(self.wd, flush=False)

        self.eta = eta / Ha
        self.hilbert = hilbert
        self.task = self.construct_integral_task()

        self.timeordered = bool(timeordered)
        if self.timeordered:
            assert self.hilbert  # Timeordered is only needed for G0W0

        self.pawcorr = None

        self.context.print('Nonperiodic BCs: ', (~self.pbc))
        if sum(self.pbc) == 1:
            raise ValueError('1-D not supported atm.')

    def construct_integral_task(self):
        if self.eta == 0:
            assert not self.hilbert
            # eta == 0 is used as a synonym for calculating the hermitian part
            # of the response function at a range of imaginary frequencies
            assert not self.wd.omega_w.real.any()
            return self.construct_hermitian_task()

        if self.hilbert:
            # The hilbert flag is used to calculate the reponse function via a
            # hilbert transform of its dissipative (spectral) part.
            assert isinstance(self.wd, NonLinearFrequencyDescriptor)
            return self.construct_hilbert_task()

        # Otherwise, we perform a literal evaluation of the response function
        # at the given frequencies with broadening eta
        return self.construct_literal_task()

    def construct_hilbert_task(self):
        if isinstance(self.integrator, PointIntegrator):
            return self.construct_point_hilbert_task()
        else:
            assert isinstance(self.integrator, TetrahedronIntegrator)
            return self.construct_tetra_hilbert_task()

    def construct_tetra_hilbert_task(self):
        return HilbertTetrahedron(self.integrator.blockcomm)

    @property
    def pbc(self):
        return self.gs.pbc

    def create_chi0(self, q_c):
        chi0_body = self.chi0_body_calc.create_chi0_body(q_c)
        chi0 = Chi0Data.from_chi0_body_data(chi0_body)
        return chi0

    def get_pw_descriptor(self, q_c):
        return SingleQPWDescriptor.from_q(q_c, self.ecut, self.gs.gd)

    def calculate(self, q_c, spin='all'):
        """Calculate chi0 (possibly with optical extensions).

        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector.
        spin : str or int
            If 'all' then include all spins.
            If 0 or 1, only include this specific spin.

        Returns
        -------
        chi0 : Chi0Data
            Data object containing the chi0 data arrays along with basis
            representation descriptors and blocks distribution
        """
        # Calculate body
        chi0_body = self.chi0_body_calc.calculate(q_c, spin=spin)
        qpd = chi0_body.qpd

        # Calculate optical extension
        if qpd.optical_limit:
            if not abs(self.chi0_body_calc.eshift) < 1e-8:
                raise NotImplementedError("No wings eshift available")
            chi0_opt_ext = self.chi0_opt_ext_calc.calculate(qpd=qpd, spin=spin)
        else:
            chi0_opt_ext = None

        self.context.print('\nFinished calculating chi0\n')

        return Chi0Data(chi0_body, chi0_opt_ext)

    def get_band_transitions(self):
        return self.nocc1, self.nbands  # m1, m2

    def get_spins(self, spin='all'):
        nspins = self.gs.nspins
        if spin == 'all':
            spins = range(nspins)
        else:
            assert spin in range(nspins)
            spins = [spin]

        return spins

    @timer('Calculate CHI_0')
    def update_chi0(self,
                    chi0: Chi0Data,
                    m1, m2, spins):
        """In-place calculation of chi0 (with optical extension).

        Parameters
        ----------
        chi0 : Chi0Data
            Data and representation object
        m1 : int
            Lower band cutoff for band summation
        m2 : int
            Upper band cutoff for band summation
        spins : list
            List of spin indices to include in the calculation

        Returns
        -------
        chi0 : Chi0Data
        """
        self.chi0_body_calc.update_chi0_body(chi0.body, m1, m2, spins)
        if chi0.optical_limit:
            if not abs(self.chi0_body_calc.eshift) < 1e-8:
                raise NotImplementedError("No wings eshift available")
            assert chi0.optical_extension is not None
            # Update the head and wings
            self.chi0_opt_ext_calc.update_chi0_optical_extension(
                chi0.optical_extension, m1, m2, spins)
        return chi0

    def construct_integrator(self):
        """Construct k-point integrator"""
        cls = self.get_integrator_cls()
        return cls(
            cell_cv=self.gs.gd.cell_cv,
            context=self.context,
            nblocks=self.nblocks)

    def get_integrator_cls(self):
        """Get the appointed k-point integrator class."""
        if self.integrationmode is None:
            self.context.print('Using integrator: PointIntegrator')
            cls = PointIntegrator
        elif self.integrationmode == 'tetrahedron integration':
            self.context.print('Using integrator: TetrahedronIntegrator')
            cls = TetrahedronIntegrator  # type: ignore
            if not all([self.disable_point_group,
                        self.disable_time_reversal]):
                self.check_high_symmetry_ibz_kpts()
        else:
            raise ValueError(f'Integration mode "{self.integrationmode}"'
                             ' not implemented.')
        return cls

    def check_high_symmetry_ibz_kpts(self):
        """Check that the ground state includes all corners of the IBZ."""
        ibz_vertices_kc = self.gs.get_ibz_vertices()
        # Here we mimic the k-point grid compatibility check of
        # gpaw.bztools.find_high_symmetry_monkhorst_pack()
        bzk_kc = self.gs.kd.bzk_kc
        for ibz_vertex_c in ibz_vertices_kc:
            # Relative coordinate difference to the k-point grid
            diff_kc = np.abs(bzk_kc - ibz_vertex_c)[:, self.gs.pbc].round(6)
            # The ibz vertex should exits in the BZ grid up to a reciprocal
            # lattice vector, meaning that the relative coordinate difference
            # is allowed to be an integer. Thus, at least one relative k-point
            # difference should vanish, modulo 1
            mod_diff_kc = np.mod(diff_kc, 1)
            nodiff_k = np.all(mod_diff_kc < 1e-5, axis=1)
            if not np.any(nodiff_k):
                raise ValueError(
                    'The ground state k-point grid does not include all '
                    'vertices of the IBZ. '
                    'Please use find_high_symmetry_monkhorst_pack() from '
                    'gpaw.bztools to generate your k-point grid.')

    def get_integration_domain(self, qpd, spins):
        """Get integrator domain and prefactor for the integral."""
        for spin in spins:
            assert spin in range(self.gs.nspins)
        # The integration domain is determined by the following function
        # that reduces the integration domain to the irreducible zone
        # of the little group of q.
        bzk_kv, analyzer = self.get_kpoints(
            qpd, integrationmode=self.integrationmode)
        domain = (bzk_kv, spins)

        if self.integrationmode == 'tetrahedron integration':
            # If there are non-periodic directions it is possible that the
            # integration domain is not compatible with the symmetry operations
            # which essentially means that too large domains will be
            # integrated. We normalize by vol(BZ) / vol(domain) to make
            # sure that to fix this.
            domainvol = convex_hull_volume(
                bzk_kv) * analyzer.how_many_symmetries()
            bzvol = (2 * np.pi)**3 / self.gs.volume
            factor = bzvol / domainvol
        else:
            factor = 1

        prefactor = (2 * factor * analyzer.how_many_symmetries() /
                     (self.gs.nspins * (2 * np.pi)**3))  # Remember prefactor

        if self.integrationmode is None:
            nbzkpts = self.gs.kd.nbzkpts
            prefactor *= len(bzk_kv) / nbzkpts

        return domain, analyzer, prefactor

    @timer('Get kpoints')
    def get_kpoints(self, qpd, integrationmode):
        """Get the integration domain."""
        analyzer = PWSymmetryAnalyzer(
            self.gs.kd, qpd, self.context,
            disable_point_group=self.disable_point_group,
            disable_time_reversal=self.disable_time_reversal)

        if integrationmode is None:
            K_gK = analyzer.group_kpoints()
            bzk_kc = np.array([self.gs.kd.bzk_kc[K_K[0]] for
                               K_K in K_gK])
        elif integrationmode == 'tetrahedron integration':
            bzk_kc = analyzer.get_reduced_kd(pbc_c=self.pbc).bzk_kc
            if (~self.pbc).any():
                bzk_kc = np.append(bzk_kc,
                                   bzk_kc + (~self.pbc).astype(int),
                                   axis=0)

        bzk_kv = np.dot(bzk_kc, qpd.gd.icell_cv) * 2 * np.pi
        return bzk_kv, analyzer

    def get_gs_info_string(self, tab=''):
        gs = self.gs
        gd = gs.gd

        ns = gs.nspins
        nk = gs.kd.nbzkpts
        nik = gs.kd.nibzkpts

        nocc = self.nocc1
        npocc = self.nocc2
        ngridpoints = gd.N_c[0] * gd.N_c[1] * gd.N_c[2]
        nstat = ns * npocc
        occsize = nstat * ngridpoints * 16. / 1024**2

        gs_list = [f'{tab}Ground state adapter containing:',
                   f'Number of spins: {ns}', f'Number of kpoints: {nk}',
                   f'Number of irreducible kpoints: {nik}',
                   f'Number of completely occupied states: {nocc}',
                   f'Number of partially occupied states: {npocc}',
                   f'Occupied states memory: {occsize} M / cpu']

        return f'\n{tab}'.join(gs_list)

    def get_response_info_string(self, qpd, tab=''):
        nw = len(self.wd)
        ecut = self.ecut * Ha
        nbands = self.nbands
        ngmax = qpd.ngmax
        eta = self.eta * Ha

        res_list = [f'{tab}Number of frequency points: {nw}',
                    f'Planewave cutoff: {ecut}',
                    f'Number of bands: {nbands}',
                    f'Number of planewaves: {ngmax}',
                    f'Broadening (eta): {eta}']

        return f'\n{tab}'.join(res_list)


class Chi0BodyCalculator(Chi0Calculator):

    def __init__(self, *args,
                 eshift=0.0,
                 **kwargs):
        self.eshift = eshift / Ha
        self.tmp_init(*args, **kwargs)

        metallic = self.nocc1 != self.nocc2
        if metallic:
            assert abs(self.eshift) < 1e-8, \
                'A rigid energy shift cannot be applied to the conduction '\
                'bands if there is no band gap'

    def create_chi0_body(self, q_c):
        qpd = self.get_pw_descriptor(q_c)
        return self._create_chi0_body(qpd)

    def _create_chi0_body(self, qpd):
        return Chi0BodyData(self.wd, qpd, self.get_blockdist())

    def get_blockdist(self):
        return PlaneWaveBlockDistributor(
            self.context.comm,
            self.integrator.blockcomm,
            self.integrator.kncomm)

    def calculate(self, q_c, spin='all') -> Chi0BodyData:
        """Calculate the chi0 body.

        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector.
        spin : str or int
            If 'all' then include all spins.
            If 0 or 1, only include this specific spin."""
        # Construct the output data structure
        qpd = self.get_pw_descriptor(q_c)
        self.print_info(qpd)
        chi0_body = self._create_chi0_body(qpd)

        # Integrate all transitions into partially filled and empty bands
        m1, m2 = self.get_band_transitions()
        spins = self.get_spins(spin)
        self.update_chi0_body(chi0_body, m1, m2, spins)

        return chi0_body

    def update_chi0_body(self,
                         chi0_body: Chi0BodyData,
                         m1, m2, spins):
        """In-place calculation of the body."""
        qpd = chi0_body.qpd

        # Reset PAW correction in case momentum has change
        pairden_paw_corr = self.gs.pair_density_paw_corrections
        self.pawcorr = pairden_paw_corr(chi0_body.qpd)

        self.context.print('Integrating chi0 body.')

        domain, analyzer, prefactor = self.get_integration_domain(qpd, spins)
        integrand = Chi0Integrand(self, qpd=qpd, analyzer=analyzer,
                                  optical=False, m1=m1, m2=m2)

        chi0_body.data_WgG[:] /= prefactor
        if self.hilbert:
            # Allocate a temporary array for the spectral function
            out_WgG = chi0_body.zeros()
        else:
            # Use the preallocated array for direct updates
            out_WgG = chi0_body.data_WgG
        self.integrator.integrate(domain=domain,  # Integration domain
                                  integrand=integrand,
                                  task=self.task,
                                  wd=self.wd,  # Frequency Descriptor
                                  out_wxx=out_WgG)  # Output array

        if self.hilbert:
            # The integrator only returns the spectral function and a Hilbert
            # transform is performed to return the real part of the density
            # response function.
            with self.context.timer('Hilbert transform'):
                # Make Hilbert transform
                ht = HilbertTransform(np.array(self.wd.omega_w), self.eta,
                                      timeordered=self.timeordered)
                ht(out_WgG)
            # Update the actual chi0 array
            chi0_body.data_WgG[:] += out_WgG
        chi0_body.data_WgG[:] *= prefactor

        tmp_chi0_wGG = chi0_body.copy_array_with_distribution('wGG')
        analyzer.symmetrize_wGG(tmp_chi0_wGG)
        chi0_body.data_WgG[:] = chi0_body.blockdist.distribute_as(
            tmp_chi0_wGG, chi0_body.nw, 'WgG')

    def construct_hermitian_task(self):
        return Hermitian(self.integrator.blockcomm, eshift=self.eshift)

    def construct_point_hilbert_task(self):
        return Hilbert(self.integrator.blockcomm, eshift=self.eshift)

    def construct_literal_task(self):
        return GenericUpdate(
            self.eta, self.integrator.blockcomm, eshift=self.eshift)

    def print_info(self, qpd):

        if gpaw.dry_run:
            from gpaw.mpi import SerialCommunicator
            size = gpaw.dry_run
            comm = SerialCommunicator()
            comm.size = size
        else:
            comm = self.context.comm

        q_c = qpd.q_c
        nw = len(self.wd)
        csize = comm.size
        knsize = self.integrator.kncomm.size
        bsize = self.integrator.blockcomm.size
        chisize = nw * qpd.ngmax**2 * 16. / 1024**2 / bsize

        isl = ['', f'{ctime()}',
               'Calculating chi0 body with:',
               self.get_gs_info_string(tab='    '), '',
               '    Linear response parametrization:',
               f'    q_c: [{q_c[0]}, {q_c[1]}, {q_c[2]}]',
               self.get_response_info_string(qpd, tab='    '),
               f'    comm.size: {csize}',
               f'    kncomm.size: {knsize}',
               f'    blockcomm.size: {bsize}']
        if bsize > nw:
            isl.append(
                'WARNING! Your nblocks is larger than number of frequency'
                ' points. Errors might occur, if your submodule does'
                ' not know how to handle this.')
        isl.extend(['',
                    '    Memory estimate of potentially large arrays:',
                    f'        chi0_wGG: {chisize} M / cpu',
                    '        Memory usage before allocation: '
                    f'{(maxrss() / 1024**2)} M / cpu'])
        self.context.print('\n'.join(isl))


class Chi0OpticalExtensionCalculator(Chi0Calculator):

    def __init__(self, *args,
                 intraband=True,
                 rate=0.0,
                 **kwargs):
        self.tmp_init(*args, **kwargs)

        # In the optical limit of metals, one must add the Drude dielectric
        # response from the free-space plasma frequency of the intraband
        # transitions to the head of the chi0 wings. This is handled by a
        # separate calculator, provided that intraband is set to True.
        metallic = self.nocc1 != self.nocc2
        if metallic and intraband:
            from gpaw.response.drude import Chi0DrudeCalculator
            if rate == 'eta':
                rate = self.eta * Ha  # external units
            self.rate = rate
            self.drude_calc = Chi0DrudeCalculator(
                self.kptpair_factory,
                disable_point_group=self.disable_point_group,
                disable_time_reversal=self.disable_time_reversal,
                integrationmode=self.integrationmode)
        else:
            self.drude_calc = None
            self.rate = None

    @property
    def nblocks(self):
        # The optical extensions are not distributed in memory
        # NB: There can be a mismatch with
        # self.kptpair_factory.nblocks, which seems dangerous XXX
        return 1

    def calculate(self,
                  qpd: SingleQPWDescriptor | None = None,
                  spin='all'):
        """Calculate the chi0 head and wings.

        Paramters
        ---------
        spin : str or int
            If 'all' then include all spins.
            If 0 or 1, only include this specific spin.
        """
        # Create data object
        if qpd is None:
            qpd = self.get_pw_descriptor(q_c=[0., 0., 0.])
        chi0_opt_ext = Chi0OpticalExtensionData(self.wd, qpd)

        self.print_info(qpd)

        # Define band and spin transitions
        m1, m2 = self.get_band_transitions()
        spins = self.get_spins(spin)

        # Perform the actual integration
        self.update_chi0_optical_extension(chi0_opt_ext, m1, m2, spins)

        if self.drude_calc is not None:
            # Add intraband contribution
            chi0_drude = self.drude_calc.calculate(self.wd, self.rate, spin)
            chi0_opt_ext.head_Wvv[:] += chi0_drude.chi_Zvv

        return chi0_opt_ext

    def update_chi0_optical_extension(
            self,
            chi0_optical_extension: Chi0OpticalExtensionData,
            m1, m2, spins):
        """In-place calculation of the chi0 head and wings.

        Parameters
        ----------
        m1 : int
            Lower band cutoff for band summation
        m2 : int
            Upper band cutoff for band summation
        spins : list
            List of spin indices to include in the calculation
        """
        self.context.print('Integrating chi0 head and wings.')
        chi0_opt_ext = chi0_optical_extension
        qpd = chi0_opt_ext.qpd

        domain, analyzer, prefactor = self.get_integration_domain(qpd, spins)
        integrand = Chi0Integrand(self, qpd=qpd, analyzer=analyzer,
                                  optical=True, m1=m1, m2=m2)

        # We integrate the head and wings together, using the combined index P
        # index v = (x, y, z)
        # index G = (G0, G1, G2, ...)
        # index P = (x, y, z, G1, G2, ...)
        WxvP_shape = list(chi0_opt_ext.WxvG_shape)
        WxvP_shape[-1] += 2
        tmp_chi0_WxvP = np.zeros(WxvP_shape, complex)
        self.integrator.integrate(domain=domain,  # Integration domain
                                  integrand=integrand,
                                  task=self.task,
                                  wd=self.wd,  # Frequency Descriptor
                                  out_wxx=tmp_chi0_WxvP)  # Output array
        if self.hilbert:
            with self.context.timer('Hilbert transform'):
                ht = HilbertTransform(np.array(self.wd.omega_w), self.eta,
                                      timeordered=self.timeordered)
                ht(tmp_chi0_WxvP)
        tmp_chi0_WxvP *= prefactor

        # Fill in wings part of the data, but leave out the head part (G0)
        chi0_opt_ext.wings_WxvG[..., 1:] += tmp_chi0_WxvP[..., 3:]
        analyzer.symmetrize_wxvG(chi0_opt_ext.wings_WxvG)
        # Fill in the head
        chi0_opt_ext.head_Wvv[:] += tmp_chi0_WxvP[:, 0, :3, :3]
        analyzer.symmetrize_wvv(chi0_opt_ext.head_Wvv)

    def construct_hermitian_task(self):
        return HermitianOpticalLimit()

    def construct_point_hilbert_task(self):
        return HilbertOpticalLimit()

    def construct_tetra_hilbert_task(self):
        return HilbertOpticalLimitTetrahedron()

    def construct_literal_task(self):
        return OpticalLimit(eta=self.eta)

    def print_info(self, qpd):
        """Print information about optical extension calculation."""
        isl = ['',
               f'{ctime()}',
               'Calculating chi0 optical extensions with:',
               self.get_gs_info_string(tab='    '),
               '',
               '    Linear response parametrization:',
               self.get_response_info_string(qpd, tab='    ')]
        self.context.print('\n'.join(isl))


class Chi0(Chi0Calculator):
    """Class for calculating non-interacting response functions.
    Tries to be backwards compatible, for now. """

    def __init__(self,
                 calc,
                 *,
                 frequencies: Union[dict, Array1D] = None,
                 ecut=50,
                 world=mpi.world, txt='-', timer=None,
                 nblocks=1,
                 nbands=None,
                 domega0=None,  # deprecated
                 omega2=None,  # deprecated
                 omegamax=None,  # deprecated
                 **kwargs):
        """Construct Chi0 object.

        Parameters
        ----------
        calc : str
            The groundstate calculation file that the linear response
            calculation is based on.
        frequencies :
            Input parameters for frequency_grid.
            Can be array of frequencies to evaluate the response function at
            or dictionary of paramaters for build-in nonlinear grid
            (see :ref:`frequency grid`).
        ecut : float
            Energy cutoff.
        hilbert : bool
            Switch for hilbert transform. If True, the full density response
            is determined from a hilbert transform of its spectral function.
            This is typically much faster, but does not work for imaginary
            frequencies.
        nbands : int
            Maximum band index to include.
        timeordered : bool
            Switch for calculating the time ordered density response function.
            In this case the hilbert transform cannot be used.
        eta : float
            Artificial broadening of spectra.
        intraband : bool
            Switch for including the intraband contribution to the density
            response function.
        world : MPI comm instance
            MPI communicator.
        txt : str
            Output file.
        timer : gpaw.utilities.timing.timer instance
        nblocks : int
            Divide the response function into nblocks. Useful when the response
            function is large.
        disable_point_group : bool
            Do not use the point group symmetry operators.
        disable_time_reversal : bool
            Do not use time reversal symmetry.
        integrationmode : str
            Integrator for the kpoint integration.
            If == 'tetrahedron integration' then the kpoint integral is
            performed using the linear tetrahedron method.
        eshift : float
            Shift unoccupied bands
        rate : float,str
            Phenomenological scattering rate to use in optical limit Drude term
            (in eV). If rate='eta', then use input artificial broadening eta as
            rate. Note, for consistency with the formalism the rate is
            implemented as omegap^2 / (omega + 1j * rate)^2 which differ from
            some literature by a factor of 2.


        Attributes
        ----------
        kptpair_factory : gpaw.response.pair.KPointPairFactory instance
            Class for calculating matrix elements of pairs of wavefunctions.

        """
        from gpaw.response.pair import get_gs_and_context
        gs, context = get_gs_and_context(calc, txt, world, timer)
        nbands = nbands or gs.bd.nbands

        wd = new_frequency_descriptor(
            gs, context, nbands, frequencies,
            domega0=domega0,
            omega2=omega2, omegamax=omegamax)

        kptpair_factory = KPointPairFactory(gs, context, nblocks=nblocks)

        super().__init__(wd=wd, kptpair_factory=kptpair_factory,
                         nbands=nbands, ecut=ecut, **kwargs)


def new_frequency_descriptor(gs, context, nbands, frequencies=None, *,
                             domega0=None, omega2=None, omegamax=None):
    if domega0 is not None or omega2 is not None or omegamax is not None:
        assert frequencies is None
        frequencies = {'type': 'nonlinear',
                       'domega0': domega0,
                       'omega2': omega2,
                       'omegamax': omegamax}
        warnings.warn(f'Please use frequencies={frequencies}')

    elif frequencies is None:
        frequencies = {'type': 'nonlinear'}

    if (isinstance(frequencies, dict) and
        frequencies.get('omegamax') is None):
        omegamax = find_maximum_frequency(gs.kpt_u, context,
                                          nbands=nbands)
        frequencies['omegamax'] = omegamax * Ha

    wd = FrequencyDescriptor.from_array_or_dict(frequencies)
    return wd
