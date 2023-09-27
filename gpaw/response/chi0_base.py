from abc import ABC, abstractmethod

import numpy as np

from ase.units import Ha

from gpaw.bztools import convex_hull_volume
from gpaw.response import timer
from gpaw.response.frequencies import NonLinearFrequencyDescriptor
from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.response.integrators import (
    Integrand, PointIntegrator, TetrahedronIntegrator)
from gpaw.response.symmetry import PWSymmetryAnalyzer


class Chi0Integrand(Integrand):
    def __init__(self, chi0calc, optical, qpd, analyzer, m1, m2):
        self._chi0calc = chi0calc

        # In a normal response calculation, we include transitions from all
        # completely and partially unoccupied bands to range(m1, m2)
        self.gs = chi0calc.gs
        self.n1 = 0
        self.n2 = self.gs.nocc2
        assert m1 <= m2
        self.m1 = m1
        self.m2 = m2

        self.context = chi0calc.context
        self.kptpair_factory = chi0calc.kptpair_factory

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
        kptfinder = self.gs.kpoints.kptfinder
        K1 = kptfinder.find(k_c)
        K2 = kptfinder.find(k_c + qpd.q_c)

        ik1 = kd.bz2ibz_k[K1]
        ik2 = kd.bz2ibz_k[K2]
        kpt1 = gs.kpt_qs[ik1][s]
        assert kd.comm.size == 1
        kpt2 = gs.kpt_qs[ik2][s]
        deps_nm = np.subtract(kpt1.eps_n[self.n1:self.n2][:, np.newaxis],
                              kpt2.eps_n[self.m1:self.m2])
        return deps_nm.reshape(-1)


class Chi0ComponentCalculator:
    """Base class for the Chi0XXXCalculator suite."""

    def __init__(self, kptpair_factory,
                 context=None,
                 disable_point_group=False,
                 disable_time_reversal=False,
                 integrationmode=None):
        """Set up attributes common to all chi0 related calculators."""
        self.kptpair_factory = kptpair_factory
        self.gs = kptpair_factory.gs

        if context is None:
            context = kptpair_factory.context
        assert kptpair_factory.context.comm is context.comm
        self.context = context

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal

        # Set up integrator
        self.integrationmode = integrationmode
        self.integrator = self.construct_integrator()

    @property
    def nblocks(self):
        return self.kptpair_factory.nblocks

    @property
    def pbc(self):
        return self.gs.pbc

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
        kpoints, analyzer = self.get_kpoints(
            qpd, integrationmode=self.integrationmode)
        bzk_kv = kpoints.bzk_kv
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
            self.gs.kpoints, qpd, self.context,
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

        from gpaw.response.kpoints import ResponseKPointGrid
        kpoints = ResponseKPointGrid(self.gs.kd, qpd.gd.icell_cv, bzk_kc)
        # Two illogical things here:
        #  * Analyzer is for original kpoints, not those we just reduced
        #  * The kpoints object has another bzk_kc array than self.gs.kd.
        #    We could make a new kd, but I am not sure about ramifications.
        return kpoints, analyzer

    def get_gs_info_string(self, tab=''):
        gs = self.gs
        gd = gs.gd

        ns = gs.nspins
        nk = gs.kd.nbzkpts
        nik = gs.kd.nibzkpts

        nocc = self.gs.nocc1
        npocc = self.gs.nocc2
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


class Chi0ComponentPWCalculator(Chi0ComponentCalculator, ABC):
    """Base class for Chi0XXXCalculators, which utilize a plane-wave basis."""

    def __init__(self, kptpair_factory,
                 *,
                 wd,
                 hilbert=True,
                 nbands=None,
                 timeordered=False,
                 ecut=None,
                 eta=0.2,
                 **kwargs):
        """Set up attributes to calculate the chi0 body and optical extensions.
        """
        super().__init__(kptpair_factory, **kwargs)

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

    @property
    def pair_calc(self):
        return self.kptpair_factory.pair_calculator()

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

    @abstractmethod
    def construct_hermitian_task(self):
        """Integral task for the hermitian part of chi0."""

    def construct_hilbert_task(self):
        if isinstance(self.integrator, PointIntegrator):
            return self.construct_point_hilbert_task()
        else:
            assert isinstance(self.integrator, TetrahedronIntegrator)
            return self.construct_tetra_hilbert_task()

    @abstractmethod
    def construct_point_hilbert_task(self):
        """Integral task for point integrating the spectral part of chi0."""

    @abstractmethod
    def construct_tetra_hilbert_task(self):
        """Integral task for tetrahedron integration of the spectral part."""

    @abstractmethod
    def construct_literal_task(self):
        """Integral task for a literal evaluation of chi0."""

    def get_pw_descriptor(self, q_c):
        return SingleQPWDescriptor.from_q(q_c, self.ecut, self.gs.gd)

    def get_band_transitions(self):
        return self.gs.nocc1, self.nbands  # m1, m2

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