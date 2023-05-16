from __future__ import annotations

import warnings
from functools import partial
from time import ctime
from typing import Union

import numpy as np
from ase.units import Ha

import gpaw
import gpaw.mpi as mpi
from gpaw.bztools import convex_hull_volume
from gpaw.response.chi0_data import Chi0Data, Chi0DrudeData
from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.response.frequencies import (FrequencyDescriptor,
                                       FrequencyGridDescriptor,
                                       NonLinearFrequencyDescriptor)
from gpaw.response.hilbert import HilbertTransform
from gpaw.response.integrators import (Integrator, PointIntegrator,
                                       TetrahedronIntegrator)
from gpaw.response import timer
from gpaw.response.pair import PairDensityCalculator
from gpaw.response.pw_parallelization import block_partition
from gpaw.response.symmetry import PWSymmetryAnalyzer
from gpaw.typing import Array1D
from gpaw.utilities.memory import maxrss
from gpaw.response.integrands import Integrand


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
    def __init__(self, chi0calc, optical, qpd, analyzer, bandsum):
        self._chi0calc = chi0calc

        mat_kwargs, eig_kwargs = chi0calc.get_integrator_arguments(
            qpd, analyzer, bandsum)
        if optical:
            self._matrix_element = partial(
                chi0calc.get_optical_matrix_element, **mat_kwargs)
            self._eigenvalues = partial(
                chi0calc.get_eigenvalues, **eig_kwargs)
        else:
            self._matrix_element = partial(
                chi0calc.get_matrix_element, **mat_kwargs)
            self._eigenvalues = partial(
                chi0calc.get_eigenvalues, **eig_kwargs)

    def matrix_element(self, *args, **kwargs):
        return self._matrix_element(*args, **kwargs)

    def eigenvalue(self, *args, **kwargs):
        return self._eigenvalues(*args, **kwargs)


class Chi0Calculator(Integrand):
    def __init__(self, wd, pair,
                 hilbert=True,
                 intraband=True,
                 nbands=None,
                 timeordered=False,
                 context=None,
                 ecut=None,
                 eta=0.2,
                 disable_point_group=False, disable_time_reversal=False,
                 disable_non_symmorphic=True,
                 integrationmode=None,
                 rate=0.0, eshift=0.0):

        if context is None:
            context = pair.context

        # TODO: More refactoring to avoid non-orthogonal inputs.
        assert pair.context.comm is context.comm
        self.context = context

        self.pair = pair
        self.gs = pair.gs

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        self.integrationmode = integrationmode
        self.eshift = eshift / Ha

        self.nblocks = pair.nblocks

        # XXX this is redundant as pair also does it.
        self.blockcomm, self.kncomm = block_partition(self.context.comm,
                                                      self.nblocks)

        if ecut is None:
            ecut = 50.0
        ecut /= Ha
        self.ecut = ecut

        self.eta = eta / Ha

        self.nbands = nbands or self.gs.bd.nbands

        self.wd = wd
        self.context.print(self.wd, flush=False)

        if not isinstance(self.wd, NonLinearFrequencyDescriptor):
            assert not hilbert

        self.hilbert = hilbert
        self.timeordered = bool(timeordered)
        if self.timeordered:
            assert self.hilbert  # Timeordered is only needed for G0W0

        if self.eta == 0.0:
            assert not hilbert
            assert not timeordered
            assert not self.wd.omega_w.real.any()

        self.pawcorr = None

        if sum(self.pbc) == 1:
            raise ValueError('1-D not supported atm.')

        self.context.print('Nonperiodic BCs: ', (~self.pbc), flush=False)

        if integrationmode is not None:
            self.context.print('Using integration method: ' +
                               self.integrationmode)
        else:
            self.context.print('Using integration method: PointIntegrator')

        # Number of completely filled bands and number of non-empty bands.
        self.nocc1, self.nocc2 = self.gs.count_occupied_bands()
        metallic = self.nocc1 != self.nocc2

        if metallic:
            assert abs(eshift) < 1e-8,\
                'A rigid energy shift cannot be applied to the conduction '\
                'bands if there is no band gap'

        # In the optical limit of metals, one must add the Drude dielectric
        # response from the free-space plasma frequency of the intraband
        # transitions to the head of the chi0 wings. This is handled by a
        # separate calculator, provided that intraband is set to True.
        if metallic and intraband:
            if rate == 'eta':
                rate = eta
            self.rate = rate
            self.drude_calc = Chi0DrudeCalculator(
                pair,
                disable_point_group=disable_point_group,
                disable_time_reversal=disable_time_reversal,
                disable_non_symmorphic=disable_non_symmorphic,
                integrationmode=integrationmode)
        else:
            self.drude_calc = None
            self.rate = None

    @property
    def pbc(self):
        return self.gs.pbc

    def create_chi0(self, q_c):
        # Extract descriptor arguments
        plane_waves = (q_c, self.ecut, self.gs.gd)
        parallelization = (self.context.comm, self.blockcomm, self.kncomm)

        # Construct the Chi0Data object
        # In the future, the frequencies should be specified at run-time
        # by Chi0.calculate(), in which case Chi0Data could also initialize
        # the frequency descriptor XXX
        chi0 = Chi0Data.from_descriptor_arguments(self.wd,
                                                  plane_waves,
                                                  parallelization)

        return chi0

    def calculate(self, q_c, spin='all'):
        """Calculate response function.

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
        chi0 = self.create_chi0(q_c)
        self.print_info(chi0.qpd)

        # Do all transitions into partially filled and empty bands
        m1 = self.nocc1
        m2 = self.nbands
        spins = self.get_spins(spin)

        chi0 = self.update_chi0(chi0, m1, m2, spins)

        if self.drude_calc is not None and chi0.optical_limit:
            # Add intraband contribution
            chi0_drude = self.drude_calc.calculate(self.wd, self.rate, spin)
            chi0.chi0_Wvv[:] += chi0_drude.chi_Zvv

        return chi0

    def get_spins(self, spin):
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
        """In-place calculation of the response function.

        Parameters
        ----------
        chi0 : Chi0Data
            Data and representation object
        m1 : int
            Lower band cutoff for band summation
        m2 : int
            Upper band cutoff for band summation
        spins : str or list(ints)
            If 'all' then include all spins.
            If [0] or [1], only include this specific spin.

        Returns
        -------
        chi0 : Chi0Data
        """
        assert m1 <= m2

        # Parse spins
        nspins = self.gs.nspins
        if spins == 'all':
            spins = range(nspins)
        else:
            for spin in spins:
                assert spin in range(nspins)

        qpd = chi0.qpd
        optical_limit = chi0.optical_limit  # Calculating the optical limit?

        # Reset PAW correction in case momentum has change
        pairden_paw_corr = self.gs.pair_density_paw_corrections
        self.pawcorr = pairden_paw_corr(qpd)

        # Integrate chi0 body
        self.context.print('Integrating response function.')
        self._update_chi0_body(chi0, m1, m2, spins)

        if optical_limit:
            # Update the head and wings
            self._update_chi0_wings(chi0, m1, m2, spins)

        return chi0

    def _update_chi0_body(self,
                          chi0: Chi0Data,
                          m1, m2, spins):
        """In-place calculation of the body."""
        qpd = chi0.qpd

        integrator = self.initialize_integrator()
        domain, analyzer, prefactor = self.get_integration_domain(qpd, spins)
        bandsum = self.get_band_summation(m1, m2)
        kind, extraargs = self.get_integral_kind()

        integrand = Chi0Integrand(self, qpd=qpd, analyzer=analyzer,
                                  optical=False,
                                  bandsum=bandsum)

        chi0.chi0_WgG[:] /= prefactor
        if self.hilbert:
            # Allocate a temporary array for the spectral function
            out_WgG = chi0.zeros()
        else:
            # Use the preallocated array for direct updates
            out_WgG = chi0.chi0_WgG
        integrator.integrate(kind=kind,  # Kind of integral
                             domain=domain,  # Integration domain
                             integrand=integrand,
                             x=self.wd,  # Frequency Descriptor
                             out_wxx=out_WgG,  # Output array
                             **extraargs)
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
            chi0.chi0_WgG[:] += out_WgG
        chi0.chi0_WgG[:] *= prefactor

        tmp_chi0_wGG = chi0.copy_array_with_distribution('wGG')
        analyzer.symmetrize_wGG(tmp_chi0_wGG)
        # The line below is borderline illegal and should be changed! XXX
        chi0.chi0_WgG[:] = chi0.blockdist.distribute_as(tmp_chi0_wGG,
                                                        chi0.nw, 'WgG')

    def _update_chi0_wings(self,
                           chi0: Chi0Data,
                           m1, m2, spins):
        """In-place calculation of the optical limit wings."""
        qpd = chi0.qpd

        integrator = self.initialize_integrator(block_distributed=False)
        domain, analyzer, prefactor = self.get_integration_domain(qpd, spins)
        bandsum = self.get_band_summation(m1, m2)
        kind, extraargs = self.get_integral_kind()

        integrand = Chi0Integrand(self, qpd=qpd, analyzer=analyzer,
                                  bandsum=bandsum, optical=True)

        # We integrate the head and wings together, using the combined index P
        # index v = (x, y, z)
        # index G = (G0, G1, G2, ...)
        # index P = (x, y, z, G1, G2, ...)
        WxvP_shape = list(chi0.WxvG_shape)
        WxvP_shape[-1] += 2
        tmp_chi0_WxvP = np.zeros(WxvP_shape, complex)
        integrator.integrate(kind=kind + ' wings',  # Kind of integral
                             domain=domain,  # Integration domain
                             integrand=integrand,
                             x=self.wd,  # Frequency Descriptor
                             out_wxx=tmp_chi0_WxvP,  # Output array
                             **extraargs)
        if self.hilbert:
            with self.context.timer('Hilbert transform'):
                ht = HilbertTransform(np.array(self.wd.omega_w), self.eta,
                                      timeordered=self.timeordered)
                ht(tmp_chi0_WxvP)
        tmp_chi0_WxvP *= prefactor

        # Fill in wings part of the data, but leave out the head part (G0)
        chi0.chi0_WxvG[..., 1:] += tmp_chi0_WxvP[..., 3:]
        # Fill in the head
        chi0.chi0_Wvv[:] += tmp_chi0_WxvP[:, 0, :3, :3]
        analyzer.symmetrize_wxvG(chi0.chi0_WxvG)
        analyzer.symmetrize_wvv(chi0.chi0_Wvv)

    def initialize_integrator(self, block_distributed=True) -> Integrator:
        """The integrator class is a general class for brillouin zone
        integration that can integrate user defined functions over user
        defined domains and sum over bands."""
        integrator: Integrator

        cls = self.get_integrator_cls()

        kwargs = dict(
            cell_cv=self.gs.gd.cell_cv,
            context=self.context)
        self.update_integrator_kwargs(kwargs,
                                      block_distributed=block_distributed)

        integrator = cls(**kwargs)

        return integrator

    def get_integrator_cls(self):
        """Get the appointed k-point integrator class."""
        if self.integrationmode is None:
            cls = PointIntegrator
        elif self.integrationmode == 'tetrahedron integration':
            cls = TetrahedronIntegrator  # type: ignore
            if not all([self.disable_point_group,
                        self.disable_time_reversal,
                        self.disable_non_symmorphic]):
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

    def update_integrator_kwargs(self, kwargs, block_distributed=True):
        # Update the energy shift
        kwargs['eshift'] = self.eshift

        # Update nblocks
        if block_distributed:
            kwargs['nblocks'] = self.nblocks

    def get_integration_domain(self, qpd, spins):
        """Get integrator domain and prefactor for the integral."""
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

    def get_integrator_arguments(self, qpd, analyzer, bandsum):
        # Prepare keyword arguments for the integrator
        mat_kwargs = {'qpd': qpd,
                      'symmetry': analyzer,
                      'integrationmode': self.integrationmode}
        eig_kwargs = {'qpd': qpd}

        mat_kwargs.update(bandsum)
        eig_kwargs.update(bandsum)

        return mat_kwargs, eig_kwargs

    def get_band_summation(self, m1, m2):
        """Define band summation."""
        # In a normal response calculation, we include transitions from all
        # completely and partially unoccupied bands to range(m1, m2)
        bandsum = {'n1': 0, 'n2': self.nocc2, 'm1': m1, 'm2': m2}

        return bandsum

    def get_integral_kind(self):
        """Determine what "kind" of integral to make."""
        extraargs = {}
        if self.eta == 0:
            # If eta is 0 then we must be working with imaginary frequencies.
            # In this case chi is hermitian and it is therefore possible to
            # reduce the computational costs by a only computing half of the
            # response function.
            kind = 'hermitian response function'
        elif self.hilbert:
            # The spectral function integrator assumes that the form of the
            # integrand is a function (a matrix element) multiplied by
            # a delta function and should return a function of at user defined
            # x's (frequencies). Thus the integrand is tuple of two functions
            # and takes an additional argument (x).
            kind = 'spectral function'
        else:
            # Otherwise, we can make no simplifying assumptions of the
            # form of the response function and we simply perform a brute
            # force calculation of the response function.
            kind = 'response function'
            extraargs['eta'] = self.eta

        return kind, extraargs

    @timer('Get kpoints')
    def get_kpoints(self, qpd, integrationmode):
        """Get the integration domain."""
        analyzer = PWSymmetryAnalyzer(
            self.gs.kd, qpd, self.context,
            disable_point_group=self.disable_point_group,
            disable_time_reversal=self.disable_time_reversal,
            disable_non_symmorphic=self.disable_non_symmorphic)

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

    @timer('Get matrix element')
    def get_matrix_element(self, k_v, s, n1, n2,
                           m1, m2, *, qpd,
                           symmetry, integrationmode=None):
        """A function that returns pair-densities.

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
        n1 : int
            Lower occupied band index.
        n2 : int
            Upper occupied band index.
        m1 : int
            Lower unoccupied band index.
        m2 : int
            Upper unoccupied band index.
        qpd : SingleQPWDescriptor instance
        symmetry: gpaw.response.pair.PWSymmetryAnalyzer instance
            Symmetry analyzer object for handling symmetries of the kpoints.
        integrationmode : str
            The integration mode employed.

        Return
        ------
        n_nmG : ndarray
            Pair densities.
        """
        return self._get_any_matrix_element(
            k_v, s, n1, n2, m1, m2, qpd=qpd,
            symmetry=symmetry, integrationmode=integrationmode,
            block=True,
            target_method=self.pair.get_pair_density,
        ).reshape(-1, qpd.ngmax)

    def _get_any_matrix_element(
            self, k_v, s, n1, n2, m1, m2, *, qpd,
            symmetry, integrationmode=None,
            block, target_method):
        assert m1 <= m2

        k_c = np.dot(qpd.gd.cell_cv, k_v) / (2 * np.pi)

        weight = np.sqrt(symmetry.get_kpoint_weight(k_c) /
                         symmetry.how_many_symmetries())
        if self.pawcorr is None:
            pairden_paw_corr = self.gs.pair_density_paw_corrections
            self.pawcorr = pairden_paw_corr(qpd)

        kptpair = self.pair.get_kpoint_pair(qpd, s, k_c, n1, n2,
                                            m1, m2, block=block)
        m_m = np.arange(m1, m2)
        n_n = np.arange(n1, n2)
        n_nmG = target_method(qpd, kptpair, n_n, m_m,
                              pawcorr=self.pawcorr,
                              block=block)

        if integrationmode is None:
            n_nmG *= weight

        df_nm = kptpair.get_occupation_differences(n_n, m_m)
        df_nm[df_nm <= 1e-20] = 0.0
        n_nmG *= df_nm[..., np.newaxis]**0.5

        return n_nmG

    @timer('Get matrix element')
    def get_optical_matrix_element(self, k_v, s,
                                   n1, n2,
                                   m1, m2, *,
                                   qpd, symmetry,
                                   integrationmode=None):
        """A function that returns optical pair densities, that is the
        head and wings matrix elements, indexed by:
        # P = (x, y, v, G1, G2, ...)."""

        return self._get_any_matrix_element(
            k_v, s, n1, n2, m1, m2, qpd=qpd,
            symmetry=symmetry, integrationmode=integrationmode,
            block=False,
            target_method=self.pair.get_optical_pair_density,
        ).reshape(-1, qpd.ngmax + 2)

    @timer('Get eigenvalues')
    def get_eigenvalues(self, k_v, s, n1, n2,
                        m1, m2, *, qpd,
                        gs=None, filter=False):
        """A function that can return the eigenvalues.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        if gs is None:
            gs = self.gs

        kd = gs.kd
        k_c = np.dot(qpd.gd.cell_cv, k_v) / (2 * np.pi)
        q_c = qpd.q_c
        K1 = self.pair.find_kpoint(k_c)
        K2 = self.pair.find_kpoint(k_c + q_c)

        ik1 = kd.bz2ibz_k[K1]
        ik2 = kd.bz2ibz_k[K2]
        kpt1 = gs.kpt_qs[ik1][s]
        assert gs.kd.comm.size == 1
        kpt2 = gs.kpt_qs[ik2][s]
        deps_nm = np.subtract(kpt1.eps_n[n1:n2][:, np.newaxis],
                              kpt2.eps_n[m1:m2])

        if filter:
            fermi_level = self.gs.fermi_level
            deps_nm[kpt1.eps_n[n1:n2] > fermi_level, :] = np.nan
            deps_nm[:, kpt2.eps_n[m1:m2] < fermi_level] = np.nan

        return deps_nm.reshape(-1)

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
        ecut = self.ecut * Ha
        nbands = self.nbands
        ngmax = qpd.ngmax
        eta = self.eta * Ha
        csize = comm.size
        knsize = self.kncomm.size
        bsize = self.blockcomm.size
        chisize = nw * qpd.ngmax**2 * 16. / 1024**2 / bsize

        p = partial(self.context.print, flush=False)

        p('%s' % ctime())
        p('Called response.chi0.calculate with:')
        p(self.get_gs_info_string(tab='    '))
        p()
        p('    Linear response parametrization:')
        p('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]))
        p('    Number of frequency points: %d' % nw)
        if bsize > nw:
            p('WARNING! Your nblocks is larger than number of frequency'
              ' points. Errors might occur, if your submodule does'
              ' not know how to handle this.')
        p('    Planewave cutoff: %f' % ecut)
        p('    Number of bands: %d' % nbands)
        p('    Number of planewaves: %d' % ngmax)
        p('    Broadening (eta): %f' % eta)
        p('    comm.size: %d' % csize)
        p('    kncomm.size: %d' % knsize)
        p('    blockcomm.size: %d' % bsize)
        p()
        p('    Memory estimate of potentially large arrays:')
        p('        chi0_wGG: %f M / cpu' % chisize)
        p('        Memory usage before allocation: %f M / cpu' % (maxrss() /
                                                                  1024**2))
        self.context.print('')

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

        nls = '\n' + tab  # newline string
        gs_str = tab + 'Ground state adapter containing:'
        gs_str += nls + 'Number of spins: %d' % ns
        gs_str += nls + 'Number of kpoints: %d' % nk
        gs_str += nls + 'Number of irredicible kpoints: %d' % nik
        gs_str += nls + 'Number of completely occupied states: %d' % nocc
        gs_str += nls + 'Number of partially occupied states: %d' % npocc
        gs_str += nls + 'Occupied states memory: %f M / cpu' % occsize

        return gs_str


class Chi0DrudeCalculator(Chi0Calculator):
    """Class for calculating the plasma frequency contribution to Chi0,
    that is, the contribution from intraband transitions inside of metallic
    bands. This corresponds directly to the dielectric function in the Drude
    model."""

    def __init__(self, pair,
                 disable_point_group=False,
                 disable_time_reversal=False,
                 disable_non_symmorphic=True,
                 integrationmode=None):
        self.pair = pair
        self.gs = pair.gs
        self.context = pair.context

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        self.integrationmode = integrationmode

        # Number of completely filled bands and number of non-empty bands.
        self.nocc1, self.nocc2 = self.gs.count_occupied_bands()

    def calculate(self, wd, rate, spin='all'):
        """Calculate the Drude dielectric response.

        Parameters
        ----------
        wd : FrequencyDescriptor
            Frequencies to evaluate the reponse function at.
        rate : float
            Plasma frequency decay rate (in eV), corresponding to the
            imaginary part of the complex frequency.
        spin : str or int
            If 'all' then include all spins.
            If 0 or 1, only include this specific spin.
        """
        self.print_info(wd, rate)

        # Parse the spin input
        spins = self.get_spins(spin)

        chi0_drude = Chi0DrudeData.from_frequency_descriptor(wd, rate)
        self._calculate(chi0_drude, spins)

        return chi0_drude

    def _calculate(self, chi0_drude: Chi0DrudeData, spins):
        """In-place calculation of the Drude dielectric response function,
        based on the free-space plasma frequency of the intraband transitions.
        """
        # Create a dummy plane-wave descriptor. We need this for the symmetry
        # analysis -> see discussion in gpaw.response.jdos
        qpd = SingleQPWDescriptor.from_q([0., 0., 0.],
                                         ecut=1e-3, gd=self.gs.gd)

        integrator = self.initialize_integrator()
        domain, analyzer, prefactor = self.get_integration_domain(qpd, spins)
        kind, extraargs = self.get_integral_kind()

        from gpaw.response.integrands import PlasmaFrequencyIntegrand
        integrand = PlasmaFrequencyIntegrand(self, qpd, analyzer)

        # Integrate using temporary array
        tmp_plasmafreq_wvv = np.zeros((1,) + chi0_drude.vv_shape, complex)
        integrator.integrate(kind=kind,  # Kind of integral
                             domain=domain,  # Integration domain
                             integrand=integrand,
                             out_wxx=tmp_plasmafreq_wvv,  # Output array
                             **extraargs)  # Extra args for int. method
        tmp_plasmafreq_wvv *= prefactor

        # Store the plasma frequency itself and print it for anyone to use
        plasmafreq_vv = tmp_plasmafreq_wvv[0].copy()
        analyzer.symmetrize_wvv(plasmafreq_vv[np.newaxis])
        chi0_drude.plasmafreq_vv += 4 * np.pi * plasmafreq_vv
        self.context.print('Plasma frequency:', flush=False)
        self.context.print((chi0_drude.plasmafreq_vv**0.5 * Ha).round(2))

        # Calculate the Drude dielectric response function from the
        # free-space plasma frequency
        # χ_D(ω+iη) = ω_p^2 / (ω+iη)^2
        assert chi0_drude.zd.upper_half_plane
        chi0_drude.chi_Zvv += plasmafreq_vv[np.newaxis] \
            / chi0_drude.zd.hz_z[:, np.newaxis, np.newaxis]**2

    def update_integrator_kwargs(self, *unused, **ignored):
        """The Drude calculator uses only standard integrator kwargs."""
        pass

    def get_integral_kind(self):
        """Define what "kind" of integral to make."""
        extraargs = {}
        # The plasma frequency integral is special in the way, that only
        # the spectral part is needed
        kind = 'spectral function'
        if self.integrationmode is None:
            # Calculate intraband transitions at finite fermi smearing
            extraargs['intraband'] = True  # Calculate intraband
        elif self.integrationmode == 'tetrahedron integration':
            # Calculate intraband transitions at T=0
            fermi_level = self.gs.fermi_level
            extraargs['x'] = FrequencyGridDescriptor([-fermi_level])

        return kind, extraargs

    def print_info(self, wd, rate):
        p = partial(self.context.print, flush=False)

        p('%s' % ctime())
        p('Called chi0_drude.calculate() with:')
        p('    Number of frequency points: %d' % len(wd))
        p('    Plasma frequency decay rate: %f eV' % rate)
        p()
        p(self.get_gs_info_string(tab='    '))
        self.context.print('')


class Chi0(Chi0Calculator):
    """Class for calculating non-interacting response functions.
    Tries to be backwards compatible, for now. """

    def __init__(self,
                 calc,
                 *,
                 frequencies: Union[dict, Array1D] = None,
                 ecut=50,
                 threshold=1,
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
        threshold : float
            Numerical threshold for the optical limit k dot p perturbation
            theory expansion (used in gpaw/response/pair.py).
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
        disable_non_symmorphic : bool
            Do no use non symmorphic symmetry operators.
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
        pair : gpaw.response.pair.PairDensity instance
            Class for calculating matrix elements of pairs of wavefunctions.

        """
        from gpaw.response.pair import get_gs_and_context
        gs, context = get_gs_and_context(calc, txt, world, timer)
        nbands = nbands or gs.bd.nbands

        wd = new_frequency_descriptor(
            gs, context, nbands, frequencies,
            domega0=domega0,
            omega2=omega2, omegamax=omegamax)

        pair = PairDensityCalculator(
            gs, context,
            threshold=threshold,
            nblocks=nblocks)

        super().__init__(wd=wd, pair=pair, nbands=nbands, ecut=ecut, **kwargs)


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
