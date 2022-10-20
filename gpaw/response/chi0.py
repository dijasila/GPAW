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
from gpaw.response.chi0_data import Chi0Data
from gpaw.response.frequencies import (FrequencyDescriptor,
                                       FrequencyGridDescriptor,
                                       NonLinearFrequencyDescriptor)
from gpaw.response.hilbert import HilbertTransform
from gpaw.response.integrators import (Integrator, PointIntegrator,
                                       TetrahedronIntegrator)
from gpaw.response import timer
from gpaw.response.pair import NoCalculatorPairDensity
from gpaw.response.pw_parallelization import block_partition
from gpaw.response.symmetry import PWSymmetryAnalyzer
from gpaw.typing import Array1D
from gpaw.utilities.memory import maxrss


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


class Chi0Calculator:
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
                 ftol=1e-6,
                 rate=0.0, eshift=0.0):

        if context is None:
            context = pair.context

        # TODO: More refactoring to avoid non-orthogonal inputs.
        assert pair.context.world is context.world
        self.context = context

        self.pair = pair
        self.gs = pair.gs

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        self.integrationmode = integrationmode
        self.eshift = eshift / Ha

        self.nblocks = pair.nblocks
        self.calc = self.gs._calc  # XXX remove me

        # XXX this is redundant as pair also does it.
        self.blockcomm, self.kncomm = block_partition(self.context.world,
                                                      self.nblocks)

        if ecut is None:
            ecut = 50.0
        ecut /= Ha
        self.ecut = ecut

        self.eta = eta / Ha
        if rate == 'eta':
            self.rate = self.eta
        else:
            self.rate = rate / Ha

        self.nbands = nbands or self.gs.bd.nbands
        self.include_intraband = intraband

        self.wd = wd
        self.context.print(self.wd, flush=False)

        if not isinstance(self.wd, NonLinearFrequencyDescriptor):
            assert not hilbert

        self.hilbert = hilbert
        self.timeordered = bool(timeordered)

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
        self.nocc1, self.nocc2 = self.gs.count_occupied_bands(ftol)

    @property
    def pbc(self):
        return self.gs.pbc

    def create_chi0(self, q_c):
        # Extract descriptor arguments
        plane_waves = (q_c, self.ecut, self.gs.gd)
        parallelization = (self.context.world, self.blockcomm, self.kncomm)

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
            (not used in transverse response functions)

        Returns
        -------
        chi0 : Chi0Data
            Data object containing the chi0 data arrays along with basis
            representation descriptors and blocks distribution
        """
        gs = self.gs

        if spin == 'all':
            spins = range(gs.nspins)
        else:
            assert spin in range(gs.nspins)
            spins = [spin]

        chi0 = self.create_chi0(q_c)

        self.print_chi(chi0.pd)

        if chi0.optical_limit:
            self.plasmafreq_vv = np.zeros((3, 3), complex)
        else:
            self.plasmafreq_vv = None

        # Do all transitions into partially filled and empty bands
        m1 = self.nocc1
        m2 = self.nbands

        chi0 = self.update_chi0(chi0, m1, m2, spins)

        return chi0

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

        pd = chi0.pd
        optical_limit = chi0.optical_limit  # Calculating the optical limit?

        # Reset PAW correction in case momentum has change
        pairden_paw_corr = self.gs.pair_density_paw_corrections
        self.pawcorr = pairden_paw_corr(pd, alter_optical_limit=True)

        # Integrate chi0 body
        self.context.print('Integrating response function.')
        self._update_chi0_body(chi0, m1, m2, spins)

        if optical_limit:
            # Integrate the chi0 wings
            self._update_chi0_wings(chi0, m1, m2, spins)

            # In the optical limit of metals, additional work must be performed
            # (one must add the Drude dielectric response from the free-space
            # plasma frequency of the intraband transitions to the head of the
            # chi0 wings).
            if self.nocc1 != self.nocc2 and self.include_intraband:
                self._update_chi0_drude(chi0, m1, m2, spins)

            # In the optical limit, we fill in the G=0 entries of chi0_wGG with
            # the wings evaluated along the z-direction by default.
            # The x = 1 wing represents the left vertical block, which is
            # distributed in chi0_wGG
            chi0.chi0_wGG[:, :, 0] = chi0.chi0_wxvG[:, 1, 2,
                                                    chi0.blocks1d.myslice]
            if self.blockcomm.rank == 0:  # rank with G=0 row
                # The x = 0 wing represents the upper horizontal block
                chi0.chi0_wGG[:, 0, :] = chi0.chi0_wxvG[:, 0, 2, :]
                chi0.chi0_wGG[:, 0, 0] = chi0.chi0_wvv[:, 2, 2]

        return chi0

    def _update_chi0_body(self,
                          chi0: Chi0Data,
                          m1, m2, spins):
        """In-place calculation of the body."""
        chi0_wGG = chi0.chi0_wGG  # Change notation
        pd = chi0.pd

        integrator = self.initialize_integrator()
        domain, analyzer, prefactor = self.get_integration_domain(pd, spins)
        mat_kwargs, eig_kwargs = self.get_integrator_arguments(pd, m1, m2,
                                                               analyzer)
        kind, extraargs = self.get_integral_kind()

        get_matrix_element = partial(
            self.get_matrix_element, **mat_kwargs)
        get_eigenvalues = partial(
            self.get_eigenvalues, **eig_kwargs)

        chi0_wGG /= prefactor
        integrator.integrate(kind=kind,  # Kind of integral
                             domain=domain,  # Integration domain
                             integrand=(get_matrix_element,
                                        get_eigenvalues),
                             x=self.wd,  # Frequency Descriptor
                             out_wxx=chi0_wGG,  # Output array
                             **extraargs)
        if self.hilbert:
            # The integrator only returns the spectral function and a Hilbert
            # transform is performed to return the real part of the density
            # response function.
            with self.context.timer('Hilbert transform'):
                # Make Hilbert transform
                ht = HilbertTransform(np.array(self.wd.omega_w), self.eta,
                                      timeordered=self.timeordered)
                ht(chi0_wGG)
        chi0_wGG *= prefactor

        tmp_chi0_wGG = chi0.blockdist.redistribute(chi0_wGG, chi0.nw)
        analyzer.symmetrize_wGG(tmp_chi0_wGG)
        chi0_wGG[:] = chi0.blockdist.redistribute(tmp_chi0_wGG, chi0.nw)

    def _update_chi0_wings(self,
                           chi0: Chi0Data,
                           m1, m2, spins):
        """In-place calculation of the optical limit wings."""
        pd = chi0.pd

        integrator = self.initialize_integrator(block_distributed=False)
        domain, analyzer, prefactor = self.get_integration_domain(pd, spins)
        mat_kwargs, eig_kwargs = self.get_integrator_arguments(pd, m1, m2,
                                                               analyzer)
        kind, extraargs = self.get_integral_kind()

        get_optical_matrix_element = partial(
            self.get_optical_matrix_element, **mat_kwargs)
        get_eigenvalues = partial(
            self.get_eigenvalues, **eig_kwargs)

        tmp_chi0_wxvx = np.zeros(np.array(chi0.chi0_wxvG.shape) +
                                 [0, 0, 0, 2],  # Do both head and wings
                                 complex)
        integrator.integrate(kind=kind + ' wings',  # Kind of integral
                             domain=domain,  # Integration domain
                             integrand=(get_optical_matrix_element,
                                        get_eigenvalues),
                             x=self.wd,  # Frequency Descriptor
                             out_wxx=tmp_chi0_wxvx,  # Output array
                             **extraargs)
        if self.hilbert:
            with self.context.timer('Hilbert transform'):
                ht = HilbertTransform(np.array(self.wd.omega_w), self.eta,
                                      timeordered=self.timeordered)
                ht(tmp_chi0_wxvx)
        tmp_chi0_wxvx *= prefactor

        # Fill in wings part of the data, but leave out the head
        chi0.chi0_wxvG[..., 1:] += tmp_chi0_wxvx[..., 3:]
        # Fill in the head
        chi0.chi0_wvv += tmp_chi0_wxvx[:, 0, :3, :3]
        analyzer.symmetrize_wxvG(chi0.chi0_wxvG)
        analyzer.symmetrize_wvv(chi0.chi0_wvv)

    def _update_chi0_drude(self,
                           chi0: Chi0Data,
                           m1, m2, spins):
        """In-place calculation of the Drude dielectric response function,
        based on the free-space plasma frequency of the intraband transitions.
        """
        pd = chi0.pd

        integrator = self.initialize_integrator(block_distributed=False)
        domain, analyzer, prefactor = self.get_integration_domain(pd, spins)
        (mat_kwargs,
         eig_kwargs) = self.get_integrator_arguments(pd, m1, m2, analyzer,
                                                     only_intraband=True)
        kind, extraargs = self.get_integral_kind(only_intraband=True)

        get_plasmafreq_matrix_element = partial(
            self.get_plasmafreq_matrix_element, **mat_kwargs)
        get_plasmafreq_eigenvalue = partial(
            self.get_plasmafreq_eigenvalue, **eig_kwargs)

        tmp_plasmafreq_wvv = np.zeros((1, 3, 3), complex)  # Output array
        integrator.integrate(kind=kind,  # Kind of integral
                             domain=domain,  # Integration domain
                             integrand=(get_plasmafreq_matrix_element,
                                        get_plasmafreq_eigenvalue),
                             out_wxx=tmp_plasmafreq_wvv,  # Output array
                             **extraargs)  # Extra args for int. method
        tmp_plasmafreq_wvv *= prefactor

        # Store the plasma frequency itself and print it for anyone to use
        plasmafreq_vv = tmp_plasmafreq_wvv[0].copy()
        analyzer.symmetrize_wvv(plasmafreq_vv[np.newaxis])
        self.plasmafreq_vv += 4 * np.pi * plasmafreq_vv
        self.context.print('Plasma frequency:', flush=False)
        self.context.print((self.plasmafreq_vv**0.5 * Ha).round(2), flush=True)
    
        # Calculate the Drude dielectric response function from the
        # free-space plasma frequency
        try:
            with np.errstate(divide='raise'):
                drude_chi_wvv = (
                    plasmafreq_vv[np.newaxis] /
                    (self.wd.omega_w[:, np.newaxis, np.newaxis]
                     + 1.j * self.rate)**2)
        except FloatingPointError:
            raise ValueError('Please set rate to a positive value.')

        # Fill the Drude dielectric function into the chi0 head
        chi0.chi0_wvv[:] += drude_chi_wvv

    def initialize_integrator(self, block_distributed=True):
        """The integrator class is a general class for brillouin zone
        integration that can integrate user defined functions over user
        defined domains and sum over bands."""
        integrator: Integrator

        if self.integrationmode is None or \
           self.integrationmode == 'point integration':
            cls = PointIntegrator
        elif self.integrationmode == 'tetrahedron integration':
            cls = TetrahedronIntegrator  # type: ignore
        else:
            raise ValueError(f'Integration mode "{self.integrationmode}"'
                             ' not implemented.')

        kwargs = dict(
            cell_cv=self.gs.gd.cell_cv,
            context=self.context,
            eshift=self.eshift)

        if block_distributed:
            integrator = cls(**kwargs, nblocks=self.nblocks)
        else:
            integrator = cls(**kwargs)

        return integrator

    def get_integration_domain(self, pd, spins):
        """Get integrator domain and prefactor for the integral."""
        # The integration domain is determined by the following function
        # that reduces the integration domain to the irreducible zone
        # of the little group of q.
        bzk_kv, analyzer = self.get_kpoints(
            pd, integrationmode=self.integrationmode)
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

    def get_integrator_arguments(self, pd, m1, m2, analyzer,
                                 only_intraband=False):
        # Prepare keyword arguments for the integrator
        mat_kwargs = {'pd': pd,
                      'symmetry': analyzer,
                      'integrationmode': self.integrationmode}
        eig_kwargs = {'pd': pd}

        # Define band summation.
        if not only_intraband:
            # Normally, we include transitions from all completely and
            # partially filled bands to range(m1, m2)
            bandsum = {'n1': 0, 'n2': self.nocc2, 'm1': m1, 'm2': m2}
        else:
            # When doing a calculation of the intraband response, we need only
            # the partially filled bands
            bandsum = {'n1': self.nocc1, 'n2': self.nocc2}
            mat_kwargs.pop('integrationmode')  # Can we clean up here? XXX
        mat_kwargs.update(bandsum)
        eig_kwargs.update(bandsum)

        return mat_kwargs, eig_kwargs

    def get_integral_kind(self, only_intraband=False):
        """Determine what "kind" of integral to make."""
        extraargs = {}  # Initialize extra arguments to integration method.
        if only_intraband:
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
        elif self.eta == 0:
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
            extraargs['timeordered'] = self.timeordered

        return kind, extraargs

    def reduce_ecut(self, ecut, chi0: Chi0Data):
        """
        Function to provide chi0 quantities with reduced ecut
        needed for ecut extrapolation. See g0w0.py for usage.
        """
        from gpaw.pw.descriptor import (PWDescriptor,
                                        PWMapping)
        from gpaw.response.pw_parallelization import Blocks1D
        nG = chi0.pd.ngmax
        blocks1d = chi0.blocks1d

        # The copy() is only required when doing GW_too, since we need
        # to run this whole thin twice.
        chi0_wGG = chi0.blockdist.redistribute(chi0.chi0_wGG.copy(), chi0.nw)

        pd = chi0.pd
        chi0_wxvG = chi0.chi0_wxvG
        chi0_wvv = chi0.chi0_wvv

        if ecut == pd.ecut:
            pdi = pd
            G2G = None

        elif ecut < pd.ecut:  # construct subset chi0 matrix with lower ecut
            pdi = PWDescriptor(ecut, pd.gd, dtype=pd.dtype,
                               kd=pd.kd)
            nG = pdi.ngmax
            blocks1d = Blocks1D(self.pair.blockcomm, nG)
            G2G = PWMapping(pdi, pd).G2_G1
            chi0_wGG = chi0_wGG.take(G2G, axis=1).take(G2G, axis=2)

            if chi0_wxvG is not None:
                chi0_wxvG = chi0_wxvG.take(G2G, axis=3)

            if self.pawcorr is not None:
                self.pawcorr = self.pawcorr.reduce_ecut(G2G)

        return pdi, blocks1d, G2G, chi0_wGG, chi0_wxvG, chi0_wvv

    @timer('Get kpoints')
    def get_kpoints(self, pd, integrationmode=None):
        """Get the integration domain."""
        analyzer = PWSymmetryAnalyzer(
            self.gs.kd, pd, self.context,
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

        bzk_kv = np.dot(bzk_kc, pd.gd.icell_cv) * 2 * np.pi

        return bzk_kv, analyzer

    @timer('Get matrix element')
    def get_matrix_element(self, k_v, s, n1, n2,
                           m1, m2, *, pd,
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
        pd : PlanewaveDescriptor instance
        kd : KpointDescriptor instance
            Calculator kpoint descriptor.
        symmetry: gpaw.response.pair.PWSymmetryAnalyzer instance
            Symmetry analyzer object for handling symmetries of the kpoints.
        integrationmode : str
            The integration mode employed.

        Return
        ------
        n_nmG : ndarray
            Pair densities.
        """
        assert m1 <= m2

        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)

        nG = pd.ngmax
        weight = np.sqrt(symmetry.get_kpoint_weight(k_c) /
                         symmetry.how_many_symmetries())
        if self.pawcorr is None:
            pairden_paw_corr = self.gs.pair_density_paw_corrections
            self.pawcorr = pairden_paw_corr(pd, alter_optical_limit=True)

        kptpair = self.pair.get_kpoint_pair(pd, s, k_c, n1, n2,
                                            m1, m2, block=True)
        m_m = np.arange(m1, m2)
        n_n = np.arange(n1, n2)
        n_nmG = self.pair.get_pair_density(pd, kptpair, n_n, m_m,
                                           pawcorr=self.pawcorr,
                                           block=True)

        if integrationmode is None:
            n_nmG *= weight

        df_nm = kptpair.get_occupation_differences(n_n, m_m)
        df_nm[df_nm <= 1e-20] = 0.0
        n_nmG *= df_nm[..., np.newaxis]**0.5

        return n_nmG.reshape(-1, nG)

    @timer('Get matrix element')
    def get_optical_matrix_element(self, k_v, s,
                                   n1, n2,
                                   m1, m2, *,
                                   pd, symmetry,
                                   integrationmode=None):
        """A function that returns optical pair densities.
        NB: In dire need of further documentation! XXX"""
        assert m1 <= m2

        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)

        nG = pd.ngmax
        weight = np.sqrt(symmetry.get_kpoint_weight(k_c) /
                         symmetry.how_many_symmetries())
        if self.pawcorr is None:
            pairden_paw_corr = self.gs.pair_density_paw_corrections
            self.pawcorr = pairden_paw_corr(pd, alter_optical_limit=True)

        kptpair = self.pair.get_kpoint_pair(pd, s, k_c, n1, n2,
                                            m1, m2, block=False)
        m_m = np.arange(m1, m2)
        n_n = np.arange(n1, n2)
        n_nmG = self.pair.get_full_pair_density(pd, kptpair, n_n, m_m,
                                                pawcorr=self.pawcorr,
                                                block=False)

        if integrationmode is None:
            n_nmG *= weight

        df_nm = kptpair.get_occupation_differences(n_n, m_m)
        df_nm[df_nm <= 1e-20] = 0.0
        n_nmG *= df_nm[..., np.newaxis]**0.5

        return n_nmG.reshape(-1, nG + 2)

    @timer('Get eigenvalues')
    def get_eigenvalues(self, k_v, s, n1, n2,
                        m1, m2, *, pd,
                        gs=None, filter=False):
        """A function that can return the eigenvalues.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        if gs is None:
            gs = self.gs

        kd = gs.kd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        q_c = pd.kd.bzk_kc[0]
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

    def get_plasmafreq_matrix_element(self, k_v, s, n1, n2,
                                      *, pd,
                                      symmetry,
                                      integrationmode=None):
        """NB: In dire need of documentation! XXX."""
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        kpt1 = self.pair.get_k_point(s, k_c, n1, n2)
        n_n = range(n1, n2)

        vel_nv = self.pair.intraband_pair_density(kpt1, n_n)

        if self.integrationmode is None:
            f_n = kpt1.f_n
            width = self.gs.get_occupations_width()
            if width > 1e-15:
                dfde_n = - 1. / width * (f_n - f_n**2.0)
            else:
                dfde_n = np.zeros_like(f_n)
            vel_nv *= np.sqrt(-dfde_n[:, np.newaxis])
            weight = np.sqrt(symmetry.get_kpoint_weight(k_c) /
                             symmetry.how_many_symmetries())
            vel_nv *= weight

        return vel_nv

    def get_plasmafreq_eigenvalue(self, k_v, s,
                                  n1, n2, *, pd):
        """A function that can return the intraband eigenvalues.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        gs = self.gs
        kd = gs.kd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = self.pair.find_kpoint(k_c)
        ik = kd.bz2ibz_k[K1]
        kpt1 = gs.kpt_qs[ik][s]
        assert gs.kd.comm.size == 1

        return kpt1.eps_n[n1:n2]

    def print_chi(self, pd):
        gs = self.gs
        gd = gs.gd

        if gpaw.dry_run:
            from gpaw.mpi import SerialCommunicator
            size = gpaw.dry_run
            world = SerialCommunicator()
            world.size = size
        else:
            world = self.context.world

        q_c = pd.kd.bzk_kc[0]
        nw = len(self.wd)
        ecut = self.ecut * Ha
        ns = gs.nspins
        nbands = self.nbands
        nk = gs.kd.nbzkpts
        nik = gs.kd.nibzkpts
        ngmax = pd.ngmax
        eta = self.eta * Ha
        wsize = world.size
        knsize = self.kncomm.size
        nocc = self.nocc1
        npocc = self.nocc2
        ngridpoints = gd.N_c[0] * gd.N_c[1] * gd.N_c[2]
        nstat = (ns * npocc + world.size - 1) // world.size
        occsize = nstat * ngridpoints * 16. / 1024**2
        bsize = self.blockcomm.size
        chisize = nw * pd.ngmax**2 * 16. / 1024**2 / bsize

        p = partial(self.context.print, flush=False)

        p('%s' % ctime())
        p('Called response.chi0.calculate with')
        p('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]))
        p('    Number of frequency points: %d' % nw)
        if bsize > nw:
            p('WARNING! Your nblocks is larger than number of frequency'
              ' points. Errors might occur, if your submodule does'
              ' not know how to handle this.')
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
        p()
        p('    Memory estimate of potentially large arrays:')
        p('        chi0_wGG: %f M / cpu' % chisize)
        p('        Occupied states: %f M / cpu' % occsize)
        p('        Memory usage before allocation: %f M / cpu' % (maxrss() /
                                                                  1024**2))
        self.context.print('')


class Chi0(Chi0Calculator):
    """Class for calculating non-interacting response functions.
    Tries to be backwards compatible, for now. """

    def __init__(self,
                 calc,
                 *,
                 frequencies: Union[dict, Array1D] = None,
                 ecut=50,
                 ftol=1e-6, threshold=1,
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
        ftol : float
            Threshold determining whether a band is completely filled
            (f > 1 - ftol) or completely empty (f < ftol).
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

        wd = new_frequency_descriptor(gs, context, nbands, frequencies,
                                      domega0=domega0,
                                      omega2=omega2, omegamax=omegamax)

        pair = NoCalculatorPairDensity(gs, context,
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
