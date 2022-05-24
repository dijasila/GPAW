from __future__ import annotations
import warnings
from functools import partial
from time import ctime
from typing import Union

import gpaw
import gpaw.mpi as mpi
import numpy as np
from ase.units import Ha
from ase.utils.timing import Timer, timer
from gpaw.bztools import convex_hull_volume
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.response.frequencies import (FrequencyDescriptor,
                                       LinearFrequencyDescriptor,
                                       NonLinearFrequencyDescriptor)
from gpaw.response.pw_parallelization import (block_partition, Blocks1D,
                                              PlaneWaveBlockDistributor)
from gpaw.response.integrators import PointIntegrator, TetrahedronIntegrator
from gpaw.response.pair import PairDensity, PWSymmetryAnalyzer
from gpaw.utilities.blas import gemm
from gpaw.utilities.memory import maxrss
from gpaw.typing import Array1D
from gpaw.response.chi0_data import Chi0Data


def find_maximum_frequency(calc, nbands=0, fd=None):
    """Determine the maximum electron-hole pair transition energy."""
    epsmin = 10000.0
    epsmax = -10000.0
    for kpt in calc.wfs.kpt_u:
        epsmin = min(epsmin, kpt.eps_n[0])
        epsmax = max(epsmax, kpt.eps_n[nbands - 1])

    if fd is not None:
        print('Minimum eigenvalue: %10.3f eV' % (epsmin * Ha), file=fd)
        print('Maximum eigenvalue: %10.3f eV' % (epsmax * Ha), file=fd)

    return epsmax - epsmin


class Chi0:
    """Class for calculating non-interacting response functions."""

    def __init__(self,
                 calc,
                 *,
                 frequencies: Union[dict, Array1D] = None,
                 ecut=50, gammacentered=False, hilbert=True, nbands=None,
                 timeordered=False, eta=0.2, ftol=1e-6, threshold=1,
                 real_space_derivatives=False, intraband=True,
                 world=mpi.world, txt='-', timer=None,
                 nblocks=1,
                 disable_point_group=False, disable_time_reversal=False,
                 disable_non_symmorphic=True,
                 integrationmode=None,
                 rate=0.0, eshift=0.0,
                 domega0=None,  # deprecated
                 omega2=None,  # deprecated
                 omegamax=None  # deprecated
                 ):
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
        gammacentered : bool
            Center the grid of plane waves around the gamma point or q-vector
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
        real_space_derivatives : bool
            Switch for calculating nabla matrix elements (in the optical limit)
            using a real space finite difference approximation.
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
        if domega0 is not None or omega2 is not None or omegamax is not None:
            assert frequencies is None
            frequencies = {'type': 'nonlinear',
                           'domega0': domega0,
                           'omega2': omega2,
                           'omegamax': omegamax}
            warnings.warn(f'Please use frequencies={frequencies}')

        elif frequencies is None:
            frequencies = {'type': 'nonlinear'}

        self.timer = timer or Timer()

        self.pair = PairDensity(calc, ecut,
                                ftol, threshold,
                                real_space_derivatives, world, txt,
                                self.timer,
                                nblocks=nblocks)

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        self.integrationmode = integrationmode
        self.eshift = eshift / Ha

        calc = self.pair.calc
        self.calc = calc

        self.fd = self.pair.fd

        self.vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))

        self.world = world

        self.blockcomm, self.kncomm = block_partition(world, nblocks)

        self.nblocks = nblocks

        if ecut is not None:
            ecut /= Ha

        self.ecut = ecut
        self.gammacentered = gammacentered

        self.eta = eta / Ha
        if rate == 'eta':
            self.rate = self.eta
        else:
            self.rate = rate / Ha

        self.nbands = nbands or self.calc.wfs.bd.nbands
        self.include_intraband = intraband

        if (isinstance(frequencies, dict) and
            frequencies.get('omegamax') is None):
            omegamax = find_maximum_frequency(self.calc,
                                              nbands=self.nbands,
                                              fd=self.fd)
            frequencies['omegamax'] = omegamax * Ha

        self.wd = FrequencyDescriptor.from_array_or_dict(frequencies)
        print(self.wd, file=self.fd)

        # self.blocks1d = None  # Plane wave basis depends on q
        # self.blockdist = None

        if not isinstance(self.wd, NonLinearFrequencyDescriptor):
            assert not hilbert

        self.hilbert = hilbert
        self.timeordered = bool(timeordered)

        if self.eta == 0.0:
            assert not hilbert
            assert not timeordered
            assert not self.wd.omega_w.real.any()

        self.nocc1 = self.pair.nocc1  # number of completely filled bands
        self.nocc2 = self.pair.nocc2  # number of non-empty bands

        self.Q_aGii = None

        self.pbc = self.calc.wfs.gd.pbc_c
        if sum(self.pbc) < 2:
            raise ValueError('Only one non-periodic direction supported atm.')

        print('Nonperiodic BCs: ', (~self.pbc),
              file=self.fd)

        if integrationmode is not None:
            print('Using integration method: ' + self.integrationmode,
                  file=self.fd)
        else:
            print('Using integration method: PointIntegrator', file=self.fd)

    def create_chi0(self, q_c, extend_head=True):
        q_c = np.asarray(q_c, dtype=float)
        optical_limit = np.allclose(q_c, 0.0)

        pd = self.get_PWDescriptor(q_c, self.gammacentered)

        # Initialize block distibution of plane wave basis
        nG = pd.ngmax
        if optical_limit and extend_head:
            nG += 2
        blocks1d = Blocks1D(self.blockcomm, nG)
        blockdist = PlaneWaveBlockDistributor(self.world,
                                              self.blockcomm,
                                              self.kncomm,
                                              self.wd, blocks1d)

        chi0 = Chi0Data(self.wd, blockdist, pd, optical_limit, extend_head)

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
        A_x : ndarray
            Output array. If None, the output array is created.

        Returns
        -------
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        chi0_wGG : ndarray
            The response function.
        chi0_wxvG : ndarray or None
            (Only in optical limit) Wings of the density response function.
        chi0_wvv : ndarray or None
            (Only in optical limit) Head of the density response function.

        """
        wfs = self.calc.wfs

        if spin == 'all':
            spins = range(wfs.nspins)
        else:
            assert spin in range(wfs.nspins)
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
        q_c : list or ndarray
            Momentum vector..
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        chi0_wGG : ndarray
            The response function.
        chi0_wxvG : ndarray or None
            Wings of the density response function.
        chi0_wvv : ndarray or None
            Head of the density response function.
        m1 : int
            Lower band cutoff for band summation
        m2 : int
            Upper band cutoff for band summation
        spins : str or list(ints)
            If 'all' then include all spins.
            If [0] or [1], only include this specific spin.
        extend_head : bool
            If True: Extend the wings and head of chi in the optical limit to
            take into account the non-analytic nature of chi. Effectively
            means that chi has dimension (nw, nG + 2, nG + 2) in the optical
            limit. This simplifies the code and should only be switched off
            for parts of the code that do not support this feature i.e., GW
            RPA total energy and RALDA.
        """
        assert m1 <= m2
        # Parse spins
        wfs = self.calc.wfs
        if spins == 'all':
            spins = range(wfs.nspins)
        else:
            for spin in spins:
                assert spin in range(wfs.nspins)

        pd = chi0.pd
        # Are we calculating the optical limit.
        optical_limit = chi0.optical_limit

        # Use wings in optical limit, if head cannot be extended
        if optical_limit and not chi0.extend_head:
            wings = True
        else:
            wings = False

        # Reset PAW correction in case momentum has change
        self.Q_aGii = self.pair.initialize_paw_corrections(pd)
        A_wxx = chi0.chi0_wGG  # Change notation

        # Initialize integrator. The integrator class is a general class
        # for brillouin zone integration that can integrate user defined
        # functions over user defined domains and sum over bands.
        if self.integrationmode is None or \
           self.integrationmode == 'point integration':
            integrator = PointIntegrator(self.pair.calc.wfs.gd.cell_cv,
                                         comm=self.world,
                                         timer=self.timer,
                                         txt=self.fd,
                                         eshift=self.eshift,
                                         nblocks=self.nblocks)
            intnoblock = PointIntegrator(self.pair.calc.wfs.gd.cell_cv,
                                         comm=self.world,
                                         timer=self.timer,
                                         eshift=self.eshift,
                                         txt=self.fd)
        elif self.integrationmode == 'tetrahedron integration':
            integrator = TetrahedronIntegrator(self.pair.calc.wfs.gd.cell_cv,
                                               comm=self.world,
                                               timer=self.timer,
                                               eshift=self.eshift,
                                               txt=self.fd,
                                               nblocks=self.nblocks)
            intnoblock = TetrahedronIntegrator(self.pair.calc.wfs.gd.cell_cv,
                                               comm=self.world,
                                               timer=self.timer,
                                               eshift=self.eshift,
                                               txt=self.fd)
        else:
            print('Integration mode ' + self.integrationmode +
                  ' not implemented.', file=self.fd)
            raise NotImplementedError

        # The integration domain is determined by the following function
        # that reduces the integration domain to the irreducible zone
        # of the little group of q.
        bzk_kv, PWSA = self.get_kpoints(pd,
                                        integrationmode=self.integrationmode)
        domain = (bzk_kv, spins)

        if self.integrationmode == 'tetrahedron integration':
            # If there are non-periodic directions it is possible that the
            # integration domain is not compatible with the symmetry operations
            # which essentially means that too large domains will be
            # integrated. We normalize by vol(BZ) / vol(domain) to make
            # sure that to fix this.
            domainvol = convex_hull_volume(bzk_kv) * PWSA.how_many_symmetries()
            bzvol = (2 * np.pi)**3 / self.vol
            factor = bzvol / domainvol
        else:
            factor = 1

        prefactor = (2 * factor * PWSA.how_many_symmetries() /
                     (wfs.nspins * (2 * np.pi)**3))  # Remember prefactor

        if self.integrationmode is None:
            if self.calc.wfs.kd.refine_info is not None:
                nbzkpts = self.calc.wfs.kd.refine_info.mhnbzkpts
            else:
                nbzkpts = self.calc.wfs.kd.nbzkpts
            prefactor *= len(bzk_kv) / nbzkpts

        A_wxx /= prefactor
        if wings:
            chi0.chi0_wxvG /= prefactor
            chi0.chi0_wvv /= prefactor

        # The functions that are integrated are defined in the bottom
        # of this file and take a number of constant keyword arguments
        # which the integrator class accepts through the use of the
        # kwargs keyword.
        kd = self.calc.wfs.kd
        mat_kwargs = {'kd': kd, 'pd': pd,
                      'symmetry': PWSA,
                      'integrationmode': self.integrationmode}
        eig_kwargs = {'kd': kd, 'pd': pd}

        if not chi0.extend_head:
            mat_kwargs['extend_head'] = False

        # Determine what "kind" of integral to make.
        extraargs = {}  # Initialize extra arguments to integration method.
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
            extraargs['timeordered'] = self.timeordered

        # Integrate response function
        print('Integrating response function.', file=self.fd)
        # Define band summation. Includes transitions from all
        # completely and partially filled bands to range(m1, m2)
        bandsum = {'n1': 0, 'n2': self.nocc2, 'm1': m1, 'm2': m2}
        mat_kwargs.update(bandsum)
        eig_kwargs.update(bandsum)

        integrator.integrate(kind=kind,  # Kind of integral
                             domain=domain,  # Integration domain
                             integrand=(self.get_matrix_element,  # Integrand
                                        self.get_eigenvalues),  # Integrand
                             x=self.wd,  # Frequency Descriptor
                             kwargs=(mat_kwargs, eig_kwargs),
                             # Arguments for integrand functions
                             out_wxx=A_wxx,  # Output array
                             **extraargs)
        # extraargs: Extra arguments to integration method
        if wings:
            mat_kwargs['extend_head'] = True
            mat_kwargs['block'] = False
            if self.eta == 0:
                extraargs['eta'] = self.eta
            # This is horrible but we need to update the wings manually
            # in order to make them work with ralda, RPA and GW. This entire
            # section can be deleted in the future if the ralda and RPA code is
            # made compatible with the head and wing extension that other parts
            # of the code is using.
            chi0_wxvx = np.zeros(np.array(chi0.chi0_wxvG.shape) +
                                 [0, 0, 0, 2],
                                 complex)  # Notice the wxv"x" for head extend
            intnoblock.integrate(kind=kind + ' wings',  # kind'o int.
                                 domain=domain,  # Integration domain
                                 integrand=(self.get_matrix_element,  # Intgrnd
                                            self.get_eigenvalues),  # Integrand
                                 x=self.wd,  # Frequency Descriptor
                                 kwargs=(mat_kwargs, eig_kwargs),
                                 # Arguments for integrand functions
                                 out_wxx=chi0_wxvx,  # Output array
                                 **extraargs)

        if self.hilbert:
            # The integrator only returns the spectral function and a Hilbert
            # transform is performed to return the real part of the density
            # response function.
            with self.timer('Hilbert transform'):
                # Make Hilbert transform
                ht = HilbertTransform(np.array(self.wd.omega_w), self.eta,
                                      timeordered=self.timeordered)
                ht(A_wxx)
                if wings:
                    ht(chi0_wxvx)

        # In the optical limit additional work must be performed
        # for the intraband response.
        # Only compute the intraband response if there are partially
        # unoccupied bands and only if the user has not disabled its
        # calculation using the include_intraband keyword.
        if optical_limit and self.nocc1 != self.nocc2:
            # The intraband response is essentially just the calculation
            # of the free space Drude plasma frequency. The calculation is
            # similarly to the interband transitions documented above.
            mat_kwargs = {'kd': kd, 'symmetry': PWSA,
                          'n1': self.nocc1, 'n2': self.nocc2,
                          'pd': pd}  # Integrand arguments
            eig_kwargs = {'kd': kd,
                          'n1': self.nocc1, 'n2': self.nocc2,
                          'pd': pd}  # Integrand arguments
            domain = (bzk_kv, spins)  # Integration domain
            fermi_level = self.pair.fermi_level  # Fermi level

            # Not so elegant solution but it works
            plasmafreq_wvv = np.zeros((1, 3, 3), complex)  # Output array
            print('Integrating intraband density response.', file=self.fd)

            # Depending on which integration method is used we
            # have to pass different arguments
            extraargs = {}
            if self.integrationmode is None:
                # Calculate intraband transitions at finite fermi smearing
                extraargs['intraband'] = True  # Calculate intraband
            elif self.integrationmode == 'tetrahedron integration':
                # Calculate intraband transitions at T=0
                extraargs['x'] = LinearFrequencyDescriptor([-fermi_level])

            intnoblock.integrate(kind='spectral function',  # Kind of integral
                                 domain=domain,  # Integration domain
                                 # Integrands
                                 integrand=(self.get_intraband_response,
                                            self.get_intraband_eigenvalue),
                                 # Integrand arguments
                                 kwargs=(mat_kwargs, eig_kwargs),
                                 out_wxx=plasmafreq_wvv,  # Output array
                                 **extraargs)  # Extra args for int. method

            # Again, not so pretty but that's how it is
            plasmafreq_vv = plasmafreq_wvv[0].copy()
            if self.include_intraband:

                if chi0.extend_head:
                    va = min(chi0.blockdist.blocks1d.a, 3)
                    vb = min(chi0.blockdist.blocks1d.b, 3)
                    A_wxx[:, :vb - va, :3] += (plasmafreq_vv[va:vb] /
                                               (self.wd.omega_w[:, np.newaxis,
                                                                np.newaxis] +
                                                1e-10 + self.rate * 1j)**2)
                elif self.blockcomm.rank == 0:
                    A_wxx[:, 0, 0] += (plasmafreq_vv[2, 2] /
                                       (self.wd.omega_w + 1e-10 +
                                        self.rate * 1j)**2)

            # Save the plasmafrequency
            try:
                self.plasmafreq_vv += 4 * np.pi * plasmafreq_vv * prefactor
            except AttributeError:
                self.plasmafreq_vv = 4 * np.pi * plasmafreq_vv * prefactor

            PWSA.symmetrize_wvv(self.plasmafreq_vv[np.newaxis])
            print('Plasma frequency:', file=self.fd)
            print((self.plasmafreq_vv**0.5 * Ha).round(2),
                  file=self.fd)

        # The response function is integrated only over the IBZ. The
        # chi calculated above must therefore be extended to include the
        # response from the full BZ. This extension can be performed as a
        # simple post processing of the response function that makes
        # sure that the response function fulfills the symmetries of the little
        # group of q. Due to the specific details of the implementation the chi
        # calculated above is normalized by the number of symmetries (as seen
        # below) and then symmetrized.
        A_wxx *= prefactor

        tmpA_wxx = chi0.blockdist.redistribute(A_wxx)
        if chi0.extend_head:
            PWSA.symmetrize_wxx(tmpA_wxx,
                                optical_limit=optical_limit)
        else:
            PWSA.symmetrize_wGG(tmpA_wxx)
            if wings:
                chi0.chi0_wxvG += chi0_wxvx[..., 2:]
                chi0.chi0_wvv += chi0_wxvx[:, 0, :3, :3]
                PWSA.symmetrize_wxvG(chi0.chi0_wxvG)
                PWSA.symmetrize_wvv(chi0.chi0_wvv)
        chi0.blockdist.redistribute(tmpA_wxx, A_wxx)

        # If point summation was used then the normalization of the
        # response function is not right and we have to make up for this
        # fact.

        if wings:
            chi0.chi0_wxvG *= prefactor
            chi0.chi0_wvv *= prefactor

        # In the optical limit, we have extended the wings and the head to
        # account for their nonanalytic behaviour which means that the size of
        # the chi0_wGG matrix is nw * (nG + 2)**2. Below we extract these
        # parameters.
        if optical_limit and chi0.extend_head:
            # We always return chi0 in the extend_head=False format. This
            # makes the update terminology inaccurate as we have to make a
            # new Chi0Data instance. In the future, extend_head=True should
            # be confined inside Chi0.update_chi0, so that the update
            # terminology is self-consistent
            chi0_new = self.create_chi0(pd.kd.bzk_kc[0], extend_head=False)
            # Make a wings object, but extended
            chi0_wxvG = np.zeros((len(chi0.wd), 2, 3,
                                  chi0.blockdist.blocks1d.N), complex)
            # The wings are extracted
            chi0_wxvG[:, 1, :,
                      chi0.blockdist.blocks1d.myslice] = np.transpose(
                A_wxx[..., 0:3], (0, 2, 1))
            va = min(chi0.blockdist.blocks1d.a, 3)
            vb = min(chi0.blockdist.blocks1d.b, 3)
            # print(self.world.rank, va, vb, chi0_wxvG[:, 0, va:vb].shape,
            #       A_wxx[:, va:vb].shape, A_wxx.shape)
            chi0_wxvG[:, 0, va:vb] = A_wxx[:, :vb - va]

            # Add contributions from different ranks
            self.blockcomm.sum(chi0_wxvG)
            # Insert values into the new Chi0Data object
            chi0_new.chi0_wvv[:] = chi0_wxvG[:, 0, :3, :3]
            chi0_new.chi0_wxvG = chi0_wxvG[..., 2:]
            # Jesus, this is complicated

            # It is easiest to redistribute over freqs to pick body
            tmpA_wxx = chi0.blockdist.redistribute(A_wxx)
            chi0_wGG = tmpA_wxx[:, 2:, 2:]
            chi0_new.chi0_wGG = chi0.blockdist.redistribute(chi0_wGG)

            # Rename
            chi0 = chi0_new

        elif optical_limit:
            # Since chi_wGG is nonanalytic in the head
            # and wings we have to take care that
            # these are handled correctly. Note that
            # it is important that the wings are overwritten first.
            chi0.chi0_wGG[:, :, 0] = chi0.chi0_wxvG[
                :, 1, 2,
                chi0.blockdist.blocks1d.myslice]

            if self.blockcomm.rank == 0:
                chi0.chi0_wGG[:, 0, :] = chi0.chi0_wxvG[:, 0, 2, :]
                chi0.chi0_wGG[:, 0, 0] = chi0.chi0_wvv[:, 2, 2]
        else:
            pass  # chi0_wGG = A_wxx

        return chi0

    def get_PWDescriptor(self, q_c, gammacentered=False):
        """Get the planewave descriptor of q_c."""
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, self.calc.wfs.gd,
                          complex, qd, gammacentered=gammacentered)
        return pd

    @timer('Get kpoints')
    def get_kpoints(self, pd, integrationmode=None):
        """Get the integration domain."""
        # Use symmetries
        PWSA = PWSymmetryAnalyzer
        PWSA = PWSA(self.calc.wfs.kd, pd,
                    timer=self.timer, txt=self.fd,
                    disable_point_group=self.disable_point_group,
                    disable_time_reversal=self.disable_time_reversal,
                    disable_non_symmorphic=self.disable_non_symmorphic)

        if integrationmode is None:
            K_gK = PWSA.group_kpoints()
            bzk_kc = np.array([self.calc.wfs.kd.bzk_kc[K_K[0]] for
                               K_K in K_gK])
        elif integrationmode == 'tetrahedron integration':
            bzk_kc = PWSA.get_reduced_kd(pbc_c=self.pbc).bzk_kc
            if (~self.pbc).any():
                bzk_kc = np.append(bzk_kc,
                                   bzk_kc + (~self.pbc).astype(int),
                                   axis=0)

        bzk_kv = np.dot(bzk_kc, pd.gd.icell_cv) * 2 * np.pi

        return bzk_kv, PWSA

    @timer('Get matrix element')
    def get_matrix_element(self, k_v, s, n1=None, n2=None,
                           m1=None, m2=None,
                           pd=None, kd=None,
                           symmetry=None, integrationmode=None,
                           extend_head=True, block=True):
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
            PWSA object for handling symmetries of the kpoints.
        integrationmode : str
            The integration mode employed.
        extend_head: Bool
            Extend the head to include non-analytic behaviour

        Return
        ------
        n_nmG : ndarray
            Pair densities.
        """
        assert m1 <= m2

        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)

        q_c = pd.kd.bzk_kc[0]

        optical_limit = np.allclose(q_c, 0.0)

        extrapolate_q = False
        if self.calc.wfs.kd.refine_info is not None:
            K1 = self.pair.find_kpoint(k_c)
            label = kd.refine_info.label_k[K1]
            if label == 'zero':
                return None
            elif (kd.refine_info.almostoptical and label == 'mh'):
                if not hasattr(self, 'pd0'):
                    self.pd0 = self.get_PWDescriptor([0, ] * 3)
                pd = self.pd0
                extrapolate_q = True

        nG = pd.ngmax
        weight = np.sqrt(symmetry.get_kpoint_weight(k_c) /
                         symmetry.how_many_symmetries())
        if self.Q_aGii is None:
            self.Q_aGii = self.pair.initialize_paw_corrections(pd)

        kptpair = self.pair.get_kpoint_pair(pd, s, k_c, n1, n2,
                                            m1, m2, block=block)

        m_m = np.arange(m1, m2)
        n_n = np.arange(n1, n2)

        n_nmG = self.pair.get_pair_density(pd, kptpair, n_n, m_m,
                                           Q_aGii=self.Q_aGii, block=block)

        if integrationmode is None:
            n_nmG *= weight

        df_nm = kptpair.get_occupation_differences(n_n, m_m)
        df_nm[df_nm <= 1e-20] = 0.0
        n_nmG *= df_nm[..., np.newaxis]**0.5

        if extrapolate_q:
            q_v = np.dot(q_c, pd.gd.icell_cv) * (2 * np.pi)
            nq_nm = np.dot(n_nmG[:, :, :3], q_v)
            n_nmG = n_nmG[:, :, 2:]
            n_nmG[:, :, 0] = nq_nm

        if not extend_head and optical_limit:
            n_nmG = np.copy(n_nmG[:, :, 2:])
            optical_limit = False

        if extend_head and optical_limit:
            return n_nmG.reshape(-1, nG + 2 * optical_limit)
        else:
            return n_nmG.reshape(-1, nG)

    @timer('Get eigenvalues')
    def get_eigenvalues(self, k_v, s, n1=None, n2=None,
                        m1=None, m2=None,
                        kd=None, pd=None, wfs=None,
                        filter=False):
        """A function that can return the eigenvalues.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        if wfs is None:
            wfs = self.calc.wfs

        kd = wfs.kd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        q_c = pd.kd.bzk_kc[0]
        K1 = self.pair.find_kpoint(k_c)
        K2 = self.pair.find_kpoint(k_c + q_c)

        ik1 = kd.bz2ibz_k[K1]
        ik2 = kd.bz2ibz_k[K2]
        kpt1 = wfs.kpt_qs[ik1][s]
        assert wfs.kd.comm.size == 1
        kpt2 = wfs.kpt_qs[ik2][s]
        deps_nm = np.subtract(kpt1.eps_n[n1:n2][:, np.newaxis],
                              kpt2.eps_n[m1:m2])

        if filter:
            fermi_level = self.pair.fermi_level
            deps_nm[kpt1.eps_n[n1:n2] > fermi_level, :] = np.nan
            deps_nm[:, kpt2.eps_n[m1:m2] < fermi_level] = np.nan

        return deps_nm.reshape(-1)

    def get_intraband_response(self, k_v, s, n1=None, n2=None,
                               kd=None, symmetry=None, pd=None,
                               integrationmode=None):
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        kpt1 = self.pair.get_k_point(s, k_c, n1, n2)
        n_n = range(n1, n2)

        vel_nv = self.pair.intraband_pair_density(kpt1, n_n)

        if self.integrationmode is None:
            f_n = kpt1.f_n
            assert self.calc.wfs.occupations.name in ['fermi-dirac',
                                                      'zero-width']
            width = getattr(self.calc.wfs.occupations, '_width', 0.0) / Ha
            if width > 1e-15:
                dfde_n = - 1. / width * (f_n - f_n**2.0)
            else:
                dfde_n = np.zeros_like(f_n)
            vel_nv *= np.sqrt(-dfde_n[:, np.newaxis])
            weight = np.sqrt(symmetry.get_kpoint_weight(k_c) /
                             symmetry.how_many_symmetries())
            vel_nv *= weight

        return vel_nv

    @timer('Intraband eigenvalue')
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
        K1 = self.pair.find_kpoint(k_c)
        ik = kd.bz2ibz_k[K1]
        kpt1 = wfs.kpt_qs[ik][s]
        assert wfs.kd.comm.size == 1

        return kpt1.eps_n[n1:n2]

    @timer('redist')
    def xxxredistribute(self, in_wGG, out_x=None):
        return self.blockdist.redistribute(in_wGG, out_x)

    @timer('dist freq')
    def xxxdistribute_frequencies(self, chi0_wGG):
        return self.blockdist.distribute_frequencies(chi0_wGG)

    def print_chi(self, pd):
        calc = self.calc
        gd = calc.wfs.gd

        if gpaw.dry_run:
            from gpaw.mpi import SerialCommunicator
            size = gpaw.dry_run
            world = SerialCommunicator()
            world.size = size
        else:
            world = self.world

        q_c = pd.kd.bzk_kc[0]
        nw = len(self.wd)
        ecut = self.ecut * Ha
        ns = calc.wfs.nspins
        nbands = self.nbands
        nk = calc.wfs.kd.nbzkpts
        nik = calc.wfs.kd.nibzkpts
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

        p = partial(print, file=self.fd)

        p('%s' % ctime())
        p('Called response.chi0.calculate with')
        p('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]))
        p('    Number of frequency points: %d' % nw)
        if bsize > nw:
            p('WARNING! Your nblocks is larger than number of frequency'
              ' points. Errors might occur, if your submodule don'''
              't know how to handle this.')
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
        p()


class HilbertTransform:

    def __init__(self, omega_w, eta, timeordered=False, gw=False,
                 blocksize=500):
        """Analytic Hilbert transformation using linear interpolation.

        Hilbert transform::

           oo
          /           1                1
          |dw' (-------------- - --------------) S(w').
          /     w - w' + i eta   w + w' + i eta
          0

        With timeordered=True, you get::

           oo
          /           1                1
          |dw' (-------------- - --------------) S(w').
          /     w - w' - i eta   w + w' + i eta
          0

        With gw=True, you get::

           oo
          /           1                1
          |dw' (-------------- + --------------) S(w').
          /     w - w' + i eta   w + w' + i eta
          0

        """

        self.blocksize = blocksize

        if timeordered:
            self.H_ww = self.H(omega_w, -eta) + self.H(omega_w, -eta, -1)
        elif gw:
            self.H_ww = self.H(omega_w, eta) - self.H(omega_w, -eta, -1)
        else:
            self.H_ww = self.H(omega_w, eta) + self.H(omega_w, -eta, -1)

    def H(self, o_w, eta, sign=1):
        """Calculate transformation matrix.

        With s=sign (+1 or -1)::

                        oo
                       /       dw'
          X (w, eta) = | ---------------- S(w').
           s           / s w - w' + i eta
                       0

        Returns H_ij so that X_i = np.dot(H_ij, S_j), where::

            X_i = X (omega_w[i]) and S_j = S(omega_w[j])
                   s
        """

        nw = len(o_w)
        H_ij = np.zeros((nw, nw), complex)
        do_j = o_w[1:] - o_w[:-1]
        for i, o in enumerate(o_w):
            d_j = o_w - o * sign
            y_j = 1j * np.arctan(d_j / eta) + 0.5 * np.log(d_j**2 + eta**2)
            y_j = (y_j[1:] - y_j[:-1]) / do_j
            H_ij[i, :-1] = 1 - (d_j[1:] - 1j * eta) * y_j
            H_ij[i, 1:] -= 1 - (d_j[:-1] - 1j * eta) * y_j
        return H_ij

    def __call__(self, S_wx):
        """Inplace transform"""
        B_wx = S_wx.reshape((len(S_wx), -1))
        nw, nx = B_wx.shape
        tmp_wx = np.zeros((nw, min(nx, self.blocksize)), complex)
        for x in range(0, nx, self.blocksize):
            b_wx = B_wx[:, x:x + self.blocksize]
            c_wx = tmp_wx[:, :b_wx.shape[1]]
            gemm(1.0, b_wx, self.H_ww, 0.0, c_wx)
            b_wx[:] = c_wx
