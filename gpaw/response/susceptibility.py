from __future__ import annotations

import numpy as np

from ase.units import Hartree

from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.pair_functions import (SingleQPWDescriptor, Chi,
                                          get_pw_coordinates)
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.coulomb_kernels import get_coulomb_kernel
from gpaw.response.fxc_kernels import FXCKernel, AdiabaticFXCCalculator
from gpaw.response.dyson import DysonSolver, HXCKernel


class ChiFactory:
    r"""User interface to calculate individual elements of the four-component
    susceptibility tensor χ^μν, see [PRB 103, 245110 (2021)]."""

    def __init__(self,
                 chiks_calc: ChiKSCalculator,
                 fxc_calculator: AdiabaticFXCCalculator | None = None):
        """Contruct a many-body susceptibility factory."""
        self.chiks_calc = chiks_calc
        self.gs = chiks_calc.gs
        self.context = chiks_calc.context
        self.dyson_solver = DysonSolver(self.context)

        # If no fxc_calculator is supplied, fall back to default
        if fxc_calculator is None:
            fxc_calculator = AdiabaticFXCCalculator.from_rshe_parameters(
                self.gs, self.context)
        else:
            assert fxc_calculator.gs is chiks_calc.gs
            assert fxc_calculator.context is chiks_calc.context
        self.fxc_calculator = fxc_calculator

        # Prepare a buffer for the fxc kernels
        self.fxc_kernel_cache: dict[str, FXCKernel] = {}

    def __call__(self, spincomponent, q_c, complex_frequencies,
                 fxc=None, hxc_scaling=None, txt=None) -> tuple[Chi, Chi]:
        r"""Calculate a given element (spincomponent) of the four-component
        Kohn-Sham susceptibility tensor and construct a corresponding many-body
        susceptibility object within a given approximation to the
        exchange-correlation kernel.

        Parameters
        ----------
        spincomponent : str
            Spin component (μν) of the susceptibility.
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
        q_c : list or ndarray
            Wave vector
        complex_frequencies : np.array or ComplexFrequencyDescriptor
            Array of complex frequencies to evaluate the response function at
            or a descriptor of those frequencies.
        fxc : str (None defaults to ALDA)
            Approximation to the (local) xc kernel.
            Choices: RPA, ALDA, ALDA_X, ALDA_x
        hxc_scaling : None or HXCScaling
            Supply an HXCScaling object to scale the hxc kernel.
        txt : str
            Save output of the calculation of this specific component into
            a file with the filename of the given input.
        """
        # Fall back to ALDA per default
        if fxc is None:
            fxc = 'ALDA'

        # Initiate new output file, if supplied
        if txt is not None:
            self.context.new_txt_and_timer(txt)

        # Print to output file
        self.context.print('---------------', flush=False)
        self.context.print('Calculating susceptibility spincomponent='
                           f'{spincomponent} with q_c={q_c}', flush=False)
        self.context.print('---------------')

        # Calculate chiks
        chiks = self.calculate_chiks(spincomponent, q_c, complex_frequencies)

        # Construct the hxc kernel
        hartree_kernel = self.get_hartree_kernel(spincomponent, chiks.qpd)
        xc_kernel = self.get_xc_kernel(fxc, spincomponent, chiks.qpd)
        hxc_kernel = HXCKernel(hartree_kernel, xc_kernel, scaling=hxc_scaling)

        # Solve dyson equation
        chi = self.dyson_solver(chiks, hxc_kernel)

        return chiks, chi

    def get_hartree_kernel(self, spincomponent, qpd):
        if spincomponent in ['+-', '-+']:
            # No Hartree term in Dyson equation
            return None
        else:
            return get_coulomb_kernel(qpd, self.gs.kd.N_c)

    def get_xc_kernel(self,
                      fxc: str,
                      spincomponent: str,
                      qpd: SingleQPWDescriptor):
        """Get the requested xc-kernel object."""
        if fxc == 'RPA':
            # No xc-kernel
            return None

        if qpd.gammacentered:
            # When using a gamma-centered plane-wave basis, we can reuse the
            # fxc kernel for all q-vectors. Thus, we keep a cache of calculated
            # kernels
            key = f'{fxc},{spincomponent}'
            if key not in self.fxc_kernel_cache:
                self.fxc_kernel_cache[key] = self.fxc_calculator(
                    fxc, spincomponent, qpd)
            fxc_kernel = self.fxc_kernel_cache[key]
        else:
            # Always compute the kernel
            fxc_kernel = self.fxc_calculator(fxc, spincomponent, qpd)

        return fxc_kernel

    def calculate_chiks(self, spincomponent, q_c, complex_frequencies):
        """Calculate the Kohn-Sham susceptibility."""
        q_c = np.asarray(q_c)
        if isinstance(complex_frequencies, ComplexFrequencyDescriptor):
            zd = complex_frequencies
        else:
            zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

        # Perform actual calculation
        chiks = self.chiks_calc.calculate(spincomponent, q_c, zd)
        # Distribute frequencies over world
        chiks = chiks.copy_with_global_frequency_distribution()

        return chiks


def spectral_decomposition(chi, pos_eigs=1, neg_eigs=0):
    """Decompose the susceptibility in terms of spectral functions.

    The full spectrum of induced excitations,

                        1
    S_GG'^(μν)(q,ω) = - ‾ χ_GG'^(μν")(q,ω)
                        π

    is extracted and separated into contributions corresponding to the pos_eigs
    and neg_eigs largest positive and negative eigenvalues respectively.
    """
    assert chi.distribution == 'zGG'
    # Extract the spectrum of induced excitations
    chid = chi.copy_dissipative_part()
    omega_w = chid.zd.omega_w * Hartree  # frequency grid in eV
    G_Gc = get_pw_coordinates(chid.qpd)  # plane-wave basis
    chid_wGG = chid.blocks1d.all_gather(chid.array)  # collect all frequencies
    S_wGG = - chid_wGG / np.pi

    # Initiate an EigendecomposedSpectrum object with the full spectrum
    full_spectrum = EigendecomposedSpectrum.from_spectrum(omega_w, G_Gc, S_wGG)

    # Separate the positive and negative eigenvalues for each frequency
    Apos = full_spectrum.get_positive_eigenvalue_spectrum()
    Aneg = full_spectrum.get_negative_eigenvalue_spectrum()

    # Keep only a fixed number of eigenvalues
    Apos = Apos.reduce_number_of_eigenvalues(pos_eigs)
    Aneg = Aneg.reduce_number_of_eigenvalues(neg_eigs)

    return Apos, Aneg


class EigendecomposedSpectrum:
    """Data object for eigendecomposed susceptibility spectra."""

    def __init__(self, omega_w, G_Gc, s_we, v_wGe, vinv_weG, A_w=None):
        """Construct the EigendecomposedSpectrum.

        Parameters
        ----------
        omega_w : np.array
            Frequencies in eV
        G_Gc : np.array
            Reciprocal lattice vectors in relative coordinates
        s_we : np.array
            Sorted eigenvalues (in decreasing order) at all frequencies.
            Here, e is the eigenvalue index.
        v_wGe : np.array
            Eigenvectors for corresponding to the (sorted) eigenvalues. With
            all eigenvalues present in the representation, v_Ge is the full
            eigenvector matrix V.
        vinv_weG : np.array
            Rows of the inverted eigenvector matrix V^-1 corresponding to the
            (sorted) eigenvalues.
        A_w : np.array or None
            Full spectral weight as a function of frequency. If given as None,
            A_w will be calculated as the sum of all eigenvalues (equal to the
            trace of the spectrum, if no eigenvalues have been discarded).
        """
        self.omega_w = omega_w
        self.G_Gc = G_Gc

        self.s_we = s_we
        self.v_wGe = v_wGe
        self.vinv_weG = vinv_weG

        self._A_w = A_w

    @classmethod
    def from_spectrum(cls, omega_w, G_Gc, S_wGG):
        """Perform an eigenvalue decomposition of a given spectrum."""
        # Find eigenvalues and eigenvectors of the spectrum
        s_wK, v_wGK = np.linalg.eigh(S_wGG)
        vinv_wKG = np.linalg.inv(v_wGK)

        # Sort by spectral intensity (eigenvalues in descending order)
        sorted_indices_wK = np.argsort(-s_wK)
        s_we = np.take_along_axis(s_wK, sorted_indices_wK, axis=1)
        v_wGe = np.take_along_axis(
            v_wGK, sorted_indices_wK[:, np.newaxis, :], axis=2)
        vinv_weG = np.take_along_axis(
            vinv_wKG, sorted_indices_wK[..., np.newaxis], axis=1)

        return cls(omega_w, G_Gc, s_we, v_wGe, vinv_weG)

    @classmethod
    def from_file(cls, filename):
        """Construct the eigendecomposed spectrum from a .pckl file."""
        import pickle
        assert isinstance(filename, str) and filename[-5:] == '.pckl'
        with open(filename, 'rb') as fd:
            omega_w, G_Gc, s_we, v_wGe, vinv_weG, A_w = pickle.load(fd)
        return cls(omega_w, G_Gc, s_we, v_wGe, vinv_weG, A_w=A_w)

    def write(self, filename):
        """Write the eigendecomposed spectrum as a .pckl file."""
        import pickle
        assert isinstance(filename, str) and filename[-5:] == '.pckl'
        with open(filename, 'wb') as fd:
            pickle.dump((self.omega_w, self.G_Gc,
                         self.s_we, self.v_wGe, self.vinv_weG,
                         self._A_w), fd)

    @property
    def nw(self):
        return len(self.omega_w)

    @property
    def nG(self):
        return self.v_wGe.shape[1]

    @property
    def neigs(self):
        return self.s_we.shape[1]

    @property
    def A_w(self):
        if self._A_w is None:
            self._A_w = np.nansum(self.s_we, axis=1)
        return self._A_w

    @property
    def A_wGG(self):
        """Generate the spectrum from the eigenvalues and eigenvectors."""
        A_wGG = []
        for s_e, v_Ge, vinv_eG in zip(self.s_we, self.v_wGe, self.vinv_weG):
            emask = ~np.isnan(s_e)
            A_wGG.append(v_Ge[:, emask] @ np.diag(s_e[emask]) @ vinv_eG[emask])
        return np.array(A_wGG)

    def get_positive_eigenvalue_spectrum(self):
        """Create a new EigendecomposedSpectrum from the positive eigenvalues.

        This is especially useful in order to separate the full spectrum of
        induced excitations, see [PRB 103, 245110 (2021)],

        S_GG'^μν(q,ω) = A_GG'^μν(q,ω) - A_(-G'-G)^νμ(-q,-ω)

        into the ν and μ components of the spectrum. Since the spectral
        function A_GG'^μν(q,ω) is positive definite or zero (in regions without
        excitations), A_GG'^μν(q,ω) simply corresponds to the positive
        eigenvalue contribution to the full spectrum S_GG'^μν(q,ω).
        """
        # Find the maximum number of positive eigenvalues across the entire
        # frequency range
        pos_we = self.s_we > 0.
        npos_max = np.max(np.sum(pos_we, axis=1))

        # Allocate new arrays filled with nan to accomodate all the positive
        # eigenvalues
        s_we = np.empty((self.nw, npos_max),
                        dtype=self.s_we.dtype)
        v_wGe = np.empty((self.nw, self.nG, npos_max),
                         dtype=self.v_wGe.dtype)
        vinv_weG = np.empty((self.nw, npos_max, self.nG),
                            dtype=self.vinv_weG.dtype)
        s_we[:] = np.nan
        v_wGe[:] = np.nan
        vinv_weG[:] = np.nan

        # Fill arrays with the positive eigenvalue data
        for w, (s_e, v_Ge, vinv_eG) in enumerate(zip(
                self.s_we, self.v_wGe, self.vinv_weG)):
            pos_e = s_e > 0.
            npos = np.sum(pos_e)
            s_we[w, :npos] = s_e[pos_e]
            v_wGe[w, :, :npos] = v_Ge[:, pos_e]
            vinv_weG[w, :npos] = vinv_eG[pos_e]

        return EigendecomposedSpectrum(self.omega_w, self.G_Gc,
                                       s_we, v_wGe, vinv_weG)

    def get_negative_eigenvalue_spectrum(self):
        """Create a new EigendecomposedSpectrum from the negative eigenvalues.

        The spectrum is created by reversing and negating the spectrum,

        -S_GG'^μν(q,-ω) = -A_GG'^μν(q,-ω) + A_(-G'-G)^νμ(-q,ω),

        from which the spectral function A_GG'^νμ(q,ω) can be extracted as the
        positive eigenvalue contribution, thanks to the reciprocity relation

                                  ˍˍ
        χ_GG'^μν(q,ω) = χ_(-G'-G)^νμ(-q,ω),
                   ˍ
        in which n^μ(r) denotes the hermitian conjugate [n^μ(r)]^†, and which
        is valid for μν ∊ {00,0z,zz,+-} in collinear systems without spin-orbit
        coupling.
        """
        # Flip and negate the spectral function
        omega_w = - self.omega_w[::-1]
        s_we = - self.s_we[::-1, ::-1]
        v_wGe = self.v_wGe[::-1, :, ::-1]
        vinv_weG = self.vinv_weG[::-1, ::-1]
        inverted_spectrum = EigendecomposedSpectrum(omega_w, self.G_Gc,
                                                    s_we, v_wGe, vinv_weG)

        return inverted_spectrum.get_positive_eigenvalue_spectrum()

    def reduce_number_of_eigenvalues(self, neigs):
        """Create a new spectrum with only the neigs largest eigenvalues.

        The returned EigendecomposedSpectrum is constructed to retain knowledge
        of the full spectral weight of the unreduced spectrum through the A_w
        attribute.
        """
        assert self.neigs >= neigs
        # Check that the available eigenvalues are in descending order
        assert all([np.all(np.logical_not(s_e[1:] - s_e[:-1] > 0.))
                    for s_e in self.s_we]),\
            'Eigenvalues needs to be sorted in descending order!'

        # Keep only the neigs largest eigenvalues
        s_we = self.s_we[:, :neigs]
        v_wGe = self.v_wGe[..., :neigs]
        vinv_weG = self.vinv_weG[:, :neigs]

        return EigendecomposedSpectrum(self.omega_w, self.G_Gc,
                                       s_we, v_wGe, vinv_weG,
                                       # Keep the full spectral weight
                                       A_w=self.A_w)
