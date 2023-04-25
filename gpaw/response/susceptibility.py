import pickle

import numpy as np

from ase.units import Hartree

from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKS, ChiKSCalculator
from gpaw.response.coulomb_kernels import get_coulomb_kernel
from gpaw.response.localft import LocalPAWFTCalculator
from gpaw.response.fxc_kernels import AdiabaticFXCCalculator
from gpaw.response.dyson import DysonSolver, HXCKernel


class Chi:
    """Many-body susceptibility in a plane-wave basis."""

    def __init__(self,
                 chiks: ChiKS,
                 hxc_kernel: HXCKernel,
                 dyson_solver: DysonSolver):
        """Construct the many-body susceptibility based on its ingredients."""
        # Extract properties from chiks
        self.qpd = chiks.qpd
        self.zd = chiks.zd
        self.blockdist = chiks.blockdist
        self.distribution = chiks.distribution
        self.world = self.blockdist.world
        self.blocks1d = chiks.blocks1d

        # Solve dyson equation
        self.array = dyson_solver(chiks, hxc_kernel)

        # Store the fxc kernel for temporary backwards compatibility
        self.fxc_kernel = hxc_kernel.fxc_kernel

    def write_macroscopic_component(self, filename):
        """Write the spatially averaged (macroscopic) component of the
        susceptibility to a file along with the frequency grid."""
        from gpaw.response.pair_functions import write_pair_function
        chi_Z = self.get_macroscopic_component()
        if self.world.rank == 0:
            write_pair_function(filename, self.zd, chi_Z)

    def get_macroscopic_component(self):
        """Get the macroscopic (G=0) component, all-gathered."""
        assert self.distribution == 'zGG'
        chi_zGG = self.array
        chi_z = chi_zGG[:, 0, 0]  # Macroscopic component
        chi_Z = self.blocks1d.all_gather(chi_z)
        return chi_Z

    def write_reduced_array(self, filename, *, reduced_ecut):
        """Write the susceptibility within a reduced plane-wave basis to a file
        along with the frequency grid."""
        G_Gc, chi_ZGG = self.get_reduced_array(reduced_ecut=reduced_ecut)
        if self.world.rank == 0:
            write_susceptibility_array(filename, self.zd, G_Gc, chi_ZGG)

    def get_reduced_array(self, *, reduced_ecut):
        """Get data array with a reduced ecut, gathered on root."""
        assert self.distribution == 'zGG'
        chi_zGG = self.array

        # Map the susceptibilities to a reduced plane-wave representation
        qpd = self.qpd
        mask_G = get_pw_reduction_map(qpd, reduced_ecut)
        chi_zGG = np.ascontiguousarray(chi_zGG[:, mask_G, :][:, :, mask_G])
        chi_ZGG = self.blocks1d.gather(chi_zGG)

        # Get reduced plane wave repr. as coordinates on the reciprocal lattice
        G_Gc = get_pw_coordinates(qpd)[mask_G]

        return G_Gc, chi_ZGG

    def write_reduced_diagonal(self, filename, *, reduced_ecut):
        """Write the diagonal of the many-body susceptibility within a reduced
        plane-wave basis to a file along with the frequency grid."""
        G_Gc, chi_ZG = self.get_reduced_diagonal(reduced_ecut=reduced_ecut)
        if self.world.rank == 0:
            write_susceptibility_array(filename, self.zd, G_Gc, chi_ZG)

    def get_reduced_diagonal(self, *, reduced_ecut):
        """Get the diagonal of the reduced data array, gathered on root."""
        assert self.distribution == 'zGG'
        chi_zGG = self.array

        # Map the susceptibilities to a reduced plane-wave representation
        qpd = self.qpd
        mask_G = get_pw_reduction_map(qpd, reduced_ecut)
        chi_zG = np.ascontiguousarray(chi_zGG[:, mask_G, mask_G])
        chi_ZG = self.blocks1d.gather(chi_zG)

        # Get reduced plane wave repr. as coordinates on the reciprocal lattice
        G_Gc = get_pw_coordinates(qpd)[mask_G]

        return G_Gc, chi_ZG


class ChiFactory:
    r"""User interface to calculate individual elements of the four-component
    susceptibility tensor χ^μν, see [PRB 103, 245110 (2021)]."""

    def __init__(self, chiks_calc: ChiKSCalculator):
        """Contruct a many-body susceptibility factory."""
        self.chiks_calc = chiks_calc

        self.gs = chiks_calc.gs
        self.context = chiks_calc.context

        # Prepare a buffer for chiks
        self._chiks = None

    def __call__(self, spincomponent, q_c, complex_frequencies,
                 fxc_kernel=None, fxc=None, fxc_calculator=None,
                 hxc_scaling=None, txt=None) -> Chi:
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
        fxc_kernel : FXCKernel
            Exchange-correlation kernel (calculated elsewhere). Use this input
            carefully! The plane-wave representation in the supplied kernel has
            to match the representation of chiks.
            If no kernel is supplied, the ChiFactory will calculate one itself
            according to keywords fxc and fxc_calculator.
        fxc : str (None defaults to ALDA)
            Approximation to the (local) xc kernel.
            Choices: ALDA, ALDA_X, ALDA_x
        fxc_calculator : AdiabaticFXCCalculator or None
            Calculator for the xc kernel.
        hxc_scaling : None or HXCScaling
            Supply an HXCScaling object to scale the hxc kernel.
        txt : str
            Save output of the calculation of this specific component into
            a file with the filename of the given input.
        """
        # Initiate new output file, if supplied
        if txt is not None:
            self.context.new_txt_and_timer(txt)

        # Print to output file
        self.context.print('---------------', flush=False)
        self.context.print('Calculating susceptibility spincomponent='
                           f'{spincomponent} with q_c={q_c}', flush=False)
        self.context.print('---------------')

        # Calculate chiks (or get it from the buffer)
        chiks = self.get_chiks(spincomponent, q_c, complex_frequencies)

        # Calculate the Coulomb kernel
        if spincomponent in ['+-', '-+']:
            assert fxc != 'RPA'
            # No Hartree term in Dyson equation
            Vbare_G = None
        else:
            Vbare_G = get_coulomb_kernel(chiks.qpd, self.gs.kd.N_c)

        # Calculate the xc kernel, if it has not been supplied by the user
        if fxc_kernel is None:
            # Use ALDA as the default fxc
            if fxc is None:
                fxc = 'ALDA'
            # In RPA, we neglect the xc-kernel
            if fxc == 'RPA':
                assert fxc_calculator is None,\
                    "With fxc='RPA', there is no xc kernel to be calculated,"\
                    "rendering the fxc_calculator input irrelevant"
            else:
                # If no localft_calc is supplied, fall back to the default
                if fxc_calculator is None:
                    fxc_calculator = \
                        AdiabaticFXCCalculator.from_rshe_parameters(
                            self.gs, self.context)

                # Perform an actual kernel calculation
                fxc_kernel = fxc_calculator(
                    fxc, chiks.spincomponent, chiks.qpd)
        else:
            assert fxc is None and fxc_calculator is None,\
                'Supplying an xc kernel overwrites any specification of how'\
                'to calculate the kernel'

        # Construct the hxc kernel and dyson solver
        hxc_kernel = HXCKernel(Vbare_G, fxc_kernel, scaling=hxc_scaling)
        dyson_solver = DysonSolver(self.context)

        return Chi(chiks, hxc_kernel, dyson_solver)

    def get_chiks(self, spincomponent, q_c, complex_frequencies):
        """Get chiks from buffer."""
        q_c = np.asarray(q_c)
        if isinstance(complex_frequencies, ComplexFrequencyDescriptor):
            zd = complex_frequencies
        else:
            zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

        if self._chiks is None or\
            not (spincomponent == self._chiks.spincomponent and
                 np.allclose(q_c, self._chiks.q_c) and
                 zd.almost_eq(self._chiks.zd)):
            # Calculate new chiks, if buffer is empty or if we are
            # considering a new set of spincomponent, q-vector and frequencies
            chiks = self.chiks_calc.calculate(spincomponent, q_c, zd)
            # Distribute frequencies over world
            chiks = chiks.copy_with_global_frequency_distribution()

            # Fill buffer
            self._chiks = chiks

        return self._chiks


def get_pw_reduction_map(qpd, ecut):
    """Get a mask to reduce the plane wave representation.

    Please remark, that the response code currently works with one q-vector
    at a time, at thus only a single plane wave representation at a time.

    Returns
    -------
    mask_G : nd.array (dtype=bool)
        Mask which reduces the representation
    """
    assert ecut is not None
    ecut /= Hartree
    assert ecut <= qpd.ecut

    # List of all plane waves
    G_Gv = np.array([qpd.G_Qv[Q] for Q in qpd.Q_qG[0]])

    if qpd.gammacentered:
        mask_G = ((G_Gv ** 2).sum(axis=1) <= 2 * ecut)
    else:
        mask_G = (((G_Gv + qpd.K_qv[0]) ** 2).sum(axis=1) <= 2 * ecut)

    return mask_G


def get_pw_coordinates(qpd):
    """Get the reciprocal lattice vector coordinates corresponding to a
    givne plane wave basis.

    Please remark, that the response code currently works with one q-vector
    at a time, at thus only a single plane wave representation at a time.

    Returns
    -------
    G_Gc : nd.array (dtype=int)
        Coordinates on the reciprocal lattice
    """
    # List of all plane waves
    G_Gv = np.array([qpd.G_Qv[Q] for Q in qpd.Q_qG[0]])

    # Use cell to get coordinates
    B_cv = 2.0 * np.pi * qpd.gd.icell_cv
    return np.round(np.dot(G_Gv, np.linalg.inv(B_cv))).astype(int)


def get_inverted_pw_mapping(qpd1, qpd2):
    """Get the planewave coefficients mapping GG' of qpd1 into -G-G' of qpd2"""
    G1_Gc = get_pw_coordinates(qpd1)
    G2_Gc = get_pw_coordinates(qpd2)

    mG2_G1 = []
    for G1_c in G1_Gc:
        found_match = False
        for G2, G2_c in enumerate(G2_Gc):
            if np.all(G2_c == -G1_c):
                mG2_G1.append(G2)
                found_match = True
                break
        if not found_match:
            raise ValueError('Could not match qpd1 and qpd2')

    # Set up mapping from GG' to -G-G'
    invmap_GG = tuple(np.meshgrid(mG2_G1, mG2_G1, indexing='ij'))

    return invmap_GG


def symmetrize_reciprocity(qpd, X_wGG):
    """In collinear systems without spin-orbit coupling, the plane wave
    susceptibility is reciprocal in the sense that e.g.

    χ_(GG')^(+-)(q, ω) = χ_(-G'-G)^(+-)(-q, ω)

    This method symmetrizes A_wGG in the case where q=0.
    """
    from gpaw.test.response.test_chiks import get_inverted_pw_mapping

    q_c = qpd.q_c
    if np.allclose(q_c, 0.):
        invmap_GG = get_inverted_pw_mapping(qpd, qpd)
        for X_GG in X_wGG:
            # Symmetrize [χ_(GG')(q, ω) + χ_(-G'-G)(-q, ω)] / 2
            X_GG[:] = (X_GG + X_GG[invmap_GG].T) / 2.


def write_susceptibility_array(filename, zd, G_Gc, chi_zx):
    """Write the dynamic susceptibility as a pickle file."""
    # For now, we assume that the complex frequencies lie on a horizontal
    # contour
    assert zd.horizontal_contour
    omega_w = zd.omega_w * Hartree  # Ha -> eV
    chi_wx = chi_zx

    # Write pickle file
    with open(filename, 'wb') as fd:
        pickle.dump((omega_w, G_Gc, chi_wx), fd)


def read_susceptibility_array(filename):
    """Read a stored susceptibility component file"""
    assert isinstance(filename, str)
    with open(filename, 'rb') as fd:
        omega_w, G_Gc, chi_wx = pickle.load(fd)

    return omega_w, G_Gc, chi_wx
