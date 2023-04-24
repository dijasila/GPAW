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

        # XXX To do XXX
        # * Stop writing results related to chiks (and remove self._chiks)
        # * Find a different way to expose the fxc_kernel
        # * Use Blocks1D to distinguish between gather and all_gather (collect)

        self.fxc_kernel = hxc_kernel.fxc_kernel
        self._chiks = chiks

    def write_macroscopic_component(self, filename):
        """Calculate the spatially averaged (macroscopic) component of the
        susceptibility and write it to a file along with the frequency grid."""
        from gpaw.response.pair_functions import write_pair_function
        chi_z = self.get_macroscopic_component()
        if self.world.rank == 0:
            write_pair_function(filename, self.zd, chi_z)

    def get_macroscopic_component(self):
        """Get the macroscopic (G=0) component data, collected on all ranks"""
        assert self.distribution == 'zGG'
        chi_zGG = self.array
        chi_z = chi_zGG[:, 0, 0]  # Macroscopic component
        chi_z = self.blocks1d.collect(chi_z)  # Collect distributed frequencies
        return chi_z

    def write_reduced_arrays(self, filename, *, reduced_ecut):
        """Calculate the many-body susceptibility and write it to a file along
        with the Kohn-Sham susceptibility and frequency grid within a reduced
        plane-wave basis."""
        omega_w, G_Gc, chiks_wGG, chi_wGG = self.get_reduced_arrays(
            reduced_ecut=reduced_ecut)

        if self.world.rank == 0:
            write_component(omega_w, G_Gc, chiks_wGG, chi_wGG, filename)

    def get_reduced_arrays(self, *, reduced_ecut):
        """Get data arrays with a reduced ecut, gathered on root."""
        omega_w, chiks_wGG, chi_wGG = self.get_distributed_arrays()

        # Map the susceptibilities to a reduced plane-wave representation
        qpd = self.qpd
        mask_G = get_pw_reduction_map(qpd, reduced_ecut)
        chiks_wGG = np.ascontiguousarray(chiks_wGG[:, mask_G, :][:, :, mask_G])
        chi_wGG = np.ascontiguousarray(chi_wGG[:, mask_G, :][:, :, mask_G])

        # Get reduced plane wave repr. as coordinates on the reciprocal lattice
        G_Gc = get_pw_coordinates(qpd)[mask_G]

        # Gather all frequencies from world to root
        chiks_wGG = self.gather(chiks_wGG)
        chi_wGG = self.gather(chi_wGG)

        return omega_w, G_Gc, chiks_wGG, chi_wGG

    def write_reduced_diagonals(self, filename, *, reduced_ecut):
        """Calculate the many-body susceptibility and write its diagonal in
        plane-wave components to a file along with the diagonal of the
        Kohn-Sham susceptibility, frequency grid and reduced plane-wave basis.
        """
        omega_w, G_Gc, chiks_wG, chi_wG = self.get_reduced_diagonals(
            reduced_ecut=reduced_ecut)

        if self.world.rank == 0:
            write_diagonal(omega_w, G_Gc, chiks_wG, chi_wG, filename)

    def get_reduced_diagonals(self, *, reduced_ecut):
        """Get the diagonal of the reduced data arrays, gathered on root."""
        omega_w, chiks_wGG, chi_wGG = self.get_distributed_arrays()

        # Map the susceptibilities to a reduced plane-wave representation
        qpd = self.qpd
        mask_G = get_pw_reduction_map(qpd, reduced_ecut)
        chiks_wG = np.ascontiguousarray(chiks_wGG[:, mask_G, mask_G])
        chi_wG = np.ascontiguousarray(chi_wGG[:, mask_G, mask_G])

        # Get reduced plane wave repr. as coordinates on the reciprocal lattice
        G_Gc = get_pw_coordinates(qpd)[mask_G]

        # Gather all frequencies from world to root
        chiks_wG = self.gather(chiks_wG)
        chi_wG = self.gather(chi_wG)

        return omega_w, G_Gc, chiks_wG, chi_wG

    def get_distributed_arrays(self):
        """Get data arrays, frequency distributed over world."""
        # For now, we assume that eta is fixed -> z index == w index
        omega_w = self.zd.omega_w * Hartree
        chiks_wGG = self._chiks.array
        chi_wGG = self.array

        return omega_w, chiks_wGG, chi_wGG

    def gather(self, X_zx):
        """Gather a full susceptibility array to root."""
        # Allocate arrays to gather (all need to be the same shape)
        blocks1d = self.blocks1d
        shape = (blocks1d.blocksize,) + X_zx.shape[1:]
        tmp_zx = np.empty(shape, dtype=X_zx.dtype)
        tmp_zx[:blocks1d.nlocal] = X_zx

        # Allocate array for the gathered data
        if self.world.rank == 0:
            # Make room for all frequencies
            Npadded = blocks1d.blocksize * blocks1d.blockcomm.size
            shape = (Npadded,) + X_zx.shape[1:]
            allX_zx = np.empty(shape, dtype=X_zx.dtype)
        else:
            allX_zx = None

        self.world.gather(tmp_zx, 0, allX_zx)

        # Return array for w indeces on frequency grid
        if allX_zx is not None:
            allX_zx = allX_zx[:len(self.zd)]

        return allX_zx


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
                 fxc_kernel=None, fxc=None, localft_calc=None,
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
            according to keywords fxc and localft_calc.
        fxc : str (None defaults to ALDA)
            Approximation to the (local) xc kernel.
            Choices: ALDA, ALDA_X, ALDA_x
        localft_calc : LocalFTCalculator or None
            Calculator used to Fourier transform the fxc kernel into plane-wave
            components. If None, the default LocalPAWFTCalculator is used.
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
                assert localft_calc is None,\
                    "With fxc='RPA', there is no xc kernel to be calculated,"\
                    "rendering the localft_calc input irrelevant"
            else:
                # If no localft_calc is supplied, fall back to the default
                if localft_calc is None:
                    localft_calc = LocalPAWFTCalculator(self.gs, self.context)

                # Perform an actual kernel calculation
                fxc_calculator = AdiabaticFXCCalculator(localft_calc)
                fxc_kernel = fxc_calculator(
                    fxc, chiks.spincomponent, chiks.qpd)
        else:
            assert fxc is None and localft_calc is None,\
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


def write_component(omega_w, G_Gc, chiks_wGG, chi_wGG, filename):
    """Write the dynamic susceptibility as a pickle file."""
    assert isinstance(filename, str)
    with open(filename, 'wb') as fd:
        pickle.dump((omega_w, G_Gc, chiks_wGG, chi_wGG), fd)


def read_component(filename):
    """Read a stored susceptibility component file"""
    assert isinstance(filename, str)
    with open(filename, 'rb') as fd:
        omega_w, G_Gc, chiks_wGG, chi_wGG = pickle.load(fd)

    return omega_w, G_Gc, chiks_wGG, chi_wGG


def write_diagonal(omega_w, G_Gc, chiks_wG, chi_wG, filename):
    """Write the diagonal of a dynamic susceptibility as a pickle file."""
    assert isinstance(filename, str)
    with open(filename, 'wb') as fd:
        pickle.dump((omega_w, G_Gc, chiks_wG, chi_wG), fd)


def read_diagonal(filename):
    """Read stored susceptibility diagonal file."""
    assert isinstance(filename, str)
    with open(filename, 'rb') as fd:
        omega_w, G_Gc, chiks_wG, chi_wG = pickle.load(fd)

    return omega_w, G_Gc, chiks_wG, chi_wG
