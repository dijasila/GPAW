import pickle

import numpy as np

from ase.units import Hartree

from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKS, ChiKSCalculator
from gpaw.response.coulomb_kernels import get_coulomb_kernel
from gpaw.response.fxc_kernels import FXCFactory
from gpaw.response.dyson import DysonSolver


class Chi:
    """Many-body susceptibility in a plane-wave basis."""

    def __init__(self, chiks: ChiKS,
                 Vbare_G, Kxc_GG,
                 dyson_solver: DysonSolver):
        """Construct the many-body susceptibility based on its ingredients."""
        assert chiks.distribution == 'zGG' and\
            chiks.blockdist.blockcomm.size == chiks.blockdist.world.size,\
            "Chi assumes that chiks's frequencies are distributed over world"
        self.chiks = chiks
        self.world = chiks.blockdist.world

        self.Vbare_G = Vbare_G
        self.Kxc_GG = Kxc_GG  # Use Kxc_G in the future XXX

        self.dyson_solver = dyson_solver

    def write_macroscopic_component(self, filename):
        """Calculate the spatially averaged (macroscopic) component of the
        susceptibility and write it to a file together with the Kohn-Sham
        susceptibility and the frequency grid."""
        from gpaw.response.df import write_response_function

        # For now, we assume that eta is fixed, so we don't need to write it
        omega_w = self.chiks.zd.omega_w * Hartree

        chiks_wGG = self.chiks.array
        chi_wGG = self._calculate()

        # Macroscopic component
        chiks_w = chiks_wGG[:, 0, 0]
        chi_w = chi_wGG[:, 0, 0]

        # Collect all frequencies from world
        chiks_w = self.collect(chiks_w)
        chi_w = self.collect(chi_w)

        if self.world.rank == 0:
            write_response_function(filename, omega_w, chiks_w, chi_w)

    def write_component_array(self, filename, *, reduced_ecut):
        """Calculate the many-body susceptibility and write it to a file along
        with the Kohn-Sham susceptibility and frequency grid within a reduced
        plane-wave basis."""

        # For now, we assume that eta is fixed, so we don't need to write it
        omega_w = self.chiks.zd.omega_w * Hartree

        chiks_wGG = self.chiks.array
        chi_wGG = self._calculate()

        # Get susceptibility in a reduced plane-wave representation
        pd = self.chiks.pd
        mask_G = get_pw_reduction_map(pd, reduced_ecut)
        chiks_wGG = np.ascontiguousarray(chiks_wGG[:, mask_G, :][:, :, mask_G])
        chi_wGG = np.ascontiguousarray(chi_wGG[:, mask_G, :][:, :, mask_G])

        # Get reduced plane wave repr. as coordinates on the reciprocal lattice
        G_Gc = get_pw_coordinates(pd)[mask_G]

        # Gather all frequencies from world to root
        chiks_wGG = self.gather(chiks_wGG)
        chi_wGG = self.gather(chi_wGG)

        if self.world.rank == 0:
            write_component(omega_w, G_Gc, chiks_wGG, chi_wGG, filename)

    def _calculate(self):
        """Calculate chi_zGG."""
        return self.dyson_solver.invert_dyson(self.chiks.array, self.Khxc_GG)

    @property
    def Khxc_GG(self):
        """Hartree-exchange-correlation kernel."""
        # Allocate array
        nG = self.chiks.array.shape[2]
        Khxc_GG = np.zeros((nG, nG), dtype=complex)

        if self.Vbare_G is not None:  # Add the Hartree kernel
            Khxc_GG.flat[::nG + 1] += self.Vbare_G
        if self.Kxc_GG is not None:  # Add the xc kernel
            # In the future, construct the xc kernel here! XXX
            Khxc_GG += self.Kxc_GG

        return Khxc_GG

    def collect(self, x_z):
        """Collect all frequencies."""
        return self.chiks.blocks1d.collect(x_z)

    def gather(self, X_zGG):
        """Gather a full susceptibility array to root."""
        # Allocate arrays to gather (all need to be the same shape)
        blocks1d = self.chiks.blocks1d
        shape = (blocks1d.blocksize,) + X_zGG.shape[1:]
        tmp_zGG = np.empty(shape, dtype=X_zGG.dtype)
        tmp_zGG[:blocks1d.nlocal] = X_zGG

        # Allocate array for the gathered data
        if self.world.rank == 0:
            # Make room for all frequencies
            Npadded = blocks1d.blocksize * blocks1d.blockcomm.size
            shape = (Npadded,) + X_zGG.shape[1:]
            allX_zGG = np.empty(shape, dtype=X_zGG.dtype)
        else:
            allX_zGG = None

        self.world.gather(tmp_zGG, 0, allX_zGG)

        # Return array for w indeces on frequency grid
        if allX_zGG is not None:
            allX_zGG = allX_zGG[:len(self.chiks.zd), :, :]

        return allX_zGG


class ChiFactory:
    r"""User interface to calculate individual elements of the four-component
    susceptibility tensor χ^μν, see [PRB 103, 245110 (2021)]."""

    def __init__(self, chiks_calc: ChiKSCalculator):
        """Contruct a many-body susceptibility factory."""
        self.chiks_calc = chiks_calc
        self.context = chiks_calc.context
        self.gs = chiks_calc.gs

        self.fxc_factory = FXCFactory(self.gs, self.context)

        # Prepare a buffer for chiks
        self._chiks = None

    def __call__(self, spincomponent, q_c, complex_frequencies,
                 fxc='ALDA', fxckwargs=None, txt=None) -> Chi:
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
        fxc : str
            Approximation to the xc kernel
        fxckwargs : dict
            Kwargs to the FXCCalculator
        txt : str
            Save output of the calculation of this specific component into
            a file with the filename of the given input.
        """
        assert isinstance(fxc, str)
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
            Vbare_G = get_coulomb_kernel(chiks.pd, self.gs.kd.N_c)

        # Calculate the exchange-correlation kernel
        if fxc == 'RPA':
            # No xc kernel by definition
            Kxc_GG = None
        else:
            Kxc_GG = self.fxc_factory(fxc, chiks, fxckwargs=fxckwargs)

        # Initiate the dyson solver
        dyson_solver = DysonSolver(self.context)

        return Chi(chiks, Vbare_G, Kxc_GG, dyson_solver)

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


def get_pw_reduction_map(pd, ecut):
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
    assert ecut <= pd.ecut

    # List of all plane waves
    G_Gv = np.array([pd.G_Qv[Q] for Q in pd.Q_qG[0]])

    if pd.gammacentered:
        mask_G = ((G_Gv ** 2).sum(axis=1) <= 2 * ecut)
    else:
        mask_G = (((G_Gv + pd.K_qv[0]) ** 2).sum(axis=1) <= 2 * ecut)

    return mask_G


def get_pw_coordinates(pd):
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
    G_Gv = np.array([pd.G_Qv[Q] for Q in pd.Q_qG[0]])

    # Use cell to get coordinates
    B_cv = 2.0 * np.pi * pd.gd.icell_cv
    return np.round(np.dot(G_Gv, np.linalg.inv(B_cv))).astype(int)


def get_inverted_pw_mapping(pd1, pd2):
    """Get the plane wave coefficients mapping GG' of pd1 into -G-G' of pd2"""
    G1_Gc = get_pw_coordinates(pd1)
    G2_Gc = get_pw_coordinates(pd2)

    mG2_G1 = []
    for G1_c in G1_Gc:
        found_match = False
        for G2, G2_c in enumerate(G2_Gc):
            if np.all(G2_c == -G1_c):
                mG2_G1.append(G2)
                found_match = True
                break
        if not found_match:
            raise ValueError('Could not match pd1 and pd2')

    # Set up mapping from GG' to -G-G'
    invmap_GG = tuple(np.meshgrid(mG2_G1, mG2_G1, indexing='ij'))

    return invmap_GG


def symmetrize_reciprocity(pd, X_wGG):
    """In collinear systems without spin-orbit coupling, the plane wave
    susceptibility is reciprocal in the sense that e.g.

    χ_(GG')^(+-)(q, ω) = χ_(-G'-G)^(+-)(-q, ω)

    This method symmetrizes A_wGG in the case where q=0.
    """
    from gpaw.test.response.test_chiks import get_inverted_pw_mapping

    q_c = pd.q_c
    if np.allclose(q_c, 0.):
        invmap_GG = get_inverted_pw_mapping(pd, pd)
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
