import pickle

import numpy as np

from ase.units import Hartree

from gpaw.response import ResponseGroundStateAdapter, ResponseContext, timer
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.chiks import ChiKS
from gpaw.response.fxc_kernels import get_fxc
from gpaw.response.coulomb_kernels import get_coulomb_kernel
from gpaw.response.pw_parallelization import Blocks1D


class FourComponentSusceptibilityTensor:
    """Class calculating the full four-component susceptibility tensor"""

    def __init__(self, gs, context=None,
                 fxc='ALDA', fxckwargs={},
                 eta=0.2, ecut=50, gammacentered=False,
                 disable_point_group=False, disable_time_reversal=False,
                 bandsummation='pairwise', nbands=None,
                 bundle_integrals=True, nblocks=1):
        """
        Currently, everything is in plane wave mode.
        If additional modes are implemented, maybe look to fxc to see how
        multiple modes can be supported.

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        fxc, fxckwargs : see gpaw.response.fxc
        eta, ecut, gammacentered
        disable_point_group,
        disable_time_reversal,
        bandsummation, nbands, bundle_integrals,
        nblocks : see gpaw.response.chiks, gpaw.response.kslrf
        """
        assert isinstance(gs, ResponseGroundStateAdapter)
        self.gs = gs
        if context is None:
            self.context = ResponseContext()
        else:
            assert isinstance(context, ResponseContext)
            self.context = context

        # The plane wave basis is defined by keywords
        self.ecut = None if ecut is None else ecut / Hartree
        self.gammacentered = gammacentered

        # Initiate Kohn-Sham susceptibility and fxc objects
        self.chiks = ChiKS(self.gs, context=self.context, eta=eta, ecut=ecut,
                           gammacentered=gammacentered,
                           disable_point_group=disable_point_group,
                           disable_time_reversal=disable_time_reversal,
                           bandsummation=bandsummation, nbands=nbands,
                           bundle_integrals=bundle_integrals, nblocks=nblocks)
        self.fxc = get_fxc(gs, self.context, fxc,
                           response='susceptibility', mode='pw', **fxckwargs)

        # Parallelization over frequencies depends on the frequency input
        self.blocks1d = None

    def get_macroscopic_component(self, spincomponent, q_c, frequencies,
                                  filename=None, txt=None):
        """Calculates the spatially averaged (macroscopic) component of the
        susceptibility tensor and write it to a file (optional).

        Parameters
        ----------
        spincomponent, q_c,
        frequencies : see gpaw.response.chiks, gpaw.response.kslrf
        filename : str
            Save chiks_w and chi_w to file of given name.
        txt : str
            Save output of the calculation of this specific component into
            a file with the filename of the given input.

        Returns
        -------
        see calculate_macroscopic_component
        """
        (omega_w,
         chiks_w,
         chi_w) = self.calculate_macroscopic_component(spincomponent, q_c,
                                                       frequencies,
                                                       txt=txt)

        if filename is not None and self.context.world.rank == 0:
            from gpaw.response.df import write_response_function
            write_response_function(filename, omega_w, chiks_w, chi_w)

        return omega_w, chiks_w, chi_w

    def calculate_macroscopic_component(self, spincomponent,
                                        q_c, frequencies, txt=None):
        """Calculates the spatially averaged (macroscopic) component of the
        susceptibility tensor.

        Parameters
        ----------
        spincomponent, q_c,
        frequencies : see gpaw.response.chiks, gpaw.response.kslrf
        txt : see get_macroscopic_component

        Returns
        -------
        omega_w, chiks_w, chi_w : nd.array, nd.array, nd.array
            omega_w: frequencies in eV
            chiks_w: macroscopic dynamic susceptibility (Kohn-Sham system)
            chi_w: macroscopic dynamic susceptibility
        """
        (pd, wd,
         chiks_wGG, chi_wGG) = self.calculate_component(spincomponent, q_c,
                                                        frequencies, txt=txt)

        # Macroscopic component
        chiks_w = chiks_wGG[:, 0, 0]
        chi_w = chi_wGG[:, 0, 0]

        # Collect data for all frequencies
        omega_w = wd.omega_w * Hartree
        chiks_w = self.collect(chiks_w)
        chi_w = self.collect(chi_w)

        return omega_w, chiks_w, chi_w

    def get_component_array(self, spincomponent, q_c, frequencies,
                            array_ecut=50, filename=None, txt=None):
        """Calculates a specific spin component of the susceptibility tensor,
        collects it as a numpy array in a reduced plane wave description
        and writes it to a file (optional).

        Parameters
        ----------
        spincomponent, q_c,
        frequencies : see gpaw.response.chiks, gpaw.response.kslrf
        array_ecut : see calculate_component_array
        filename : str
            Save chiks_w and chi_w to pickle file of given name.
        txt : str
            Save output of the calculation of this specific component into
            a file with the filename of the given input.

        Returns
        -------
        see calculate_component_array
        """
        (omega_w, G_Gc, chiks_wGG,
         chi_wGG) = self.calculate_component_array(spincomponent,
                                                   q_c,
                                                   frequencies,
                                                   array_ecut=array_ecut,
                                                   txt=txt)

        if filename is not None:
            write_component(omega_w, G_Gc, chiks_wGG, chi_wGG,
                            filename, self.context.world)

        return omega_w, G_Gc, chiks_wGG, chi_wGG

    def calculate_component_array(self, spincomponent, q_c, frequencies,
                                  array_ecut=50, txt=None):
        """Calculates a specific spin component of the susceptibility tensor
        and collects it as a numpy array in a reduced plane wave description.

        Parameters
        ----------
        spincomponent, q_c,
        frequencies : see gpaw.response.chiks, gpaw.response.kslrf
        array_ecut : float
            Energy cutoff for the reduced plane wave representation.
            The susceptibility is returned in the reduced representation.

        Returns
        -------
        omega_w, G_Gc, chiks_wGG, chi_wGG : nd.array(s)
            omega_w: frequencies in eV
            G_Gc : plane wave repr. as coordinates on the reciprocal lattice
            chiks_wGG: dynamic susceptibility (Kohn-Sham system)
            chi_wGG: dynamic susceptibility
        """
        (pd, wd,
         chiks_wGG, chi_wGG) = self.calculate_component(spincomponent, q_c,
                                                        frequencies, txt=txt)

        # Get frequencies in eV
        omega_w = wd.omega_w * Hartree

        # Get susceptibility in a reduced plane wave representation
        mask_G = get_pw_reduction_map(pd, array_ecut)
        chiks_wGG = np.ascontiguousarray(chiks_wGG[:, mask_G, :][:, :, mask_G])
        chi_wGG = np.ascontiguousarray(chi_wGG[:, mask_G, :][:, :, mask_G])

        # Get reduced plane wave repr. as coordinates on the reciprocal lattice
        G_Gc = get_pw_coordinates(pd)[mask_G]

        # Gather susceptibilities for all frequencies
        chiks_wGG = self.gather(chiks_wGG, wd)
        chi_wGG = self.gather(chi_wGG, wd)

        return omega_w, G_Gc, chiks_wGG, chi_wGG

    def calculate_component(self, spincomponent, q_c, frequencies, txt=None):
        """Calculate a single component of the susceptibility tensor.

        Parameters
        ----------
        spincomponent, q_c,
        frequencies : see gpaw.response.chiks, gpaw.response.kslrf

        Returns
        -------
        pd : PWDescriptor
            Descriptor object for the plane wave basis
        wd : FrequencyDescriptor
            Descriptor object for the calculated frequencies
        chiks_wGG : ndarray
            The process' block of the Kohn-Sham susceptibility component
        chi_wGG : ndarray
            The process' block of the full susceptibility component
        """

        # Initiate new output file, if supplied
        if txt is not None:
            self.context.new_txt_and_timer(txt)

        # Print to output file
        self.context.print('---------------', flush=False)
        self.context.print('Calculating susceptibility spincomponent='
                           f'{spincomponent} with q_c={q_c}', flush=False)
        self.context.print('---------------')

        wd = FrequencyDescriptor.from_array_or_dict(frequencies)

        # Initialize parallelization over frequencies
        self.blocks1d = Blocks1D(self.context.world, len(wd))

        return self._calculate_component(spincomponent, q_c, wd)

    def _calculate_component(self, spincomponent, q_c, wd):
        """In-place calculation of the given spin-component."""
        pd, chiks_wGG = self.calculate_ks_component(spincomponent, q_c, wd)

        Kxc_GG = self.get_xc_kernel(spincomponent, pd, chiks_wGG=chiks_wGG)
        if spincomponent in ['+-', '-+']:
            # No Hartree kernel
            assert Kxc_GG is not None
            Khxc_GG = Kxc_GG
        else:
            Khxc_GG = self.get_hartree_kernel(pd)
            if Kxc_GG is not None:  # Kxc can be None in the RPA case
                Khxc_GG += Kxc_GG

        chi_wGG = self.invert_dyson(chiks_wGG, Khxc_GG)

        self.context.print('\nFinished calculating component', flush=False)
        self.context.print('---------------')

        return pd, wd, chiks_wGG, chi_wGG

    def get_xc_kernel(self, spincomponent, pd, **ignored):
        return self.fxc(spincomponent, pd)

    def get_hartree_kernel(self, pd):
        """Calculate the Hartree kernel"""
        Kbare_G = get_coulomb_kernel(pd, self.gs.kd.N_c)
        vsqrt_G = Kbare_G ** 0.5
        Kh_GG = np.eye(len(vsqrt_G)) * vsqrt_G * vsqrt_G[:, np.newaxis]

        return Kh_GG

    def calculate_ks_component(self, spincomponent, q_c, wd):
        """Calculate a single component of the Kohn-Sham susceptibility tensor.

        Parameters
        ----------
        spincomponent, q_c : see gpaw.response.chiks, gpaw.response.kslrf
        wd : see calculate_component

        Returns
        -------
        pd : PWDescriptor
            see gpaw.response.chiks, gpaw.response.kslrf
        chiks_wGG : ndarray
            The process' block of the Kohn-Sham susceptibility component
        """
        # ChiKS calculates the susceptibility distributed over plane waves
        pd, chiks_wGG = self.chiks.calculate(q_c, wd,
                                             spincomponent=spincomponent)

        # Redistribute memory, so each block has its own frequencies, but all
        # plane waves (for easy invertion of the Dyson-like equation)
        chiks_wGG = self.chiks.distribute_frequencies(chiks_wGG)

        return pd, chiks_wGG

    @timer('Invert dyson-like equation')
    def invert_dyson(self, chiks_wGG, Khxc_GG):
        """Invert the Dyson-like equation:

        chi = chi_ks + chi_ks Khxc chi
        """
        self.context.print('Inverting Dyson-like equation')
        chi_wGG = np.empty_like(chiks_wGG)
        for w, chiks_GG in enumerate(chiks_wGG):
            chi_GG = invert_dyson_single_frequency(chiks_GG, Khxc_GG)

            chi_wGG[w] = chi_GG

        return chi_wGG

    def collect(self, a_w):
        """Collect frequencies from all blocks"""
        return self.blocks1d.collect(a_w)

    def gather(self, A_wGG, wd):
        """Gather a full susceptibility array to root."""
        # Allocate arrays to gather (all need to be the same shape)
        blocks1d = self.blocks1d
        shape = (blocks1d.blocksize,) + A_wGG.shape[1:]
        tmp_wGG = np.empty(shape, dtype=A_wGG.dtype)
        tmp_wGG[:blocks1d.nlocal] = A_wGG

        # Allocate array for the gathered data
        if self.context.world.rank == 0:
            # Make room for all frequencies
            Npadded = blocks1d.blocksize * blocks1d.blockcomm.size
            shape = (Npadded,) + A_wGG.shape[1:]
            allA_wGG = np.empty(shape, dtype=A_wGG.dtype)
        else:
            allA_wGG = None

        self.context.world.gather(tmp_wGG, 0, allA_wGG)

        # Return array for w indeces on frequency grid
        if allA_wGG is not None:
            allA_wGG = allA_wGG[:len(wd), :, :]

        return allA_wGG


class SusceptibilityFactory:
    """User interface to calculate individual elements of the four-component
    susceptibility tensor, see [PRB 103, 245110 (2021)]."""

    def __init__(self, chiks):
        """Contruct the many-bode susceptibility factory based on a given
        Kohn-Sham susceptibility calculator.

        Parameters
        ----------
        chiks: ChiKS
            ChiKS calculator object
        """
        self.chiks = chiks
        self.context = chiks.context
        self.gs = chiks.gs

        # Prepare a buffer for chiks_wGG
        self.current_spincomponent = None
        self.currentq_c = None
        self.current_frequencies = None
        self._pd = None
        self._chiks_wGG = None

    def __call__(self, spincomponent, q_c, frequencies,
                 fxc='ALDA', fxckwargs={}):
        """Calculate a given element (spincomponent) of the four-component
        Kohn-Sham susceptibility tensor and construct a corresponding many-body
        susceptibility object within a given approximation to the
        exchange-correlation kernel.

        Parameters
        ----------
        spincomponent : str or int
            What susceptibility should be calculated?
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
            'all' is an alias for '00', kept for backwards compability
            Likewise 0 or 1, can be used for 'uu' or 'dd'
        q_c : list or ndarray
            Wave vector
        frequencies : ndarray, dict or FrequencyDescriptor
            Array of frequencies to evaluate the response function at,
            dictionary of parameters for build-in frequency grids or a
            descriptor of those frequencies.
        fxc : str
            Approximation to the xc kernel
        fxckwargs : dict
            Kwargs to the FXCCalculator
        """
        assert isinstance(fxc, str)

        pd, chiks_wGG = self.get_chiks(spincomponent, q_c, frequencies)

        # Calculate the Coulomb kernel
        if spincomponent in ['+-', '-+']:
            assert fxc != 'RPA'
            # No Hartree term in Dyson equation
            Vbare_G = None
        else:
            Vbare_G = get_coulomb_kernel(pd, self.gs.kd.N_c)

        # Calculate the exchange-correlation kernel
        if fxc == 'RPA':
            # No xc kernel by definition
            Kxc_GG = None
        else:
            fxc = get_fxc(self.gs, self.context, fxc,
                          response='susceptibility', mode='pw', **fxckwargs)
            Kxc_GG = self.get_xc_kernel(fxc, spincomponent, pd,
                                        chiks_wGG=chiks_wGG)

        return Chi(pd, chiks_wGG, Vbare_G, Kxc_GG)

    @staticmethod
    def get_xc_kernel(fxc, spincomponent, pd, chiks_wGG=None):
        return fxc(spincomponent, pd)


def invert_dyson_single_frequency(chiks_GG, Khxc_GG):
    """Invert single frequency Dyson equation in plane wave basis:

    chi_GG' = chiks_GG + chiks_GG1 Khxc_G1G2 chi_G2G'
    """
    enhancement_GG = np.linalg.inv(np.eye(len(chiks_GG)) -
                                   np.dot(chiks_GG, Khxc_GG))
    chi_GG = np.dot(enhancement_GG, chiks_GG)

    return chi_GG


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


def symmetrize_reciprocity(pd, A_wGG):
    """In collinear systems without spin-orbit coupling, the plane wave
    susceptibility is reciprocal in the sense that e.g.

    χ_(GG')^(+-)(q, ω) = χ_(-G'-G)^(+-)(-q, ω)

    This method symmetrizes A_wGG in the case where q=0.
    """
    from gpaw.test.response.test_chiks import get_inverted_pw_mapping

    q_c = pd.kd.bzk_kc[0]
    if np.allclose(q_c, 0.):
        invmap_GG = get_inverted_pw_mapping(pd, pd)
        for A_GG in A_wGG:
            tmp_GG = np.zeros_like(A_GG)

            # Symmetrize [χ_(GG')(q, ω) + χ_(-G'-G)(-q, ω)] / 2
            tmp_GG += A_GG
            tmp_GG += A_GG[invmap_GG].T
            A_GG[:] = tmp_GG / 2.


def write_component(omega_w, G_Gc, chiks_wGG, chi_wGG, filename, world):
    """Write the dynamic susceptibility as a pickle file."""
    assert isinstance(filename, str)
    if world.rank == 0:
        with open(filename, 'wb') as fd:
            pickle.dump((omega_w, G_Gc, chiks_wGG, chi_wGG), fd)


def read_component(filename):
    """Read a stored susceptibility component file"""
    assert isinstance(filename, str)
    with open(filename, 'rb') as fd:
        omega_w, G_Gc, chiks_wGG, chi_wGG = pickle.load(fd)

    return omega_w, G_Gc, chiks_wGG, chi_wGG
