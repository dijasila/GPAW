import pickle

import numpy as np

from ase.units import Hartree

from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.fxc_kernels import get_fxc
from gpaw.response.coulomb_kernels import get_coulomb_kernel
from gpaw.response.dyson import DysonSolver
from gpaw.response.goldstone import get_scaled_xc_kernel
from gpaw.response.pw_parallelization import Blocks1D


class ChiFactory:
    r"""User interface to calculate individual elements of the four-component
    susceptibility tensor χ^μν, see [PRB 103, 245110 (2021)]."""

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
        self.current_q_c = None
        self.current_wd = None
        self._pd = None
        self._chiks_wGG = None
        self._blocks1d = None

    def __call__(self, spincomponent, q_c, frequencies,
                 fxc='ALDA', fxckwargs=None, txt=None):
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
        pd, wd, blocks1d, chiks_wGG = self.get_chiks(spincomponent, q_c,
                                                     frequencies)

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
            Kxc_GG = self.get_xc_kernel(fxc, spincomponent, pd,
                                        fxckwargs=fxckwargs,
                                        wd=wd,
                                        blocks1d=blocks1d,
                                        chiks_wGG=chiks_wGG)

        return Chi(self.context,
                   pd, wd, blocks1d, chiks_wGG, Vbare_G, Kxc_GG)

    def get_chiks(self, spincomponent, q_c, frequencies):
        """Get chiks_wGG from buffer."""
        q_c = np.asarray(q_c)
        wd = FrequencyDescriptor.from_array_or_dict(frequencies)

        if self._chiks_wGG is None or\
            not (spincomponent == self.current_spincomponent and
                 np.allclose(q_c, self.current_q_c) and
                 np.allclose(wd.omega_w, self.current_wd.omega_w)):
            # Calculate new chiks_wGG, if buffer is empty or if we are
            # considering a new set of spincomponent, q-vector and frequencies
            pd, chiks_wGG = self.chiks.calculate(q_c, wd,
                                                 spincomponent=spincomponent)

            # Redistribute memory, so that the frequencies are distributed over
            # the entire world
            chiks_wGG = self.chiks.distribute_frequencies(chiks_wGG)
            blocks1d = Blocks1D(self.context.world, len(wd))

            # Fill buffer
            self.current_spincomponent = spincomponent
            self.current_q_c = q_c
            self.current_wd = wd
            self._pd = pd
            self._blocks1d = blocks1d
            self._chiks_wGG = chiks_wGG

        return self._pd, self.current_wd, self._blocks1d, self._chiks_wGG

    def get_xc_kernel(self, fxc, spincomponent, pd, *,
                      fxckwargs, wd, blocks1d, chiks_wGG):
        """Calculate the xc kernel."""
        if fxckwargs is None:
            fxckwargs = {}
        assert isinstance(fxckwargs, dict)
        if 'fxc_scaling' in fxckwargs:
            assert spincomponent in ['+-', '-+']
            fxc_scaling = fxckwargs['fxc_scaling']
        else:
            fxc_scaling = None

        fxc_calculator = get_fxc(self.gs, self.context, fxc,
                                 response='susceptibility', mode='pw',
                                 **fxckwargs)

        Kxc_GG = fxc_calculator(spincomponent, pd)

        if fxc_scaling is not None:
            self.context.print('Rescaling kernel to fulfill the Goldstone '
                               'theorem')
            Kxc_GG = get_scaled_xc_kernel(pd, wd, blocks1d, chiks_wGG,
                                          Kxc_GG, fxc_scaling)

        return Kxc_GG


class Chi:
    """Many-body susceptibility in a plane-wave basis."""

    def __init__(self, context,
                 pd, wd, blocks1d, chiks_wGG, Vbare_G, Kxc_GG):
        """Construct the many-body susceptibility based on its ingredients."""
        self.context = context
        self.pd = pd
        self.wd = wd
        self.blocks1d = blocks1d
        self.chiks_wGG = chiks_wGG
        self.Vbare_G = Vbare_G
        self.Kxc_GG = Kxc_GG  # Use Kxc_G in the future XXX

        self.dysonsolver = DysonSolver(self.context)

    def write_macroscopic_component(self, filename):
        """Calculate the spatially averaged (macroscopic) component of the
        susceptibility and write it to a file together with the Kohn-Sham
        susceptibility and the frequency grid."""
        from gpaw.response.df import write_response_function

        chiks_wGG = self.chiks_wGG
        chi_wGG = self._calculate()

        # Macroscopic component
        chiks_w = chiks_wGG[:, 0, 0]
        chi_w = chi_wGG[:, 0, 0]

        # Collect all frequencies from world
        omega_w = self.wd.omega_w * Hartree
        chiks_w = self.collect(chiks_w)
        chi_w = self.collect(chi_w)

        if self.context.world.rank == 0:
            write_response_function(filename, omega_w, chiks_w, chi_w)

    def write_component_array(self, filename, *, reduced_ecut):
        """Calculate the many-body susceptibility and write it to a file along
        with the Kohn-Sham susceptibility and frequency grid within a reduced
        plane-wave basis."""

        chiks_wGG = self.chiks_wGG
        chi_wGG = self._calculate()

        # Get susceptibility in a reduced plane-wave representation
        mask_G = get_pw_reduction_map(self.pd, reduced_ecut)
        chiks_wGG = np.ascontiguousarray(chiks_wGG[:, mask_G, :][:, :, mask_G])
        chi_wGG = np.ascontiguousarray(chi_wGG[:, mask_G, :][:, :, mask_G])

        # Get reduced plane wave repr. as coordinates on the reciprocal lattice
        G_Gc = get_pw_coordinates(self.pd)[mask_G]

        # Gather all frequencies from world to root
        omega_w = self.wd.omega_w * Hartree
        chiks_wGG = self.gather(chiks_wGG)
        chi_wGG = self.gather(chi_wGG)

        if self.context.world.rank == 0:
            write_component(omega_w, G_Gc, chiks_wGG, chi_wGG, filename)

    def _calculate(self):
        """Calculate chi_wGG."""
        return self.dysonsolver.invert_dyson(self.chiks_wGG, self.Khxc_GG)

    @property
    def Khxc_GG(self):
        """Hartree-exchange-correlation kernel."""
        # Allocate array
        nG = self.chiks_wGG.shape[2]
        Khxc_GG = np.zeros((nG, nG), dtype=complex)

        if self.Vbare_G is not None:  # Add the Hartree kernel
            Khxc_GG.flat[::nG + 1] += self.Vbare_G
        if self.Kxc_GG is not None:  # Add the xc kernel
            # In the future, construct the xc kernel here! XXX
            Khxc_GG += self.Kxc_GG

        return Khxc_GG

    def collect(self, a_w):
        """Collect all frequencies."""
        return self.blocks1d.collect(a_w)

    def gather(self, A_wGG):
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
            allA_wGG = allA_wGG[:len(self.wd), :, :]

        return allA_wGG


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

    q_c = pd.q_c
    if np.allclose(q_c, 0.):
        invmap_GG = get_inverted_pw_mapping(pd, pd)
        for A_GG in A_wGG:
            # Symmetrize [χ_(GG')(q, ω) + χ_(-G'-G)(-q, ω)] / 2
            A_GG[:] = (A_GG + A_GG[invmap_GG].T) / 2.


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
