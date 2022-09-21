"""Test functionality to compute Fourier Transforms with PAW corrections"""

# General modules
import numpy as np

# Script modules
from ase import Atoms
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.mft import PlaneWaveBxc
from gpaw.response.susceptibility import get_pw_coordinates
from gpaw.test.response.test_site_kernels import get_PWDescriptor


# ---------- Actual tests ---------- #


def test_Fe_bxc(in_tmp_dir):
    """Test the symmetry relation

    (B^xc_G)^* = B^xc_-G

    for a real life system with d-electrons (bcc-Fe)."""
    # ---------- Inputs ---------- #

    # Part 1: Ground state calculation
    xc = 'LDA'
    kpts = 4
    nbands = 6
    pw = 200
    occw = 0.01
    conv = {'density': 1e-8,
            'forces': 1e-8,
            'bands': nbands}
    a = 2.867
    mm = 2.21

    # Part 2: Bxc calculation
    ecut = 100

    # Part 3: Check symmetry relation

    # ---------- Script ---------- #

    # Part 1: Ground state calculation

    atoms = bulk('Fe', 'bcc', a=a)
    atoms.set_initial_magnetic_moments([mm])
    atoms.center()

    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts={'size': (kpts, kpts, kpts), 'gamma': True},
                nbands=nbands + 4,
                occupations=FermiDirac(occw),
                parallel={'domain': 1},
                spinpol=True,
                convergence=conv
                )

    atoms.calc = calc
    atoms.get_potential_energy()

    # Part 2: Bxc calculation
    Bxc_calc = PlaneWaveBxc(calc)
    pd0 = get_PWDescriptor(atoms, calc, [0., 0., 0.],
                           ecut=ecut,
                           gammacentered=True)

    Bxc_G = Bxc_calc(pd0)

    # Part 3: Check symmetry relation
    G1_G, G2_G = get_inversion_pairs(pd0)

    assert np.allclose(np.conj(Bxc_G[G1_G]), Bxc_G[G2_G])


# ---------- Test functionality ---------- #


def get_mocked_gs_adapter(atom_centered_density, **kwargs):
    """Given an atom centered density, mock up a ground state adapter
    with that given density.

    Parameters
    ----------
    atom_centered_density : method
        Function generating an arbitrary electron density as a function
        of the input position r.
    kwargs : dict
        Arguments for the GPAW calculator.
    """
    # We use a Helium atom, since it does not have any frozen core and
    # a cutoff radius, which is more representative of the periodic table,
    # than it is the case for hydrogen
    atoms = Atoms('He', cell=[10., 10., 10.])
    atoms.center()

    calc = GPAW(**kwargs)
    gs = MockedResponseGroundStateAdapter(atoms, calc, atom_centered_density)

    return gs


class MockedResponseGroundStateAdapter:
    def __init__(self, atoms, calc, atom_centered_density):
        self.atoms = atoms
        self.atom_centered_density = atom_centered_density

        calc.initialize(atoms)

        self.gd = self.get_real_space_grid(calc)
        self.nt_sG = self.calculate_pseudo_density()
        self.setups = self.initialize_paw_setups(calc)
        self.D_asp = self.initialize_core_electrons(calc)

    @staticmethod
    def get_real_space_grid(calc):
        """Take the real-space grid from the initialized calculator."""
        return calc.wfs.gd

    def calculate_pseudo_density(self, atom_centered_density):
        # Calculate and return pseudo density
        pass

    def initialize_paw_setups(self):
        # Create PAW setups and fill in the pseudo and all electron
        # density on the radial grid
        pass

    def initialize_core_electrons(self):
        # Setup core electron array to reflect no core electrons
        pass

    def all_electron_density(self, gridrefinement=1):
        # Calculate and return the all electron density
        pass


def get_inversion_pairs(pd0):
    """Get all pairs of G-indices which correspond to inverted reciprocal
    lattice vectors G and -G."""
    G_Gc = get_pw_coordinates(pd0)

    G1_G = []
    G2_G = []
    paired_indices = []
    for G1, G1_c in enumerate(G_Gc):
        if G1 in paired_indices:
            continue  # Already paired

        for G2, G2_c in enumerate(G_Gc):
            if np.all(G2_c == -G1_c):
                G1_G.append(G1)
                G2_G.append(G2)
                paired_indices += [G1, G2]
                break

    assert len(np.unique(paired_indices)) == len(G_Gc)

    return G1_G, G2_G
