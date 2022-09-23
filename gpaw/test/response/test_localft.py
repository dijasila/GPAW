"""Test functionality to compute Fourier Transforms with PAW corrections"""

# General modules
import numpy as np
import pytest
from functools import partial

# Script modules
from ase import Atoms
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.lfc import LFC
from gpaw.atom.radialgd import AERadialGridDescriptor
from gpaw.xc.pawcorrection import PAWXCCorrection
from gpaw.utilities.partition import AtomPartition
from gpaw.response.localft import LocalPAWFT
from gpaw.response.mft import PlaneWaveBxc
from gpaw.response.susceptibility import get_pw_coordinates
from gpaw.test.response.test_site_kernels import get_PWDescriptor


# ---------- Test parametrization ---------- #


def ae_1s_density(r_g, a=1.0):
    """Construct the radial dependence of the density from a 1s orbital on the
    radial grid r_g."""
    assert np.all(r_g >= 0)
    prefactor = 1 / (np.pi * a**3.)
    n_g = prefactor * np.exp(-2. * r_g / a)

    return n_g


def ae_1s_density_plane_waves(pd, pos_v, a=1.0):
    """Calculate the plane-wave components of the density from a 1s
    orbital centered at a given position analytically."""
    # List of all plane waves
    G_Gv = np.array([pd.G_Qv[Q] for Q in pd.Q_qG[0]])
    Gnorm_G = np.linalg.norm(G_Gv, axis=1)

    position_prefactor_G = np.exp(-1.j * np.dot(G_Gv, pos_v))
    atomcentered_n_G = 1 / (1 + (Gnorm_G * a / 2.)**2.)**2.

    n_G = position_prefactor_G * atomcentered_n_G

    return n_G


# ---------- Actual tests ---------- #


@pytest.mark.response
def test_atomic_orbital_densities(in_tmp_dir):
    """Test that the LocalPAWFT is able to correctly Fourier transform
    the all-electron density of atomic orbitals."""
    # ---------- Inputs ---------- #

    # Ground state calculator
    mode = PW(200)
    nbands = 6

    # Atomic densities
    a_a = np.linspace(0.5, 1.5, 10)

    # Plane-wave components
    ecut = 100

    # To do: Test the calculator params of LocalPAWFT XXX
    # To do: Test the all-electron version of LocalPAWFT XXX
    # To do: Test different orbitals XXX
    # To do: Test combinations of orbitals XXX

    # ---------- Script ---------- #

    for a in a_a:
        # Set up ground state adapter
        atom_centered_density = partial(ae_1s_density, a=a)
        gs = get_mocked_gs_adapter(atom_centered_density,
                                   mode=mode, nbands=nbands)

        # Set up FT calculator and plane-wave descriptor
        aenft = LocalPAWFT(gs)
        pd = get_PWDescriptor(gs.atoms, gs.get_calc(), [0., 0., 0.],
                              ecut=ecut,
                              gammacentered=True)

        # Calculate the plane-wave components of the all electron density
        n_G = aenft(pd, add_total_density)

        # Calculate analytically and check validity of results
        pos_v = gs.atoms.positions[0]
        ntest_G = ae_1s_density_plane_waves(pd, pos_v, a=a)
        assert np.allclose(n_G, ntest_G)
        

@pytest.mark.response
def dont_test_Fe_bxc(in_tmp_dir):
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


def add_total_density(gd, n_sR, f_R):
    f_R += np.sum(n_sR, axis=0)


def get_mocked_gs_adapter(atom_centered_density, **kwargs):
    """Given an atom centered density, mock up a ground state adapter
    with that given density.

    Parameters
    ----------
    atom_centered_density : method
        Function generating an arbitrary radial dependency of the electron
        density as a function of the radius r_g.
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
        assert len(atoms) == 1
        self.atoms = atoms
        self.atom_centered_density = atom_centered_density

        calc.initialize(atoms)
        self._calc = calc
        self._D_asp = self._calc.density.D_asp

        self.gd = self.get_real_space_grid()
        self.setups = self.get_mocked_paw_setups()
        self.nt_sG = self.get_mocked_pseudo_density()

    def get_real_space_grid(self):
        """Take the real-space grid from the initialized calculator."""
        return self._calc.wfs.gd

    def get_mocked_paw_setups(self):
        """Mock up the PAW setups to fill in the pseudo and all electron
        density on the radial grid."""
        setup = self._calc.wfs.setups[0]  # only a single atom
        xc_correction = setup.xc_correction
        rgd = xc_correction.rgd

        # NB: Hard-coded to 1s angular dependence for now! XXX
        # Calculate all-electron partial wave
        n_g = self.atom_centered_density(rgd.r_g)
        w_jg = np.zeros((xc_correction.nj, len(n_g)), dtype=float)
        w_jg[0, :] += np.sqrt(n_g)

        # Pseudize to get the pseudo partial wave
        rcut = np.max(setup.data.rcut_j)
        gcut = rgd.floor(rcut)
        wt_jg = w_jg.copy()
        wt_jg[0, :] = rgd.pseudize(w_jg[0, :], gcut, l=0)[0]

        # No core electrons so far! XXX
        nc_g = rgd.zeros()
        nct_g = rgd.zeros()

        # Get remaining xc_correction arguments
        jl = list(enumerate(setup.data.l_j))
        lmax = int(np.sqrt(xc_correction.Lmax)) - 1
        e_xc0 = xc_correction.e_xc0
        phicorehole_g = None
        fcorehole = 0.0
        tauc_g = xc_correction.tauc_g
        tauct_g = xc_correction.tauct_g

        # Set up mocked xc_correction object
        mocked_xc_correction = PAWXCCorrection(
            w_jg,
            wt_jg,
            nc_g,
            nct_g,
            rgd,
            jl,
            lmax,
            e_xc0,
            phicorehole_g,
            fcorehole,
            tauc_g,
            tauct_g)

        setup.xc_correction = mocked_xc_correction
        setups = [setup]

        # Set setups onto the private calculator object
        self._calc.wfs.setups = setups

        return setups

    def get_mocked_pseudo_density(self):
        """Mock up the pseudo density on the real space grid."""
        # We assume a single atom, centered in the unit cell
        pos_ac = self.atoms.get_scaled_positions()
        assert pos_ac.shape[0] == 1
        assert np.allclose(pos_ac, np.array([[0.5, 0.5, 0.5]]))

        # Extract data
        rcut = np.max(self.setups[0].data.rcut_j)
        rgd = self.setups[0].xc_correction.rgd
        gcut = rgd.floor(rcut)

        # We start out by setting up a new radial grid descriptor, which
        # matches the atomic one inside the PAW sphere, but extends all the
        # way to the edge of the unit cell
        redge = np.linalg.norm(self.atoms.positions[0]) / np.sqrt(3)
        Ng = int(np.floor(redge / (rgd.a + rgd.b * redge)) + 1)
        newrgd = AERadialGridDescriptor(rgd.a, rgd.b, N=Ng)

        # Generate pseudo density and splines on the new radial grid
        # NB: Hard-coded to 1s angular dependence for now! XXX
        n_g = self.atom_centered_density(newrgd.r_g)
        nt_g, _ = newrgd.pseudize(n_g, gcut, l=0)
        spline = newrgd.spline(nt_g, l=0, rcut=redge)

        # Use the LocalizedFunctionsCollection to generate pseudo density
        # on the cubic real space grid
        nt_G = self.gd.zeros()
        lfc = LFC(self.gd, [[spline]])
        lfc.set_positions(pos_ac)
        lfc.add(nt_G)  # Add pseudo density from spline to pseudo density array

        # Make it possible to set up a spin polarized one in the future? XXX
        nt_sG = np.array([nt_G])

        return nt_sG

    @property
    def D_asp(self):
        if self._D_asp is None:
            density = self._calc.density
            atom_partition = AtomPartition(self.gd.comm,
                                           np.zeros(len(self.atoms), int),
                                           'density-gd')
            # D_asp = self.setups.empty_atomic_matrix(density.ncomponents,
            #                                         density.atom_partition)
            Dshapes_a = [(density.ncomponents, setup.ni * (setup.ni + 1) // 2)
                         for setup in self.setups]
            D_asp = atom_partition.arraydict(Dshapes_a, float)
            self._calc.wfs.calculate_atomic_density_matrices(D_asp)

            self._D_asp = D_asp

        return self._D_asp

    def get_calc(self):
        return self._calc

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
