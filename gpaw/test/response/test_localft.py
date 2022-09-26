"""Test functionality to compute Fourier Transforms with PAW corrections"""

# General modules
import numpy as np
import pytest

# Script modules
from ase.units import Bohr, Ha
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.pw.descriptor import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LFC
from gpaw.atom.radialgd import AERadialGridDescriptor
from gpaw.response.localft import LocalFTCalculator, MicroSetup
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


def ae_1s_density_plane_waves(pd, R_v, a=1.0):
    """Calculate the plane-wave components of the density from a 1s
    orbital centered at a given position analytically."""
    # List of all plane waves
    G_Gv = np.array([pd.G_Qv[Q] for Q in pd.Q_qG[0]])
    Gnorm_G = np.linalg.norm(G_Gv, axis=1)

    position_prefactor_G = np.exp(-1.j * np.dot(G_Gv, R_v))
    atomcentered_n_G = 1 / (1 + (Gnorm_G * a / 2.)**2.)**2.

    n_G = position_prefactor_G * atomcentered_n_G

    return n_G


# ---------- Actual tests ---------- #

@pytest.mark.response
def test_localft_grid_calculator(in_tmp_dir):
    """Test that the LocalGridFTCalculator is able to correctly Fourier
    transform the all-electron density of an 1s orbital."""
    # ---------- Inputs ---------- #

    # Real-space grid
    cell_volume = 1e3  # Ångstrøm
    N_grid_points = 1e6

    # 1s orbital radii
    a_a = np.linspace(0.5, 1.5, 10)  # a.u.

    # Plane-wave cutoff
    ecut = 20  # eV

    # Test tolerance
    rtol = 1e-3

    # ---------- Script ---------- #

    # Set up grid descriptor
    lattice_constant = cell_volume**(1 / 3.) / Bohr  # a.u.
    cell_cv = np.array([[lattice_constant, 0., 0.],
                        [0., lattice_constant, 0.],
                        [0., 0., lattice_constant]])
    N_c = np.array([int(N_grid_points**(1 / 3.))] * 3)
    gd = GridDescriptor(N_c, cell_cv=cell_cv)

    # Set up plane-wave descriptor
    qd = KPointDescriptor(np.array([[0., 0., 0.]]))
    pd = PWDescriptor(ecut / Ha, gd, complex, qd)

    # Initialize the LocalGridFTCalculator without a ground state adapter
    localft_calc = LocalFTCalculator.from_rshe_parameters(None, rshelmax=None)

    # Calculate the atomic radius at all grid points
    R_v = np.array([lattice_constant, lattice_constant,
                    lattice_constant]) / 2.  # Place atom at the center
    r_vR = gd.get_grid_point_coordinates()
    r_R = np.linalg.norm(r_vR - R_v[:, np.newaxis, np.newaxis, np.newaxis],
                         axis=0)

    for a in a_a:  # Test different orbital radii
        # Calculate the all-electron density on the real-space grid
        n_sR = np.array([ae_1s_density(r_R, a=a)])

        # Compute the plane-wave components numerically
        n_G = localft_calc._calculate(pd, n_sR, add_total_density)

        # Calculate analytically and check validity of results
        ntest_G = ae_1s_density_plane_waves(pd, R_v, a=a)
        assert np.allclose(n_G, ntest_G, rtol=rtol)


@pytest.mark.response
def test_localft_paw_engine(in_tmp_dir):
    """Test that the LocalPAWFTEngine is able to correctly Fourier
    transform the all-electron density of an 1s orbital."""
    # ---------- Inputs ---------- #

    # Real-space grid
    cell_volume = 1e3  # Ångstrøm
    N_grid_points = 1e6 / 2**3.

    # Radial grid (using standard parameters from Li)
    rgd_a = 0.0023570226039551583
    rgd_b = 0.0004528985507246377
    rgd_N = 2000
    rcut = 2.0  # a.u.

    # 1s orbital radii
    a_a = np.linspace(0.5, 1.5, 10)  # a.u.

    # Plane-wave cutoff
    ecut = 20  # eV

    # Settings for the expansion in real spherical harmonics
    rshe_params_p = [{}]

    # Test tolerance
    rtol = 1e-3

    # To-do: Use newrgd instead of rgd XXX
    # To-do: Find out what redge gives the most precise results XXX
    # To-do: Adjust parameters to improve tolerance XXX
    # To-do: Adjust parameters to speed up test (use fewer L?) XXX
    # To-do: Adopt adjusted parameters in grid test XXX
    # To-do: Use alternative rshe parameters, neglecting L>0 XXX

    # ---------- Script ---------- #

    # Set up grid descriptor
    lattice_constant = cell_volume**(1 / 3.) / Bohr  # a.u.
    cell_cv = np.array([[lattice_constant, 0., 0.],
                        [0., lattice_constant, 0.],
                        [0., 0., lattice_constant]])
    N_c = np.array([int(N_grid_points**(1 / 3.))] * 3)
    gd = GridDescriptor(N_c, cell_cv=cell_cv)

    # Set up atomic position at the center of the unit cell
    R_v = np.array([lattice_constant, lattice_constant,
                    lattice_constant]) / 2.
    pos_ac = np.array([[0.5, 0.5, 0.5]])  # Relative atomic positions

    # Set up radial grid descriptor
    rgd = AERadialGridDescriptor(rgd_a, rgd_b, N=rgd_N)

    # Set up plane-wave descriptor
    qd = KPointDescriptor(np.array([[0., 0., 0.]]))
    pd = PWDescriptor(ecut / Ha, gd, complex, qd)

    for rshe_params in rshe_params_p:
        # Initialize the LocalPAWFTCalculator without a ground state adapter
        localft_calc = LocalFTCalculator.from_rshe_parameters(None,
                                                              **rshe_params)

        for a in a_a:  # Test different orbital radii

            # Calculate the pseudo and ae densities on the radial grid
            n_g = ae_1s_density(rgd.r_g, a=a)
            gcut = rgd.floor(rcut)
            nt_g, _ = rgd.pseudize(n_g, gcut)

            # Set up pseudo and ae densities on the Lebedev quadrature
            from gpaw.sphere.lebedev import Y_nL
            nL = Y_nL.shape[1]
            n_sLg = np.zeros((1, nL, rgd_N), dtype=float)
            nt_sLg = np.zeros((1, nL, rgd_N), dtype=float)
            # 1s <=> (l,m) = (0,0) <=> L=0
            n_sLg[0, 0, :] += np.sqrt(4. * np.pi) * n_g  # Y_0 = 1 / sqrt(4pi)
            nt_sLg[0, 0, :] += np.sqrt(4. * np.pi) * nt_g

            # Calculate the pseudo density on the real-space grid
            # ------------------------------------------------- #
            # We start out by setting up a new radial grid descriptor, which
            # matches the atomic one inside the PAW sphere, but extends all the
            # way to the edge of the unit cell
            redge = np.sqrt(3) * lattice_constant / 2.  # cell corner distance
            Ng = int(np.floor(redge / (rgd_a + rgd_b * redge)) + 1)
            newrgd = AERadialGridDescriptor(rgd_a, rgd_b, N=Ng)
            # Generate pseudo density and splines on the new radial grid
            newn_g = ae_1s_density(newrgd.r_g, a=a)
            newnt_g, _ = newrgd.pseudize(newn_g, gcut, l=0)
            spline = newrgd.spline(newnt_g, l=0, rcut=redge)
            # Use the LocalizedFunctionsCollection to generate pseudo density
            # on the cubic real space grid
            nt_R = gd.zeros()
            lfc = LFC(gd, [[spline]])
            lfc.set_positions(pos_ac)
            lfc.add(nt_R, c_axi=np.sqrt(4. * np.pi))  # Y_0 = 1 / sqrt(4pi)
            nt_sR = np.array([nt_R])

            # Create MicroSetup
            micro_setup = MicroSetup(rgd, Y_nL, n_sLg, nt_sLg)
            micro_setups = [micro_setup]

            # Compute the plane-wave components numerically
            n_G = localft_calc.engine.calculate(pd, nt_sR, [R_v], micro_setups,
                                                add_total_density)

            # Calculate analytically and check validity of results
            ntest_G = ae_1s_density_plane_waves(pd, R_v, a=a)
            assert np.allclose(n_G, ntest_G, rtol=rtol)
        

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
