from ase.units import Bohr
from gpaw.utilities.ps2ae import PS2AE
from gpaw import GPAW
from gpaw.utilities.dipole import dipole_matrix_elements_from_calc


def test_dipole_me(gpw_files):
    """Check dipole matrix-elements for H2 molecule."""
    calc = GPAW(gpw_files['h2_pw_wfs'])

    # Method 1: evaluate all-electron wave functions on fine grid:
    t = PS2AE(calc, grid_spacing=0.05)
    psi0 = t.get_wave_function(0) * Bohr**1.5
    psi1 = t.get_wave_function(1) * Bohr**1.5
    d1_v = -t.gd.calculate_dipole_moment(psi0 * psi1) * Bohr

    # Method 2: use pseudo wave function + PAW corrections:
    d2_nnv = dipole_matrix_elements_from_calc(calc, n1=0, n2=2)[0]

    assert abs(d2_nnv[0, 0] - calc.atoms.cell.sum(0) / 2).max() < 0.04
    assert abs(d2_nnv[1, 1] - calc.atoms.cell.sum(0) / 2).max() < 0.04
    assert abs(d2_nnv[0, 1] - d1_v).max() < 1e-3

    # Method 3: same as above but with translated molecule:
    calc = GPAW(gpw_files['h2_pw_0_wfs'])
    d3_nnv = dipole_matrix_elements_from_calc(calc, n1=0, n2=2,
                                              center=[0, 0, 0])[0]

    assert abs(d3_nnv[0, 0]).max() < 0.04
    assert abs(d3_nnv[1, 1]).max() < 0.04
    assert abs(abs(d3_nnv[0, 1]) - abs(d2_nnv[0, 1])).max() < 1e-7
