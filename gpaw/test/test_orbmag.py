import pytest

import numpy as np

from gpaw.new.ase_interface import GPAW
from gpaw.spinorbit import soc_eigenstates


def test_orbmag_Ni(gpw_files):
    # Parameters

    easy_axis = 1 / np.sqrt(3) * np.ones(3)
    theta = np.rad2deg(np.arccos(easy_axis[2]))
    phi = 45

    # Collinear calculation

    calc_col = GPAW(gpw_files['fcc_Ni_col'],
                    parallel={'domain': 1, 'band': 1})

    energy_col = calc_col.get_potential_energy(calc_col.atoms)
    density = calc_col.calculation.state.density
    magmoms_col_v, _ = density.calculate_magnetic_moments()
    with pytest.raises(AssertionError, match='Collinear calculations*'):
        calc_col.get_orbital_magnetic_moments()
    orbmag_col_v = soc_eigenstates(calc_col,
                                   theta=theta,
                                   phi=phi).get_orbital_magnetic_moments()[0]

    # Non-collinear calculation without self-consistent spin-orbit

    calc_ncol = GPAW(gpw_files['fcc_Ni_ncol'],
                     parallel={'domain': 1, 'band': 1})

    energy_ncol = calc_ncol.get_potential_energy(calc_ncol.atoms)
    density = calc_ncol.calculation.state.density
    magmoms_ncol_v, _ = density.calculate_magnetic_moments()
    orbmag_ncol_v = soc_eigenstates(
        calc_ncol).get_orbital_magnetic_moments()[0]

    # Test that col and ncol give the same groundstate (with rotated magmoms)
    # and the same orbital magnetic moments from the soc_eigenstates module

    dif_energy = energy_ncol - energy_col
    dif_magmom = np.linalg.norm(magmoms_ncol_v) - magmoms_col_v[2]
    dif_orbmag = np.linalg.norm(orbmag_ncol_v - orbmag_col_v)

    assert dif_energy == pytest.approx(0, abs=1.0e-6)
    assert dif_magmom == pytest.approx(0, abs=1.0e-6)
    assert dif_orbmag == pytest.approx(0, abs=1.0e-6)

    # Non-collinear calculation with self-consistent spin-orbit

    calc_ncolsoc = GPAW(gpw_files['fcc_Ni_ncolsoc'],
                        parallel={'domain': 1, 'band': 1})

    calc_ncolsoc.get_potential_energy(calc_ncolsoc.atoms)
    orbmag_ncolsoc_v = calc_ncolsoc.get_orbital_magnetic_moments()[0]

    # Assert direction and magnitude of orbital magnetic moment
    assert np.linalg.norm(orbmag_ncolsoc_v) == pytest.approx(
        0.045176, abs=1e-6)
    assert np.dot(orbmag_ncolsoc_v, easy_axis) == pytest.approx(
        0.045176, abs=1e-6)

    # Get difference between orbital magnetic moments when soc is included
    # self-consistently. Assert that this difference doesn't change.

    dif_orbmag2 = np.linalg.norm(orbmag_ncolsoc_v - orbmag_col_v)
    dif_orbmag3 = np.linalg.norm(orbmag_ncolsoc_v - orbmag_ncol_v)

    assert dif_orbmag2 == pytest.approx(0.002516, abs=1e-6)
    assert dif_orbmag3 == pytest.approx(0.002516, abs=1e-6)
