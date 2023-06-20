import pytest
import numpy as np
from ase.build import bulk
from gpaw.new.ase_interface import GPAW
from gpaw.spinorbit import soc_eigenstates


def test_orbmag_Ni():

    # Parameters

    a = 3.48
    mm = 0.5
    easy_axis = 1 / np.sqrt(3) * np.ones(3)
    theta = np.rad2deg(np.arccos(easy_axis[2]))
    phi = 45

    mode = {'name': 'pw', 'ecut': 400}
    kpts = {'size': (4, 4, 4), 'gamma': True}
    par = {'domain': 1, 'band': 1}
    occ = {'name': 'fermi-dirac', 'width': 0.05}
    conv = {'density': 1e-04}

    # Collinear calculation

    calc = GPAW(mode=mode, xc='LDA', kpts=kpts, parallel=par,
                occupations=occ, convergence=conv)
    crystal_col = bulk('Ni', 'fcc', a=a)
    crystal_col.center()
    crystal_col.set_initial_magnetic_moments([mm])
    crystal_col.calc = calc

    energy_col = crystal_col.get_potential_energy()
    with pytest.raises(AssertionError, match='Collinear calculations*'):
        calc.get_orbital_magnetic_moments()
    magmoms_col_v, _ = calc.calculation.state.density \
        .calculate_magnetic_moments()
    orbmag_col_v = soc_eigenstates(calc, theta=theta, phi=phi) \
        .get_orbital_magnetic_moments()[0]

    # Non-collinear calculation without self-consistent spin-orbit

    calc = GPAW(mode=mode, xc='LDA', kpts=kpts, parallel=par,
                occupations=occ, convergence=conv, symmetry='off',
                magmoms=[mm * easy_axis])
    crystal_ncol = bulk('Ni', 'fcc', a=a)
    crystal_ncol.center()
    crystal_ncol.calc = calc

    energy_ncol = crystal_ncol.get_potential_energy()
    magmoms_ncol_v, _ = calc.calculation.state.density \
        .calculate_magnetic_moments()
    orbmag_ncol_v = soc_eigenstates(calc).get_orbital_magnetic_moments()[0]
    
    # Test that col and ncol give the same groundstate (with rotated magmoms)
    # and the same orbital magnetic moments from the soc_eigenstates module

    dif_energy = energy_ncol - energy_col
    dif_magmom = np.linalg.norm(magmoms_ncol_v) - magmoms_col_v[2]
    dif_orbmag = np.linalg.norm(orbmag_ncol_v - orbmag_col_v)

    assert dif_energy == pytest.approx(0, abs=1.0e-6)
    assert dif_magmom == pytest.approx(0, abs=1.0e-6)
    assert dif_orbmag == pytest.approx(0, abs=1.0e-6)
    
    # Non-collinear calculation with self-consistent spin-orbit

    calc = GPAW(mode=mode, xc='LDA', kpts=kpts, parallel=par,
                occupations=occ, convergence=conv, symmetry='off',
                magmoms=[mm * easy_axis], soc=True)
    crystal_ncolsoc = bulk('Ni', 'fcc', a=a)
    crystal_ncolsoc.center()
    crystal_ncolsoc.calc = calc

    crystal_ncolsoc.get_potential_energy()
    orbmag_ncolsoc_v = calc.get_orbital_magnetic_moments()[0]
    
    # Assert direction and magnitude of orbital magnetic moment
    assert np.linalg.norm(orbmag_ncolsoc_v) == \
           pytest.approx(0.044260262840633, abs=1e-6)
    assert np.dot(orbmag_ncolsoc_v, easy_axis) == \
           pytest.approx(0.044260262840633, abs=1e-6)

    # Get difference between orbital magnetic moments when soc is included
    # self-consistently. Assert that this difference doesn't change.

    dif_orbmag2 = np.linalg.norm(orbmag_ncolsoc_v - orbmag_col_v)
    dif_orbmag3 = np.linalg.norm(orbmag_ncolsoc_v - orbmag_ncol_v)

    assert dif_orbmag2 == pytest.approx(0.0022177, abs=1e-6)
    assert dif_orbmag3 == pytest.approx(0.0022177, abs=1e-6)
