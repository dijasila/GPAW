import pytest
import numpy as np
from ase.build import bulk
from gpaw.new.ase_interface import GPAW
from gpaw.spinorbit import soc_eigenstates
from gpaw.mpi import world


def test_orbmag_Ni():
    a = 3.48
    mm = 0.5
    theta = np.rad2deg(np.arccos(1 / np.sqrt(3)))
    phi = 45

    mode = {'name': 'pw', 'ecut': 400}
    kpts = {'size': (4, 4, 4), 'gamma': True}
    par = {'domain': 1, 'band': 1}
    occ = {'name': 'fermi-dirac', 'width': 0.05}
    conv = {'density': 1e-04}

    crystal_col = bulk('Ni', 'fcc', a=a)
    crystal_col.center()
    crystal_col.set_initial_magnetic_moments([mm])

    calc = GPAW(mode=mode, xc='LDA', kpts=kpts, parallel=par,
                occupations=occ, convergence=conv)

    crystal_col.calc = calc
    crystal_col.get_potential_energy()
    with pytest.raises(AssertionError, match='Collinear calculations*'):
        calc.get_orbital_magnetization()
    soc = soc_eigenstates(calc, theta=theta, phi=phi)
    om_col_v = soc.get_orbital_magnetization()

    crystal_ncol = bulk('Ni', 'fcc', a=a)
    crystal_ncol.center()
    magmoms = [[mm / np.sqrt(3), mm / np.sqrt(3), mm / np.sqrt(3)]]
    
    calc = GPAW(mode=mode, xc='LDA', kpts=kpts, parallel=par,
                occupations=occ, convergence=conv,
                symmetry='off', magmoms=magmoms)
    
    crystal_ncol.calc = calc
    crystal_ncol.get_potential_energy()
    soc = soc_eigenstates(calc)
    om_ncol_v = soc.get_orbital_magnetization()
    
    crystal_ncolsoc = bulk('Ni', 'fcc', a=a)
    crystal_ncolsoc.center()
    
    calc = GPAW(mode=mode, xc='LDA', kpts=kpts, parallel=par,
                occupations=occ, convergence=conv,
                symmetry='off', magmoms=magmoms, soc=True)
    
    crystal_ncolsoc.calc = calc
    crystal_ncolsoc.get_potential_energy()
    om_ncolsoc_v = calc.get_orbital_magnetization()
    assert np.linalg.norm(om_ncolsoc_v) == pytest.approx(0.0442, abs=1e-4)
    
    dif1 = np.linalg.norm(om_col_v - om_ncol_v)
    dif2 = np.linalg.norm(om_col_v - om_ncolsoc_v)
    dif3 = np.linalg.norm(om_ncol_v - om_ncolsoc_v)
    
    assert dif1 == pytest.approx(0.0, abs=1.0e-6)
    assert dif2 == pytest.approx(0.0, abs=5.0e-3)
    assert dif3 == pytest.approx(0.0, abs=5.0e-3)
