"""Test if site-kernels give the right number of elements and overall scale."""


# Import modules
from gpaw import GPAW, PW
from ase.units import Bohr
from gpaw.response.mft import StaticChiKSFactory
from gpaw.response.site_kernels import calc_K_mixed_shapes
from gpaw.response.susceptibility import get_pw_coordinates
from gpaw.test import equal
from ase.build import bulk
import numpy as np


def test_Co_hcp():
    # Use Co(hcp) as test system
    atoms = bulk('Co', 'hcp', a=2.5071, c=4.0695)
    calc = GPAW(xc='LDA',
                spinpol=True,
                mode=PW(100),
                kpts={'size': (2, 2, 2),
                      'gamma': True}
                )
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    chiksf = StaticChiKSFactory(calc, ecut=50)

    pd0 = chiksf.get_PWDescriptor([0, 0, 0])
    Omega_cell = atoms.get_volume()     # Volume of unit cell in Bohr^3
    sitePose_mv = atoms.positions
    NG = len(get_pw_coordinates(pd0))

    # Compute site-kernels
    Kuc_GGm = calc_K_mixed_shapes(pd0, sitePose_mv, shapes_m='unit cell')
    Ksph_GGm = calc_K_mixed_shapes(pd0, sitePose_mv, shapes_m='sphere',
                                   rc_m=rc_m)
    Kcyl_GGm = calc_K_mixed_shapes(pd0, sitePose_mv, shapes_m='cylinder',
                                   rc_m=rc_m, zc_m=zc_m)

    # Check shape of K-arrays
    assert Kuc_GGm.shape == (NG, NG, 1)
    assert Ksph_GGm.shape == (NG, NG, 2)
    assert Kcyl_GGm.shape == (NG, NG, 2)

    # Remove constant prefactor (atomic units)
    prefactor = np.sqrt(2 / (Omega_cell / Bohr**3)**3)
    Kuc_GGm, Ksph_GGm, Kcyl_GGm = (x / prefactor
                                   for x in [Kuc_GGm, Ksph_GGm, Kcyl_GGm])

    # Convert to SI units
    Kuc_GGm, Ksph_GGm, Kcyl_GGm = (x*Bohr**3
                                   for x in [Kuc_GGm, Ksph_GGm, Kcyl_GGm])

    # Check K_00(q=0) is proportional to the volume of the integration region
    tol = 1e-6
    equal(Kuc_GGm[0, 0, 0], Omega_cell, tolerance=tol)  # Unit cell volume
    Vsph1 = 4/3*np.pi*rc_m[0]**3  # Volume of sphere on first site
    equal(Ksph_GGm[0, 0, 0], Vsph1, tolerance=tol)
    Vsph2 = 4/3*np.pi*rc_m[1]**3  # Volume of sphere on second site
    equal(Ksph_GGm[0, 0, 1], Vsph2, tolerance=tol)
    Vcyl1 = np.pi*rc_m[0]**2*(2*rc_m[0])  # Volume of cylinder on first site
    equal(Kcyl_GGm[0, 0, 0], Vcyl1, tolerance=tol)
    height = np.sum(atoms.cell[:, -1])   # Height of unit cell
    Vcyl2 = np.pi*rc_m[1]**2*height     # Volume of cylinder on second site
    equal(Kcyl_GGm[0, 0, 1], Vcyl2, tolerance=tol)
