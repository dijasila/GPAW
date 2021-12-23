"""Test if site-kernels give the right number of elements and overall scale."""


# Import modules
from gpaw import GPAW, PW
from ase.units import Bohr
from My_classes.Calculator_classes import StaticChiKSFactory
from My_functions.Calc_site_kernels import calc_K_mixed_shapes
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
