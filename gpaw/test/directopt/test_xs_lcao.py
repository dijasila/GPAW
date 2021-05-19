from gpaw import GPAW, LCAO
from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.exstatetools import excite_and_sort
from ase import Atoms
import numpy as np
from gpaw.test import equal


def test_xc_pw(gpw_files):
    atoms = Atoms('O2', [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    atoms.center(vacuum=3.0)
    pos = atoms.get_positions()
    pos[0][2] += 0.01
    atoms.set_positions(pos)
    calc = GPAW(mode=LCAO(), spinpol=True,
                symmetry='off',
                eigensolver=DirectMinLCAO(),
                mixer={'name': 'dummy'},
                occupations={'name': 'fixed-occ-zero-width'},
                nbands='nao'
                )
    atoms.calc = calc
    atoms.get_potential_energy()
    i, a = 0, 1
    excite_and_sort(calc.wfs, i, a, (0, 0), 'lcao')
    calc.set(eigensolver=DirectMinLCAO(
        searchdir_algo={'name': 'LSR1P', 'method': 'LSR1'},
        linesearch_algo={'name': 'UnitStep'}))
    f_sn = []
    for spin in range(calc.get_number_of_spins()):
        f_n = calc.get_occupation_numbers(spin=spin)
        f_sn.append(f_n)
    prepare_mom_calculation(calc, atoms, f_sn)
    e = atoms.get_potential_energy()
    equal(e, -0.353021, 1.0e-6)
    f = atoms.get_forces()
    equal(np.min(f), -17.74544, 1.0e-3)
    equal(np.max(f), 17.77512, 1.0e-3)

    pos = atoms.get_positions()
    pos[0][2] -= 0.01
    atoms.set_positions(pos)
    e2 = atoms.get_potential_energy()
    equal(e2, -0.527327, 1.0e-6)
