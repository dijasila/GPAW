from math import sqrt

import pytest
from ase import Atoms
from ase.dft.bandgap import bandgap

from gpaw import GPAW, FermiDirac
from gpaw.test import equal


@pytest.mark.later
def test_Hubbard_U(in_tmp_dir):
    """Setup up bulk NiO in an antiferromagnetic configuration."""
    # Lattice constant:
    a = 4.19
    b = a / sqrt(2)

    m = 2
    k = 2  # number of k-points

    atoms = Atoms(
        symbols='Ni2O2',
        pbc=True,
        cell=(b, b, a),
        positions=[(0, 0, 0),
                   (b / 2, b / 2, a / 2),
                   (0, 0, a / 2),
                   (b / 2, b / 2, 0)],
        magmoms=(m, -m, 0, 0))

    name = 'ni2o2'
    for setup in ['10', '10:d,6.0']:
        calc = GPAW(
            mode='pw',
            occupations=FermiDirac(width=0.05),
            setups={'Ni': setup},
            txt=name + '.txt',
            kpts=(k, k, k),
            xc='oldPBE')
        atoms.calc = calc
        atoms.get_potential_energy()
        gap, _, _ = bandgap(calc)
        if name == 'ni2o2':
            equal(gap, 0.83, 0.1)
        else:
            equal(gap, 4.80, 0.1)
        name += '+U'
