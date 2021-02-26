from ase import Atoms
from gpaw import GPAW
import numpy as np


def test_atoms_too_close():
    atoms = Atoms('H2', [(0.0, 0.0, 0.0),
                         (0.0, 0.0, 3.995)],
                  cell=(4, 4, 4), pbc=True)

    calc = GPAW(txt=None)
    atoms.calc = calc
    try:
        calc.initialize(atoms)
        calc.set_positions(atoms)
    except RuntimeError as err:
        print('got error as expected: {}'.format(err))
    else:
        # silly exception where we actually skip the check for older numpies
        if hasattr(np, 'divmod'):
            assert 2 + 2 == 5
