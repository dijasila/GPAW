import pytest
from ase.data import atomic_numbers

from gpaw.atom.generator2 import DatasetGenerationError, generate

XC = 'PBE'


def generate_setup(name, proj, x, type, nderiv0):
    splitname = name.split('_')
    element = splitname[0]

    gen = generate(
        element,
        atomic_numbers[element],
        proj,
        x[:-1],
        x[-1],
        None,
        nderiv0,
        xc=XC,
        pseudize=type,
        scalar_relativistic=True,
        ecut=None,
    )

    if not gen.check_all():
        raise DatasetGenerationError


@pytest.mark.serial
@pytest.mark.parametrize(
    'symb, par',
    [('C',
      ('2s,s,2p,p,d',
       [1.2, 1.4, 1.1],
       ('orthonormal', (4, 50, {'inv_gpts': [2, 1.5]}, 'nc')),
       2))])
def test_generate_setup(symb, par):
    generate_setup(symb, *par)
