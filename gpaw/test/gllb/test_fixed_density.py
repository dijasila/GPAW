import pytest
from ase.build import molecule
from gpaw import GPAW
from gpaw.mpi import world


@pytest.mark.gllb
@pytest.mark.libxc
def test_fixed_density(in_tmp_dir):
    atoms = molecule('H2')
    atoms.center(vacuum=4)

    print('GS calc')
    gs_calc = GPAW(mode='lcao', basis='sz(dzp)', h=0.4,
                   xc='GLLBSC',
                   txt='gs.out',
                   )
    atoms.calc = gs_calc
    atoms.get_potential_energy()
    gs_calc.write('gs.gpw', mode='all')

    print('BS calc')
    bs_calc = gs_calc.fixed_density(parallel={'band': min(2, world.size)},
                                    txt='unocc.out',
                                    )
    bs_calc.write('unocc.gpw', mode='all')
