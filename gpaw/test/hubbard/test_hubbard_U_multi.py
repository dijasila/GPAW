from ase import Atom

from gpaw import GPAW, FermiDirac
from gpaw.cluster import Cluster


def test_Hubbard_U_Zn_multi():
    h = .35
    box = 3.

    s = Cluster([Atom('Zn')])
    s.minimal_box(box, h=h)

    c = GPAW(h=h, spinpol=True,
             xc='PBE',
             mode='lcao', basis='sz(dzp)',
             parallel=dict(kpt=1),
             charge=1, occupations=FermiDirac(width=0.1),
             setups=':d,3.0,1'
             )
    s.calc = c
    single = s.get_potential_energy()
    
    s.calc.set(setups=':d,3.0,1;p,0.0,1')  # No difference
    multi30 = s.get_potential_energy()

    s.calc.set(setups=':d,3.0,1;p,2.0,1')  # p correction
    multi32 = s.get_potential_energy()

    s.calc.set(setups=':p,2.0,1;d,3.0,1')  # swap order for d/p
    multi32_swap = s.get_potential_energy()

    assert single == multi30
    assert multi32 == multi32_swap
    assert single != multi32
