from ase import Atom

from gpaw import GPAW, FermiDirac
from gpaw.cluster import Cluster
from gpaw.test import equal


def test_Hubbard_U_Zn():
    h = .35
    box = 3.
    energy_tolerance = 0.0004

    s = Cluster([Atom('Zn')])
    s.minimal_box(box, h=h)

    E = {}
    E_U = {}
    for spin in [0, 1]:
        params = dict(h=h,
                      spinpol=spin,
                      mode='lcao',
                      basis='sz(dzp)',
                      parallel=dict(kpt=1),
                      charge=1,
                      occupations=FermiDirac(width=0.1, fixmagmom=spin))
        s.calc = GPAW(**params)
        E[spin] = s.get_potential_energy()
        s.calc = GPAW(**params, setups=':d,3.0,1')
        E_U[spin] = s.get_potential_energy()

    print("E=", E)
    equal(E[0], E[1], energy_tolerance)
    print("E_U=", E_U)
    equal(E_U[0], E_U[1], energy_tolerance)
