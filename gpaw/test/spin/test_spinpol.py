from ase import Atom, Atoms
from gpaw import GPAW, FermiDirac
from gpaw.test import equal


def test_spin_spinpol():
    a = 4.0
    n = 16
    hydrogen = Atoms([Atom('H')], cell=(a, a, a), pbc=True)
    hydrogen.center()
    calc = GPAW(gpts=(n, n, n), nbands=1, convergence={'energy': 1e-5},
                occupations=FermiDirac(0.0))
    hydrogen.calc = calc
    e1 = hydrogen.get_potential_energy()
    hydrogen.set_initial_magnetic_moments([1.0])
    e2 = hydrogen.get_potential_energy()
    de = e1 - e2
    print(de)
    equal(de, 0.7871, 1.e-4)

    energy_tolerance = 0.0006
    equal(e1, -0.499854, energy_tolerance)
    equal(e2, -1.287, energy_tolerance)
