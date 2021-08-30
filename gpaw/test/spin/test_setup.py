from pytest import approx
from ase import Atoms

from gpaw import GPAW, FermiDirac
from gpaw.cluster import Cluster


def test_H2plus():
    """Set up H2+ with the electron on one side initially"""
    h = 0.2
    box = 3
    distance = 4
    
    atoms = Cluster(Atoms('H2', positions=([0., 0., 0.],
                                           [distance, 0., 0.])))
    atoms.minimal_box(box, h=h)

    # we are not interested in convergence, but just the initial setup
    convergence = {'energy': 1e12, 'density': 1e12, 'eigenstates': 1e12}

    # not setting initial charges should still work
    atoms.calc = GPAW(h=h, charge=1,
                      convergence=convergence)
    atoms.get_potential_energy()

    # electron on left atom
    atoms.set_initial_charges([0, 1])
    atoms.set_initial_magnetic_moments([1, 0])

    atoms.calc = GPAW(h=h,
                      occupations=FermiDirac(0.1, fixmagmom=True),
                      convergence=convergence)
    atoms.get_potential_energy()

    assert atoms.calc.charge == approx(1, abs=1e-10)
    assert atoms.calc.get_occupation_numbers(0, 0)[0] == approx(1, abs=1e-5)
    assert atoms.calc.get_occupation_numbers(0, 1)[0] == approx(0, abs=1e-5)
