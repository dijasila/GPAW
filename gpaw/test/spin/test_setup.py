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

    # electron on left atom
    atoms.set_initial_charges([0, 1])
    atoms.set_initial_magnetic_moments([1, 0])

    # does not converge
    atoms.calc = GPAW(h=h, charge=1,
                      occupations=FermiDirac(0.1, fixmagmom=True),
                      convergence={'energy': 1e12, 'density': 1e12,
                                   'eigenstates': 1e12})
    atoms.get_potential_energy()
