from ase import Atoms
from ase.units import Ha

from gpaw import GPAW
from gpaw.test import equal
from gpaw.scf import WorkFunction


# FIXME/ap: Does this need a decorator?

def test_scf_criterion():
    convergence = {'eigenstates': 1.e-1,
                   'density': 1.e-1,
                   'energy': 0.1,
                   'custom': WorkFunction(0.01)}

    atoms = Atoms('HF', [(0., 0.5, 0.5),
                         (0., 0.4, -0.4)],
                  cell=(5., 5., 9.),
                  pbc=(True, True, False))
    atoms.center()
    calc = GPAW(h=0.3,
                nbands=-1,
                convergence=convergence,
                txt=None,
                poissonsolver={'dipolelayer': 'xy'})
    atoms.calc = calc
    atoms.get_potential_energy()
    fermilevel = calc.wfs.fermi_level
    workfunctions = Ha * calc.hamiltonian.get_workfunctions(fermilevel)
    calc.write('scf-criterion.gpw')

    # Flip and use saved calculator.
    atoms = Atoms('HF', [(0., 0.5, -0.5),
                         (0., 0.4, +0.4)],
                  cell=(5., 5., 9.),
                  pbc=(True, True, False))
    atoms.center()
    calc = GPAW('scf-criterion.gpw', txt=None)
    atoms.calc = calc
    atoms.get_potential_energy()
    fermilevel = calc.wfs.fermi_level
    workfunctions2 = Ha * calc.hamiltonian.get_workfunctions(fermilevel)

    equal(workfunctions[0], workfunctions2[1], 0.02)
    equal(workfunctions[1], workfunctions2[0], 0.02)


if __name__ == "__main__":
    test_scf_criterion()
