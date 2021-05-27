import numpy as np
import pytest

from ase import Atoms
from ase.units import Ha

from gpaw import GPAW
from gpaw.scf import WorkFunction, Energy


def test_scf_criterion(in_tmp_dir):
    convergence = {'eigenstates': 1.0,
                   'density': 1.0,
                   'energy': 1.0,
                   'custom': WorkFunction(1.0)}

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
    workfunctions1 = Ha * calc.hamiltonian.get_workfunctions(fermilevel)
    calc.write('scf-criterion.gpw')

    # Flip and use saved calculator; work functions should be opposite.
    atoms = Atoms('HF', [(0., 0.5, -0.5),
                         (0., 0.4, +0.4)],
                  cell=(5., 5., 9.),
                  pbc=(True, True, False))
    atoms.center()
    calc = GPAW('scf-criterion.gpw', txt=None)  # checks loading
    atoms.calc = calc
    atoms.get_potential_energy()
    fermilevel = calc.wfs.fermi_level
    workfunctions2 = Ha * calc.hamiltonian.get_workfunctions(fermilevel)

    assert workfunctions1[0] == pytest.approx(workfunctions2[1])
    assert workfunctions1[1] == pytest.approx(workfunctions2[0])
    assert calc.scf.criteria['work function'].tol == pytest.approx(1.0)

    # Try keyword rather than import syntax.
    convergence = {'eigenstates': 1.0,
                   'density': 1.0,
                   'energy': 1.0,
                   'work function': 0.5}
    calc.set(convergence=convergence)
    atoms.get_potential_energy()
    assert calc.scf.criteria['work function'].tol == pytest.approx(0.5)

    # Switch to H2 for faster calcs.
    for atom in atoms:
        atom.symbol = 'H'

    # Change a default.
    convergence = {'energy': Energy(2.0, n_old=4),
                   'density': np.inf,
                   'eigenstates': np.inf}
    calc.set(convergence=convergence)
    atoms.get_potential_energy()
    assert calc.scf.criteria['energy'].n_old == 4
