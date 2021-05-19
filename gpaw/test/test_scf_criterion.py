import pytest

from ase import Atoms
from ase.units import Ha

from gpaw import GPAW
from gpaw.scf import WorkFunction


def test_scf_criterion(in_tmp_dir):
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

    assert workfunctions[0] == pytest.approx(workfunctions2[1])
    assert workfunctions[1] == pytest.approx(workfunctions2[0])
