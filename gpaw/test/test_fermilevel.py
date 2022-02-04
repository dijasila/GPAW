import numpy as np
import pytest
from ase import Atoms

from gpaw import GPAW, Davidson, FermiDirac


def test_fermilevel(in_tmp_dir):
    atoms = Atoms('He', pbc=True)
    atoms.center(vacuum=3)
    params = dict(nbands=1,
                  eigensolver=Davidson(6),
                  occupations=FermiDirac(0.0))

    atoms.calc = GPAW(**params)
    atoms.get_potential_energy()
    assert np.isinf(atoms.calc.get_fermi_level())

    params['nbands'] = 3
    params['convergence'] = {'bands': 2}
    atoms.calc = GPAW(**params)
    atoms.get_potential_energy()
    homo, lumo = atoms.calc.get_homo_lumo()
    assert homo == pytest.approx(-15.4473, abs=0.01)
    assert lumo == pytest.approx(-0.2566, abs=0.01)

    atoms.calc.write('test.gpw')
    assert np.all(GPAW('test.gpw').get_homo_lumo() == (homo, lumo))
    ef = calc.get_fermi_level()
    equal(ef, -7.85196, 0.01)

    calc.set(occupations=FermiDirac(0.1))
    _ = atoms.get_potential_energy()
    ef = calc.get_fermi_level()
    equal(ef, -7.85196, 0.01)
    calc.write('test')
    equal(GPAW('test').get_fermi_level(), ef, 1e-8)
