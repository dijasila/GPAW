import numpy as np

from ase import Atoms
from gpaw import GPAW, FermiDirac, Davidson
from gpaw.test import equal


def test_fermilevel(in_tmp_dir):
    calc = GPAW(nbands=1,
                eigensolver=Davidson(6),
                occupations=FermiDirac(0.0))
    atoms = Atoms('He', pbc=True, calculator=calc)
    atoms.center(vacuum=3)

    _ = atoms.get_potential_energy()
    assert np.isinf(calc.get_fermi_level())
    calc.set(nbands=3, convergence={'bands': 2})
    atoms.get_potential_energy()
    homo, lumo = calc.get_homo_lumo()
    equal(homo, -15.4473, 0.01)
    equal(lumo, -0.2566, 0.01)
    calc.write('test')
    assert np.all(GPAW('test', txt=None).get_homo_lumo() == (homo, lumo))
    ef = calc.get_fermi_level()
    equal(ef, -7.85196, 0.01)

    calc.set(occupations=FermiDirac(0.1))
    _ = atoms.get_potential_energy()
    ef = calc.get_fermi_level()
    equal(ef, -7.85196, 0.01)
    calc.write('test')
    equal(GPAW('test', txt=None).get_fermi_level(), ef, 1e-8)
