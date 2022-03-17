import numpy as np
from ase import Atoms
from ase.data.vdw_alvarez import vdw_radii
import gpaw.solvation as solv
from gpaw.solvation.cavity import BAD_RADIUS_MESSAGE


def test_solvation_nan_radius():
    atoms = Atoms('H')
    atoms.center(vacuum=3.0)
    kwargs = solv.get_HW14_water_kwargs()

    def rfun(a):
        return [np.nan]

    kwargs['cavity'].effective_potential.atomic_radii = rfun
    atoms.calc = solv.SolvationGPAW(xc='LDA', h=0.24, **kwargs)
    try:
        atoms.get_potential_energy()
    except ValueError as error:
        if not error.args[0] == BAD_RADIUS_MESSAGE:
            raise
    else:
        raise AssertionError("Expected ValueError")


def test_use_alvarez():
    """Test that Alvarez vdW-radii are used if not available by default"""
    atoms = Atoms('Fe')
    atoms.center(vacuum=3.0)
    kwargs = solv.get_HW14_water_kwargs()

    convergence = {'eigenstates': 1e24, 'density': 1e24, 'energy': 1e24}
    atoms.calc = solv.SolvationGPAW(
        h=0.24, convergence=convergence, **kwargs)
    atoms.get_potential_energy()
    
    assert atoms.calc.hamiltonian.cavity.effective_potential.atomic_radii(
        atoms) == vdw_radii[26]
