from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase.build import molecule
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric)
import numpy as np

SKIP_REF_CALC = True

energy_eps = 0.0005 / 8.
forces_eps = 2e-5

h = 0.3
vac = 3.0
u0 = .180
T = 298.15
vdw_radii = vdw_radii.copy()
vdw_radii[1] = 1.09


def atomic_radii(atoms):
    return [vdw_radii[n] for n in atoms.numbers]

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(vac, h)

convergence = {
    'energy': energy_eps,
    'forces': forces_eps ** 2,  # Force error is squared
    'density': 10.,
    'eigenstates': 10.,
}

if not SKIP_REF_CALC:
    atoms.calc = GPAW(xc='LDA', h=h, convergence=convergence)
    Eref = atoms.get_potential_energy()
    print(Eref)
    Fref = atoms.get_forces()
    print(Fref)
else:
    # setups: 0.9.11271, same settings as above
    Eref = -11.987992702
    Fref = np.array(
            [[  3.02502086e-13, -1.44665920e-12, -6.05004154e+00],
             [  1.23004481e-13,  1.61482505e+00,  6.87380315e-02],
             [  2.57885170e-13, -1.61482505e+00,  6.87380315e-02]])

atoms.calc = SolvationGPAW(
    xc='LDA', h=h, convergence=convergence,
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii=atomic_radii, u0=u0),
        temperature=T
    ),
    dielectric=LinearDielectric(epsinf=1.0),
)
Etest = atoms.get_potential_energy()
Eeltest = atoms.calc.get_electrostatic_energy()
Ftest = atoms.get_forces()
equal(Etest, Eref, energy_eps * atoms.calc.get_number_of_electrons())
equal(Ftest, Fref, forces_eps)
equal(Eeltest, Etest)
