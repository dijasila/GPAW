from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase.structure import molecule
from ase.units import Bohr, mol, kcal, Pascal, m
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    RepulsiveVdWCavityDensity,
    BoltzmannSmoothedStep,
    CMDielectric,
    QuantumSurfaceInteraction,
)
import numpy as np

np.seterr(all='raise')

SKIP_VAC_CALC = True

h = 0.24
vac = 4.0
r0 = 0.4
rho0 = 1.5
epsinf = 78.36
st = 48.2

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(vac, h)

if not SKIP_VAC_CALC:
    atoms.calc = GPAW(xc='PBE', h=h)
    Evac = atoms.get_potential_energy()
    print Evac
else:
    #Evac = -14.6154407425  # h = 0.2, vac = 4.0
    Evac = -14.862428  # h = 0.24, vac = 4.0

atoms.calc = SolvationGPAW(
    xc='PBE', h=h,
    cavdens=RepulsiveVdWCavityDensity(vdw_radii, r0 * Bohr),
    smoothedstep=BoltzmannSmoothedStep(rho0 / Bohr ** 3),
    dielectric=CMDielectric(epsinf=epsinf),
    interactions=[
        QuantumSurfaceInteraction(
            surface_tension=st * 1e-3 * Pascal * m,
            delta=1e-6 / Bohr ** 3
            )
        ]
    )
Ewater = atoms.get_potential_energy()
ham = atoms.calc.hamiltonian
DGSol = (Ewater - Evac) / (kcal / mol)
print 'Delta Gsol: %s kcal / mol' % (DGSol, )

equal(DGSol, -6.3, 2.)
