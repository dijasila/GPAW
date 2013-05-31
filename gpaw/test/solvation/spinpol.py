from gpaw.test import equal
from gpaw.cluster import Cluster
from ase.structure import molecule
from ase.units import Bohr, Pascal, m
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    RepulsiveVdWCavityDensity,
    BoltzmannSmoothedStep,
    CMDielectric,
    QuantumSurfaceInteraction,
    QuantumVolumeInteraction,
    LeakedDensityInteraction
)
import numpy as np

np.seterr(all='raise')

h = 0.3
vac = 3.0
r0 = 0.4
rho0 = 1.5
epsinf = 80.

atoms = Cluster(molecule('CN'))
atoms.minimal_box(vac, h)
atoms2 = atoms.copy()
atoms2.set_initial_magnetic_moments(None)

atomss = (atoms, atoms2)
Es = []

for atoms in atomss:
    atoms.calc = SolvationGPAW(
        xc='LDA', h=h, charge=-1,
        cavdens=RepulsiveVdWCavityDensity(vdw_radii, r0 * Bohr),
        smoothedstep=BoltzmannSmoothedStep(rho0 / Bohr ** 3),
        dielectric=CMDielectric(epsinf=epsinf),
        interactions=[
            QuantumSurfaceInteraction(
                surface_tension=100. * 1e-3 * Pascal * m,
                delta=1e-6 / Bohr ** 3
                ),
            QuantumVolumeInteraction(
                pressure=-1.0 * 1e9 * Pascal
                ),
            LeakedDensityInteraction(
                charging_energy=1.0
                )
            ]
        )
    Es.append(atoms.get_potential_energy())

print 'difference: ', Es[0] - Es[1]
equal(Es[0], Es[1], 0.0002)  # compare to difference of a gas phase calc
