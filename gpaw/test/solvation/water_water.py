from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase.structure import molecule
from ase.units import Bohr, mol, kcal, Pascal, m
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    Power12VdWCavityDensity,
    BoltzmannSmoothedStep,
    LinearDielectric,
    QuantumSurfaceInteraction,
)

SKIP_VAC_CALC = True

h = 0.24
vac = 4.0
rho0 = 1. / 7.
epsinf = 78.36
st = 18.4
vdw_radii = vdw_radii[:]
vdw_radii[1] = 1.09

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
    cavdens=Power12VdWCavityDensity(vdw_radii),
    smoothedstep=BoltzmannSmoothedStep(rho0 / Bohr ** 3),
    dielectric=LinearDielectric(epsinf=epsinf),
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
