from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase.structure import molecule
from ase.units import mol, kcal, Pascal, m, Bohr
from gpaw.solvation import (
    SolvationGPAW,
    ADM12SmoothStepCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction,
    KB51Volume,
    VolumeInteraction,
    ElDensity
)
from gpaw.solvation.poisson import ADM12PoissonSolver

SKIP_VAC_CALC = True

h = 0.24
vac = 4.0

epsinf = 78.36
rhomin = 0.0001 / Bohr ** 3
rhomax = 0.0050 / Bohr ** 3
st = 50. * 1e-3 * Pascal * m
p = -0.35 * 1e9 * Pascal
T = 298.15

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
    xc='PBE', h=h, poissonsolver=ADM12PoissonSolver(),
    cavity=ADM12SmoothStepCavity(
        rhomin=rhomin, rhomax=rhomax, epsinf=epsinf,
        density=ElDensity(),
        surface_calculator=GradientSurface(),
        volume_calculator=KB51Volume()
        ),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[
        SurfaceInteraction(surface_tension=st),
        VolumeInteraction(pressure=p)
        ]
    )
Ewater = atoms.get_potential_energy()
atoms.get_forces()
ham = atoms.calc.hamiltonian
DGSol = (Ewater - Evac) / (kcal / mol)
print 'Delta Gsol: %s kcal / mol' % (DGSol, )

equal(DGSol, -6.3, 2.)
