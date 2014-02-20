from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase.structure import molecule
from ase.units import mol, kcal, Pascal, m
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction,
)

SKIP_VAC_CALC = True

h = 0.24
vac = 4.0
u0 = 0.180
epsinf = 78.36
st = 18.4 * 1e-3 * Pascal * m
T = 298.15
vdw_radii = vdw_radii[:]
vdw_radii[1] = 1.09
atomic_radii = lambda atoms: [vdw_radii[n] for n in atoms.numbers]

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
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii, u0),
        temperature=T,
        surface_calculator=GradientSurface()
        ),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[SurfaceInteraction(surface_tension=st)]
    )
Ewater = atoms.get_potential_energy()
atoms.get_forces()
DGSol = (Ewater - Evac) / (kcal / mol)
print 'Delta Gsol: %s kcal / mol' % (DGSol, )

equal(DGSol, -6.3, 2.)
