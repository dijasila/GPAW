from gpaw.cluster import Cluster
from ase.structure import molecule
from ase.data.vdw import vdw_radii
from ase.units import Bohr
from gpaw.solvation import (
    SolvationGPAW,
    Power12VdWCavityDensity,
    BoltzmannSmoothedStep,
    LinearDielectric,
)
from gpaw.solvation.poisson import ADM12PoissonSolver

h = 0.3
vac = 3.0
rho0 = 1. / 7.
epsinf = 80.
vdw_radii = vdw_radii[:]
vdw_radii[1] = 1.09

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(vac, h)
atoms.pbc = True
atoms.calc = SolvationGPAW(
    xc='LDA', h=h,
    cavdens=Power12VdWCavityDensity(vdw_radii),
    smoothedstep=BoltzmannSmoothedStep(rho0 / Bohr ** 3),
    dielectric=LinearDielectric(epsinf=epsinf),
    poissonsolver=ADM12PoissonSolver(eps=1e-7)
    )
atoms.get_potential_energy()
