from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase import Atoms
from ase.data.vdw import vdw_radii
from ase.units import Bohr
from gpaw.solvation import (
    SolvationGPAW,
    Power12VdWCavityDensity,
    BoltzmannSmoothedStep,
    LinearDielectric
)

SKIP_REF_CALC = True
dE = 1e-9

h = 0.3
vac = 3.0
rho0 = 1. / 7.
vdw_radii = vdw_radii[:]
vdw_radii[1] = 1.09

atoms = Cluster(Atoms('H'))
atoms.minimal_box(vac, h)

if not SKIP_REF_CALC:
    atoms.calc = GPAW(xc='LDA', h=h)
    Eref = atoms.get_potential_energy()
    print Eref
else:
    Eref = 0.684394163586

atoms.calc = SolvationGPAW(
    xc='LDA', h=h,
    cavdens=Power12VdWCavityDensity(vdw_radii),
    smoothedstep=BoltzmannSmoothedStep(rho0 / Bohr ** 3),
    dielectric=LinearDielectric(epsinf=1.0),
    )
Etest = atoms.get_potential_energy()
equal(Etest, Eref, dE)
