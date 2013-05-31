from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase import Atoms
from ase.data.vdw import vdw_radii
from ase.units import Bohr
from gpaw.solvation import (
    SolvationGPAW,
    RepulsiveVdWCavityDensity,
    BoltzmannSmoothedStep,
    CMDielectric
)
import numpy as np

np.seterr(all='raise')

SKIP_REF_CALC = True
dE = 1e-9

h = 0.3
vac = 3.0
r0 = 0.4
rho0 = 1.5

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
    cavdens=RepulsiveVdWCavityDensity(vdw_radii, r0 * Bohr),
    smoothedstep=BoltzmannSmoothedStep(rho0 / Bohr ** 3),
    dielectric=CMDielectric(epsinf=1.0),
    )
Etest = atoms.get_potential_energy()
equal(Etest, Eref, dE)
