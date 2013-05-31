from gpaw.cluster import Cluster
from ase.structure import molecule
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

h = 0.3
vac = 3.0
r0 = 0.4
rho0 = 1.5
epsinf = 80.

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(vac, h)
atoms.pbc = True
atoms.calc = SolvationGPAW(
    xc='LDA', h=h,
    cavdens=RepulsiveVdWCavityDensity(vdw_radii, r0 * Bohr),
    smoothedstep=BoltzmannSmoothedStep(rho0 / Bohr ** 3),
    dielectric=CMDielectric(epsinf=epsinf)
    )
atoms.get_potential_energy()
