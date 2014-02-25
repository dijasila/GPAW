from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase.structure import molecule
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric
)
import numpy as np

SKIP_REF_CALC = True
dE = 1e-9  # XXX: check: why is there a difference at all?
dF = 1e-7  # -- " --

h = 0.3
vac = 3.0
u0 = .180
T = 298.15
vdw_radii = vdw_radii[:]
vdw_radii[1] = 1.09
atomic_radii = lambda atoms: [vdw_radii[n] for n in atoms.numbers]

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(vac, h)

if not SKIP_REF_CALC:
    atoms.calc = GPAW(xc='LDA', h=h)
    Eref = atoms.get_potential_energy()
    print Eref
    Fref = atoms.get_forces()
    print Fref
else:
    Eref = -11.9880228225
    Fref = np.array(
        [
            [7.34133831e-13, -2.67889414e-13, -6.05011167e+00],
            [1.16453612e-13, 1.61504129e+00, 6.85992775e-02],
            [-1.12236388e-13, -1.61504129e+00, 6.85992775e-02]
            ]
        )


atoms.calc = SolvationGPAW(
    xc='LDA', h=h,
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii=atomic_radii, u0=u0),
        temperature=T
        ),
    dielectric=LinearDielectric(epsinf=1.0),
    )
Etest = atoms.get_potential_energy()
Ftest = atoms.get_forces()
equal(Etest, Eref, dE)
equal(Ftest, Fref, dF)
