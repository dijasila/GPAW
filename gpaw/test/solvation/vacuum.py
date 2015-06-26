from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase.structure import molecule
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric)
import numpy as np

SKIP_REF_CALC = True
dE = 1e-9  # XXX: check: why is there a difference at all?
dF = 1e-7  # -- " --

h = 0.3
vac = 3.0
u0 = .180
T = 298.15
vdw_radii = vdw_radii.copy()
vdw_radii[1] = 1.09
atomic_radii = lambda atoms: [vdw_radii[n] for n in atoms.numbers]

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(vac, h)

convergence = {
    'energy': 0.05 / 8.,
    'density': 10.,
    'eigenstates': 10.,
}

if not SKIP_REF_CALC:
    atoms.calc = GPAW(xc='LDA', h=h, convergence=convergence)
    Eref = atoms.get_potential_energy()
    print(Eref)
    Fref = atoms.get_forces()
    print(Fref)
else:
    # h=0.3, vac=3.0, setups: 0.9.11271, convergence: only energy 0.05 / 8
    Eref = -11.9838020445
    Fref = np.array(
        [
            [9.68634537e-13, -6.74234034e-13, -3.39987211e+00],
            [4.93976820e-14, 1.75606065e+00, 1.68012920e-02],
            [1.07907651e-13, -1.75606065e+00, 1.68012920e-02]
        ]
    )

atoms.calc = SolvationGPAW(
    xc='LDA', h=h, convergence=convergence,
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii=atomic_radii, u0=u0),
        temperature=T
    ),
    dielectric=LinearDielectric(epsinf=1.0),
)
Etest = atoms.get_potential_energy()
Eeltest = atoms.calc.get_electrostatic_energy()
Ftest = atoms.get_forces()
equal(Etest, Eref, dE)
equal(Ftest, Fref, dF)
equal(Eeltest, Etest)
