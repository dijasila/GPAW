import numpy as np
from gpaw.core import UniformGrid, PlaneWaves, PlaneWaveAtomCenteredFunctions


def test_acf():
    alpha = 4.0
    s = (0, 3.0, lambda r: np.exp(-alpha * r**2))
    basis = PlaneWaveAtomicOrbitals([[s]],
                                    positions=[[0.5, 0.5, 0.5]])

    s = gaussian(l=0, alpha=4.0, rcut=3.0)
    basis = AtomCenteredFunctions(
        [[s]],
        positions=[[0.5, 0.5, 0.5]])

    for kpt, wfs in zip(kpts, ibz):
        coefs = {0: np.ones((3, 1))}
        basis.add(coefs, wfs)
