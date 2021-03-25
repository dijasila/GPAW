import numpy as np
import pytest

from gpaw.mpi import world
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.fd_operators import Gradient
from gpaw.grid_descriptor import GridDescriptor
from gpaw.setup import Setup
from gpaw.spherical_harmonics import YL
from gpaw.utilities.tools import coordinates


def rlYlm(L, r_vg):
    r"""Calculates :math:`r^{l} Y_{l,m}(\theta, \varphi)` on grid."""
    rlYlm_g = np.zeros_like(r_vg[0])
    for c, n_v in YL[L]:
        rlYlm_g += c * np.prod(np.moveaxis(r_vg, 0, -1)**n_v, -1)
    return rlYlm_g


class DummySetup(Setup):
    def __init__(self, lmax):
        self.l_j = range(lmax + 1)
        self.nj = lmax + 1
        self.ni = (lmax + 1)**2


def calculate_integrals_on_regular_grid(radial_function, *,
                                        lmax=2,
                                        cell_v=[24, 24, 24],
                                        N_v=[128, 128, 128]):
    cell_v = np.asarray(cell_v, dtype=float)
    gd = GridDescriptor(N_v, cell_v, False)
    origin_v = 0.5 * cell_v
    r_vg, r2_g = coordinates(gd, origin=origin_v)
    r_g = np.sqrt(r2_g)
    radial_g = radial_function(r_g)

    grad_v = []
    for v in range(3):
        grad_v.append(Gradient(gd, v, n=3))
    grad_phi2_vg = gd.empty(3)

    Lmax = (lmax + 1)**2  # 1+3+5+...
    nabla_LLv = np.zeros((Lmax, Lmax, 3))
    rxnabla_LLv = np.zeros((Lmax, Lmax, 3))
    for L2 in range(Lmax):
        phi2_g = radial_g * rlYlm(L2, r_vg)
        for v in range(3):
            grad_v[v].apply(phi2_g, grad_phi2_vg[v])
        for L1 in range(Lmax):
            phi1_g = radial_g * rlYlm(L1, r_vg)

            def nabla(v):
                return gd.integrate(phi1_g * grad_phi2_vg[v])

            def rxnabla(v1, v2):
                return gd.integrate(phi1_g *
                                    (r_vg[v1] * grad_phi2_vg[v2] -
                                     r_vg[v2] * grad_phi2_vg[v1]))

            for v in range(3):
                v1 = (v + 1) % 3
                v2 = (v + 2) % 3
                nabla_LLv[L1, L2, v] = nabla(v)
                rxnabla_LLv[L1, L2, v] = rxnabla(v1, v2)

    return {'nabla': nabla_LLv, 'rxnabla': rxnabla_LLv}


def calculate_integrals_on_radial_grid(radial_function, *,
                                       lmax=2,
                                       h=1e-3, N=12e3,
                                       use_phit=False):
    setup = DummySetup(lmax)
    rgd = EquidistantRadialGridDescriptor(h, int(N))
    r_g = rgd.r_g
    radial_g = radial_function(r_g)
    phi_jg = [radial_g * r_g**l for l in setup.l_j]
    phit_jg = np.zeros_like(phi_jg)
    if use_phit:
        phit_jg, phi_jg = phi_jg, phit_jg
    nabla_LLv = setup.get_derivative_integrals(rgd, phi_jg, phit_jg)
    rxnabla_LLv = setup.get_magnetic_integrals(rgd, phi_jg, phit_jg)
    return {'nabla': nabla_LLv, 'rxnabla': rxnabla_LLv}


@pytest.fixture(scope='module')
def lmax():
    return 2


@pytest.fixture(scope='module')
def radial_function():
    return lambda r_g: np.exp(-0.1 * r_g**2)


@pytest.fixture(scope='module')
def integrals_on_regular_grid(lmax, radial_function):
    # Increase accuracy with the number of processes
    N = {1: 32, 2: 48, 4: 64, 8: 92}.get(world.size, 32)
    return calculate_integrals_on_regular_grid(radial_function,
                                               lmax=lmax,
                                               N_v=[N, N, N])


@pytest.fixture(scope='module')
def integrals_on_radial_grid(lmax, radial_function):
    return calculate_integrals_on_radial_grid(radial_function,
                                              lmax=lmax)


@pytest.mark.parametrize('kind', ['nabla', 'rxnabla'])
def test_integrals(kind,
                   integrals_on_regular_grid,
                   integrals_on_radial_grid):
    arr1_LLv = integrals_on_regular_grid[kind]
    arr2_LLv = integrals_on_radial_grid[kind]
    if world.rank == 0:
        np.set_printoptions(precision=4, suppress=True, linewidth=2000)
        print(kind)
        for v in range(3):
            print('xyz'[v])
            print(arr1_LLv[..., v])
            print(arr2_LLv[..., v])
    world.barrier()
    # Accuracy is increased with the number of processes
    rtol = {1: 5e-4, 2: 5e-5, 4: 9e-6, 8: 1e-6}.get(world.size, 1e-3)
    assert np.allclose(arr1_LLv, arr2_LLv, rtol=rtol, atol=1e-12)


def test_phit_integrals(lmax, radial_function, integrals_on_radial_grid):
    phit_integrals = \
        calculate_integrals_on_radial_grid(radial_function,
                                           lmax=lmax,
                                           use_phit=True)
    for kind, ref_LLv in integrals_on_radial_grid.items():
        arr_LLv = phit_integrals[kind]
        assert np.allclose(ref_LLv, -arr_LLv, rtol=0, atol=0)
