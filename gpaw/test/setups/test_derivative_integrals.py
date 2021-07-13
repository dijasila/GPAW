import numpy as np
import pytest

from gpaw.mpi import world
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.fd_operators import Gradient
from gpaw.grid_descriptor import GridDescriptor
from gpaw.setup import Setup, Setups
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
                                       use_phit=False,
                                       alt_rxnabla=False):
    setup = DummySetup(lmax)
    rgd = EquidistantRadialGridDescriptor(h, int(N))
    r_g = rgd.r_g
    radial_g = radial_function(r_g)
    phi_jg = [radial_g * r_g**l for l in setup.l_j]
    phit_jg = np.zeros_like(phi_jg)
    if use_phit:
        phit_jg, phi_jg = phi_jg, phit_jg
    nabla_LLv = setup.get_derivative_integrals(rgd, phi_jg, phit_jg)
    if alt_rxnabla:
        from .rxnabla_alternative import get_magnetic_integrals_alt
        rxnabla_LLv = get_magnetic_integrals_alt(setup, rgd, phi_jg, phit_jg)
    else:
        rxnabla_LLv = setup.get_magnetic_integrals(rgd, phi_jg, phit_jg)
    return {'nabla': nabla_LLv, 'rxnabla': rxnabla_LLv}


@pytest.fixture(scope='module')
def lmax():
    return 3


@pytest.fixture(scope='module')
def radial_function():
    return lambda r_g: np.exp(-0.1 * r_g**2)


@pytest.fixture(scope='module')
def integrals_on_regular_grid(lmax, radial_function):
    # Increase accuracy with the number of processes
    # Here, world.size = 40 triggers more accurate calculation
    N = {2: 48, 4: 64, 8: 92, 40: 256}.get(world.size, 32)
    return calculate_integrals_on_regular_grid(radial_function,
                                               lmax=lmax,
                                               N_v=[N, N, N])


@pytest.fixture(scope='module')
def integrals_on_radial_grid(lmax, radial_function):
    return calculate_integrals_on_radial_grid(radial_function,
                                              lmax=lmax)


@pytest.mark.parametrize('kind', ['nabla', 'rxnabla'])
@pytest.mark.parametrize('lmax_test', range(4))
def test_integrals(kind,
                   lmax,
                   lmax_test,
                   integrals_on_regular_grid,
                   integrals_on_radial_grid):
    # This test is parametrized over different lmax values,
    # because the accuracy of the regular grid calculation
    # depends strongly on that
    assert lmax >= lmax_test, 'this test is broken, increase lmax'
    arr1_LLv = integrals_on_regular_grid[kind]
    arr2_LLv = integrals_on_radial_grid[kind]
    if world.rank == 0 and lmax_test == lmax:
        np.set_printoptions(precision=4, suppress=True, linewidth=2000)
        print(kind)
        for v in range(3):
            print('xyz'[v])
            print(arr1_LLv[..., v])
            print(arr2_LLv[..., v])
    world.barrier()
    # Accuracy is increased with the finer grid
    # (as triggered by the number of processes)
    rtol = {2: 5e-5, 4: 9e-6, 8: 1e-6, 40: 6e-8}.get(world.size, 1e-3)
    atol = 1e-12
    if lmax_test == 3:
        # Regular grid is not fine enough for high lmax, which
        # results in large spurious values that should be zero
        atol = {2: 3e-1, 4: 4e-2, 8: 5e-3, 40: 2e-5}.get(world.size, 3e-0)
    Lmax = (lmax_test + 1)**2
    assert np.allclose(arr1_LLv[:Lmax, :Lmax], arr2_LLv[:Lmax, :Lmax],
                       rtol=rtol, atol=atol)


def test_phit_integrals(lmax, radial_function, integrals_on_radial_grid):
    phit_integrals = \
        calculate_integrals_on_radial_grid(radial_function,
                                           lmax=lmax,
                                           use_phit=True)
    for kind, ref_LLv in integrals_on_radial_grid.items():
        arr_LLv = phit_integrals[kind]
        assert np.allclose(ref_LLv, -arr_LLv, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['nabla', 'rxnabla'])
def test_skew_symmetry(kind, integrals_on_radial_grid):
    arr_LLv = integrals_on_radial_grid[kind]
    rtol = {'nabla': 1e-8, 'rxnabla': 0}[kind]
    for v in range(3):
        arr_LL = arr_LLv[..., v]
        assert np.allclose(arr_LL, -arr_LL.T, rtol=rtol, atol=1e-11)


def test_rxnabla_vs_alt_implementation(lmax, radial_function,
                                       integrals_on_radial_grid):
    # Alternative implementation supports lmax upto 2
    lmax = min(lmax, 2)
    Lmax = (lmax + 1)**2
    rxnabla_LLv = integrals_on_radial_grid['rxnabla'][:Lmax, :Lmax, :]
    alt_LLv = calculate_integrals_on_radial_grid(radial_function,
                                                 lmax=lmax,
                                                 alt_rxnabla=True)['rxnabla']
    assert np.allclose(rxnabla_LLv, alt_LLv, rtol=0, atol=1e-12)


def test_lmax_zero(radial_function):
    integrals = \
        calculate_integrals_on_radial_grid(radial_function,
                                           lmax=0)
    for kind, ref_LLv in integrals.items():
        assert np.allclose(ref_LLv, 0, rtol=0, atol=0)


@pytest.mark.parametrize('xc', ['LDA', 'PBE'])
@pytest.mark.parametrize('Z', (
    list(range(1, 43)) + list(range(44, 57)) + list(range(72, 84)) + [86]))
def test_skew_symmetry_real_setup(Z, xc):
    setup = Setups([Z], {}, {}, xc)[0]
    for v in range(3):
        arr_ii = setup.nabla_iiv[:, :, v]
        assert np.allclose(arr_ii, -arr_ii.T, rtol=5e-6, atol=1e-12)
        arr_ii = setup.rxnabla_iiv[:, :, v]
        assert np.allclose(arr_ii, -arr_ii.T, rtol=0, atol=1e-14)
