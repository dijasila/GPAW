"""Test GW band-gaps for Si."""

import pytest
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world
from gpaw.response.g0w0 import G0W0
from gpaw.utilities import compiled_with_sl

def generate_si_systems():
    a = 5.43
    si1 = bulk('Si', 'diamond', a=a)
    si2 = si1.copy()
    si2.positions -= a / 8

    return [si1, si2]


pytestmark = pytest.mark.skipif(
    world.size != 1 and not compiled_with_sl(),
    reason='world.size != 1 and not compiled_with_sl()')


def run(atoms, symm):
    atoms.calc = GPAW(mode=PW(250),
                      eigensolver='rmm-diis',
                      occupations=FermiDirac(0.01),
                      symmetry=symm,
                      kpts={'size': (2, 2, 2), 'gamma': True},
                      parallel={'domain': 1},
                      txt='si.txt')
    e = atoms.get_potential_energy()
    scalapack = atoms.calc.wfs.bd.comm.size
    atoms.calc.diagonalize_full_hamiltonian(nbands=8, scalapack=scalapack)
    atoms.calc.write('si.gpw', mode='all')
    gw = G0W0('si.gpw', 'gw',
              nbands=8,
              integrate_gamma=0,
              kpts=[(0, 0, 0), (0.5, 0.5, 0)],  # Gamma, X
              ecut=40,
              frequencies={'type': 'nonlinear',
                           'domega0': 0.1},
              eta=0.2,
              relbands=(-1, 2))  # homo, lumo, lumo+1, same as bands=(3, 6)
    results = gw.calculate()
    return e, results


@pytest.mark.response
@pytest.mark.slow
@pytest.mark.parametrize('si', generate_si_systems())
@pytest.mark.parametrize('symm', [{},
                                  'off',
                                  {'time_reversal': False},
                                  {'point_group': False}])
def test_response_gwsi(in_tmp_dir, si, symm):
    e, r = run(si, symm)
    G, X = r['eps'][0]
    results = [e, G[0], G[1] - G[0], X[1] - G[0], X[2] - X[1]]
    G, X = r['qp'][0]
    results += [G[0], G[1] - G[0], X[1] - G[0], X[2] - X[1]]

    assert results == pytest.approx(
        [-9.253,
         5.442, 2.389, 0.403, 0.000,
         6.261, 3.570, 1.323, 0.001], abs=0.003)
