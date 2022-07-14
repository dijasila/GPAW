import numpy as np
from ase import Atoms
# from ase.units import Ha
from gpaw import GPAW, PW
# from gpaw.hybrids import HybridXC
# from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw.hybrids.energy import non_self_consistent_energy
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw.xc.exx import EXX
from gpaw.mpi import world, serial_comm


def test(kpts, setup, spinpol, symmetry):
    a = Atoms('HLi',
              cell=(3, 3, 3),
              pbc=1,
              positions=[[1.4, 0, 1.05], [1.6, 0, 2]]
              # positions=[[0, 0, 1.25], [0, 0, 2]]
              # positions=[[0, 0, 0], [0, 0, 0.75]]
              )
    a.calc = GPAW(mode=PW(100, force_complex_dtype=True),
                  setups=setup,
                  kpts=kpts,
                  spinpol=spinpol,
                  # nbands=1,
                  symmetry=symmetry,
                  # parallel={'band': 1, 'kpt': 1},
                  txt=None,
                  xc='PBE')
    a.get_potential_energy()
    return a


def check(atoms, xc, i):
    # xc1 = HybridXC(xc)
    c = atoms.calc
    # xc1.initialize(c.density, c.hamiltonian, c.wfs, c.occupations)
    # xc1.set_positions(c.spos_ac)
    e = non_self_consistent_energy(c, xc)
    c.write('c.gpw', 'all')
    eps = non_self_consistent_eigenvalues(c, xc, 0, 2, snapshot=f'{i}.json')
    # eps = non_self_consistent_eigenvalues('c.gpw', xc, 0, 2,
    #                                      snapshot=f'{i}.json')
    # xc1.calculate_eigenvalues0(0, 2, None)
    # e1, v1, v2 = non_self_consistent_eigenvalues(c, xc, 0, 2, None,
    #                                             f'{i}.txt')
    if world.size > 1:
        c.write('tmp.gpw', 'all')
        c = GPAW('tmp.gpw', communicator=serial_comm, txt=None)

    xc2 = EXX(c.gs_adapter(), xc=xc, bands=(0, 2), txt=None)
    xc2.calculate()
    # e0 = xc2.get_exx_energy()
    et0 = xc2.get_total_energy()
    eps0 = xc2.get_eigenvalue_contributions()
    assert np.allclose(eps0, eps[2]), (eps0, eps)
    # assert np.allclose(v2, xc1.e_skn * Ha), (v2, xc1.e_skn * Ha, eps0)
    # assert np.allclose(eps0, xc1.e_skn * Ha)
    # print(e0, e)
    # assert np.allclose(e0, e[-3:].sum())
    # print(et0, e);asdfg
    assert np.allclose(et0, e.sum()), (et0, e)
    # ecv, evv, v_skn = xc1.test()
    # assert np.allclose(e0, ecv + evv)
    # assert np.allclose(v_skn, eps0)


def main():
    i = 0
    for spinpol in [False, True]:
        for setup in [
            # 'ae',
            'paw']:
            for symmetry in ['off', {}]:
                for kpts in [(1, 1, 1),
                             (1, 1, 2),
                             (1, 1, 3),
                             (1, 1, 4),
                             (2, 2, 1),
                             [(0, 0, 0.5)],
                             [(0, 0, 0), (0, 0, 0.5)]]:
                    atoms = test(kpts, setup, spinpol, symmetry)
                    for xc in ['EXX',
                               'PBE0', 'HSE06']:
                        print(i, spinpol, setup, symmetry, kpts, xc,
                              len(atoms.calc.wfs.kpt_u))
                        check(atoms, xc, i)
                        i += 1


main()
