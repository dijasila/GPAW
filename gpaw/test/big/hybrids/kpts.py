import numpy as np
from ase import Atoms
from ase.units import Ha
from gpaw import GPAW, PW
from gpaw.hybrids import HybridXC
from gpaw.xc.exx import EXX
from gpaw.mpi import world, serial_comm


def test(kpts, setup, spinpol, symmetry):
    a = Atoms('H2', cell=(3, 3, 3), pbc=1, positions=[[0, 0, 0], [0, 0, 0.75]])
    a.calc = GPAW(mode=PW(100, force_complex_dtype=True),
                  setups=setup,
                  kpts=kpts,
                  spinpol=spinpol,
                  # nbands=1,
                  symmetry=symmetry,
                  parallel={'band': 1, 'kpt': 1},
                  txt=None,
                  xc='PBE')
    a.get_potential_energy()
    return a


def check(atoms, xc, i):
    xc1 = HybridXC(xc)
    c = atoms.calc
    xc1.calculate_eigenvalues(c, 0, 2, None, restart=f'tmp-{i}.json')
    e = xc1.calculate_energy()

    if world.size > 1:
        c.write('tmp.gpw', 'all')
        c = GPAW('tmp.gpw', communicator=serial_comm, txt=None)

    xc2 = EXX(c, xc=xc, bands=(0, 2), txt=None)
    xc2.calculate()
    e0 = xc2.get_exx_energy()
    eps0 = xc2.get_eigenvalue_contributions()
    print(world.rank, eps0, xc1.e_skn * Ha)
    assert np.allclose(eps0, xc1.e_skn * Ha)
    assert np.allclose(e0, e[0] + e[1])
    ecv, evv, v_skn = xc1.test()
    assert np.allclose(e0, ecv + evv)
    assert np.allclose(v_skn, eps0)


def main():
    i = 0
    for spinpol in [False, True]:
        for setup in ['ae', 'paw']:
            for symmetry in ['off', {}]:
                for kpts in [(1, 1, 1),
                             (1, 1, 2),
                             (1, 1, 3),
                             (1, 1, 4),
                             (2, 2, 1),
                             [(0, 0, 0.5)],
                             [(0, 0, 0), (0, 0, 0.5)]]:
                    atoms = test(kpts, setup, spinpol, symmetry)
                    for xc in ['EXX', 'PBE0', 'HSE06']:
                        print(spinpol, setup, symmetry, kpts, xc,
                              len(atoms.calc.wfs.mykpts))
                        check(atoms, xc, i)
                        i += 1


main()
