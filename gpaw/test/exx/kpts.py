import numpy as np
from ase import Atoms
from ase.units import Ha
from gpaw import GPAW, PW
from gpaw.hybrids import HybridXC
from gpaw.xc.exx import EXX


def test(kpts, setup, spinpol, symmetry):
    a = Atoms('H2', cell=(3, 3, 3), pbc=1, positions=[[0, 0, 0], [0, 0, 0.75]])
    a.calc = GPAW(mode=PW(100, force_complex_dtype=True),
                  setups=setup,
                  kpts=kpts,
                  spinpol=spinpol,
                  symmetry=symmetry,
                  txt=None,
                  xc='PBE')
    a.get_potential_energy()
    return a


def check(atoms, xc):
    xc1 = HybridXC(xc)
    c = atoms.calc
    xc1.initialize(c.density, c.hamiltonian, c.wfs, c.occupations)
    xc1.set_positions(c.spos_ac)
    e = xc1.calculate_energy()
    # print(e)
    xc1.calculate_eigenvalues(0, 2, None)
    # print('A', xc1.e_skn * Ha)

    xc2 = EXX(c, xc=xc, bands=(0, 2), txt=None)
    xc2.calculate()
    e0 = xc2.get_exx_energy()
    eps0 = xc2.get_eigenvalue_contributions()
    # print('B', eps0)
    assert np.allclose(eps0, xc1.e_skn * Ha)
    # print(e0, e[0] + e[1])
    assert np.allclose(e0, e[0] + e[1])
    # print(xc1.description)
    ecv, evv, v_skn = xc1.test()
    # print('C', v_skn)
    assert np.allclose(e0, ecv + evv)
    assert np.allclose(v_skn, eps0)


def main():
    for spinpol in [False,
                    True]:
        for setup in ['ae',
                      'paw']:
            for symmetry in ['off',
                             {}]:
                for kpts in [
                    (1, 1, 1),
                    (1, 1, 2),
                    (1, 1, 3),
                    (1, 1, 4),
                    (2, 2, 1),
                    [(0, 0, 0.5)],
                    [(0, 0, 0), (0, 0, 0.5)]]:
                    atoms = test(kpts, setup, spinpol, symmetry)
                    for xc in ['EXX',
                               #'PBE0', 'HSE06'
                               ]:
                        print(spinpol, setup, symmetry, kpts, xc,
                              len(atoms.calc.wfs.mykpts))
                        check(atoms, xc)


main()
