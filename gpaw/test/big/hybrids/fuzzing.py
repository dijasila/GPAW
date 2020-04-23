import random
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
from gpaw.kpt_descriptor import kpts2sizeandoffsets

systems = {}


def system(parameters):
    def wrapper(func):
        systems[func.__name__] = (func, parameters)
        return func


@system([(0.5, 1.2), (1.0, 3.0), (0, 1), (0, 1), (0, 1)])
def h2(d, v, *pbc):
    atoms = Atoms('H2', [(0, 0, 0), (0, d, 0)], pbc=pbc)
    atoms.center(vacuum=v)
    return atoms


errors = [
    ('h2', (0.8, 2.0), (0, 1), 1.0, (0, 0), 'lcao', 'LDA', '')]


def fuzz():
    name = random.choise(list(systems))
    func, parameters = systems[name]
    params = [random.uniform(mn, mx)
              if isinstance(mn, float) else
              random.randint(mn, mx)
              for mn, mx in parameters]
    atoms = func(*params)
    force_indices = (random.randint(0, len(atoms) - 1), random.randint(0, 2))
    kptdens = random.uniform(1.0, 3.0)
    (n1, n2, n3), _ = kpts2sizeandoffsets(density=kptdens, atoms=atoms)
    nspins = 1 + int(atoms.get_initial_magnetic_moments().any())
    spin = random.randint(0, nspins - 1)
    kpt = random.randint(0, n1 * n2 * n3 - 1)
    mode = random.choice(['fd', 'lcao', 'pw'])
    if mode == 'pw':
        xc1 = random.choice(['LDA', 'PBE', 'PBE0', 'HSE06'])
        xc2 = random.choice(['EXX', 'PBE0', 'HSE06'])
    else:
        xc1 = random.choice(['LDA', 'PBE'])
        xc2 = ''
    return (name, params, force_indices, kptdens, spin, kpt, mode, xc1, xc2)


def test(name, params, force_indices, kptdens, spin, kpt, mode, xc1, xc2):
    func, _ = systems[name]
    atoms = func(*params)

    hybrid = '0' in xc1

    kwargs = dict(mode=mode,
                  kpts={'density': kptdens},
                  txt='fuzz.txt',
                  xc=xc1)
    if hybrid:
        kwargs.update(dict(mode={'name': 'pw', 'force_complex_dtype': True},
                           parallel={'band': 1, 'kpt': 1}))
        if xc1 == 'PBE0':
            kwargs['xc'] = ...

    atoms.calc = GPAW(**kwargs)

    a, c = force_indices
    eps = 0.01
    atoms.positions[a, c] -= eps
    e1 = a.get_potential_energy()
    atoms.positions[a, c] += 2 * eps
    e2 = a.get_potential_energy()
    f0 = (e1 - e2) / (2 * eps)

    atoms.positions[a, c] -= eps
    atoms.calc = GPAW(**kwargs)
    f = atoms.get_forces()[a, c]

    c = atoms.calc
    e = non_self_consistent_energy(c, xc)
    c.write('c.gpw', 'all')
    eps = non_self_consistent_eigenvalues(c, xc, 0, 2, snapshot=f'{i}.json')
    #eps = non_self_consistent_eigenvalues('c.gpw', xc, 0, 2,
    #                                      snapshot=f'{i}.json')
    # xc1.calculate_eigenvalues0(0, 2, None)
    # e1, v1, v2 = non_self_consistent_eigenvalues(c, xc, 0, 2, None,
    #                                             f'{i}.txt')
    if world.size > 1:
        c.write('tmp.gpw', 'all')
        c = GPAW('tmp.gpw', communicator=serial_comm, txt=None)

    xc2 = EXX(c, xc=xc, bands=(0, 2), txt=None)
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
