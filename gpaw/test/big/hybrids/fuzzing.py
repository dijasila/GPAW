import sys
import random
from types import SimpleNamespace as Parameters
from typing import List

from ase import Atoms
from ase.build import bulk
# from ase.units import Ha

from gpaw import GPAW
from gpaw.hybrids import HybridXC
from gpaw.hybrids.energy import non_self_consistent_energy
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw.xc.exx import EXX
from gpaw.mpi import world
from gpaw.kpt_descriptor import kpts2sizeandoffsets

systems = {}


def system(*parameters):
    def wrapper(func):
        systems[func.__name__] = (func, parameters)
        return func
    return wrapper


@system((0.6, 1.0), (2.0, 3.0), (0, 1), (0, 1), (0, 1))
def h2(d, v, *pbc):
    atoms = Atoms('H2', [(0, 0, 0), (0, d, 0)], pbc=pbc)
    atoms.center(vacuum=v)
    return atoms


@system((5.2, 5.6), (0, 1))
def si(a, displace):
    atoms = bulk('Si', a=a)
    if displace:
        atoms.positions[0, 0] += 0.05
    return atoms


@system()
def fe():
    atoms = bulk('Fe')
    atoms.set_initial_magnetic_moments([2.3])
    return atoms


errors = [
    Parameters(args=[0.9942365133709256, 1.5629913036448007, 1, 0, 0],
               force_indices=(1, 1), kpt=0, kptdens=2.2065524836825006,
               mode='lcao', name='h2', spin=0, xc1='LDA', xc2='')]


def fuzz():
    name = random.choice(list(systems))
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
    xc2 = ''
    if mode == 'pw':
        xc1 = random.choice(['LDA', 'PBE', 'PBE0', 'HSE06'])
        if '0' not in xc1:
            xc2 = random.choice(['EXX', 'PBE0', 'HSE06'])
    else:
        xc1 = random.choice(['LDA', 'PBE'])
    return Parameters(name=name,
                      args=params,
                      force_indices=force_indices,
                      kptdens=kptdens,
                      spin=spin,
                      kpt=kpt,
                      mode=mode,
                      xc1=xc1,
                      xc2=xc2)


def test(parameters: Parameters, out: List) -> None:
    p = parameters
    func, _ = systems[p.name]
    atoms = func(*p.args)

    hybrid = '0' in p.xc1

    kwargs = dict(mode=p.mode,
                  kpts={'density': p.kptdens},
                  txt='fuzz.txt',
                  xc=p.xc1)
    if hybrid:
        kwargs.update(dict(mode={'name': 'pw', 'force_complex_dtype': True},
                           parallel={'band': 1, 'kpt': 1},
                           eigensolver={'name': 'dav', 'niter': 1}))
        if p.xc1 == 'PBE0':
            kwargs['xc'] = HybridXC('PBE0')

    kwargs['convergence'] = {'forces': 0.01}
    atoms.calc = GPAW(**kwargs)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    ibz_index = atoms.calc.wfs.kd.bz2ibz_k[p.kpt]
    eigs = atoms.calc.get_eigenvalues(ibz_index, p.spin)
    out += [energy, forces, eigs]
    if p.xc2:
        energy2 = non_self_consistent_energy(atoms.calc, p.xc2).sum()
        eigs2 = non_self_consistent_eigenvalues(
            atoms.calc,
            p.xc2,
            kpt_indices=[ibz_index])
        out += [energy2, eigs2]
        if p.xc1 == p.xc2:
            assert abs(energy - energy2) < 1e-4
            assert abs(eigs - sum(eigs2)[p.spin, 0]).max() < 1e-5

        atoms.calc.write('fuzz.gpw', mode='all')

        exx = EXX('fuzz.gpw',
                  xc=p.xc2,
                  kpts=[ibz_index],
                  bands=(0, atoms.calc.wfs.bd.nbands),
                  txt=None)
        exx.calculate()
        energy2b = exx.get_total_energy()
        eigs2b = exx.get_eigenvalue_contributions()
        out += [energy2b, eigs2b]
        assert abs(energy2 - energy2b) < 1e-4
        assert abs(eigs2b - eigs2[2]).max() < 1e-5

    kwargs['symmetry'] = {'point_group': False}
    atoms.calc = GPAW(**kwargs)
    energy3 = atoms.get_potential_energy()
    forces3 = atoms.get_forces()
    ibz_index = atoms.calc.wfs.kd.bz2ibz_k[p.kpt]
    eigs3 = atoms.calc.get_eigenvalues(ibz_index, p.spin)
    out += [energy3, forces3, eigs3]
    assert abs(energy - energy3) < 1e-4
    assert abs(forces - forces3).max() < 0.01
    assert abs(eigs - eigs3).max() < 1e-5
    if p.xc2:
        energy4 = non_self_consistent_energy(atoms.calc, p.xc2).sum()
        eigs4 = non_self_consistent_eigenvalues(
            atoms.calc,
            p.xc2,
            kpt_indices=[ibz_index])
        out += [energy4, eigs4]
        assert abs(energy4 - energy2) < 1e-4
        assert abs(sum(eigs4) - sum(eigs2)).max() < 1e-5

    a, c = p.force_indices
    kwargs['symmetry'] = {}
    eps = 0.01
    nv = atoms.calc.setups.nvalence
    kwargs['convergence'] = {'energy': 0.01 * 2 * eps / nv}
    atoms.calc = GPAW(**kwargs)
    atoms.positions[a, c] -= eps
    e1 = atoms.get_potential_energy()
    atoms.positions[a, c] += 2 * eps
    e2 = atoms.get_potential_energy()
    force0 = (e1 - e2) / (2 * eps)
    atoms.positions[a, c] -= eps
    out += [force0]

    force = forces[a, c]
    if p.mode == 'lcao' and not atoms.pbc.all():
        # Skip test (basis functions outside box)
        pass
    else:
        assert abs(force - force0) < 0.01


def generate_parameters():
    for p in errors:
        yield p
    while True:
        yield fuzz()


if __name__ == '__main__':
    random.seed(117)
    for p in generate_parameters():
        if world.rank == 0:
            print(p, flush=True)
        out = []
        try:
            test(p, out)
        except Exception as ex:
            if world.rank == 0:
                print(p, ex, out, file=sys.stderr, flush=True)
