from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from time import time
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.units import Bohr, Ha
from gpaw.calculator import GPAW as OldGPAW
from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW as NewGPAW


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repeat')
    parser.add_argument('-p', '--pbc')
    parser.add_argument('-v', '--vacuum')
    parser.add_argument('-M', '--magmoms')
    parser.add_argument('-k', '--kpts')
    parser.add_argument('-m', '--mode')
    parser.add_argument('-c', '--code')
    parser.add_argument('-n', '--ncores')
    parser.add_argument('-s', '--symmetry')
    parser.add_argument('-f', '--complex')
    parser.add_argument('-i', '--ignore-cache', action='store_true')
    parser.add_argument('--pickle')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--fuzz', action='store_true')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('system', nargs='*')
    args = parser.parse_intermixed_args()

    if args.pickle:
        pckl_file = Path(args.pickle)
        atoms, params, result_file = pickle.loads(pckl_file.read_bytes())
        run2(atoms, params, result_file)
        return 0

    many = args.all or args.fuzz

    if many:
        system_names = args.system or list(systems)
        args.repeat = args.repeat or '1x1x1,2x1x1'
        args.vacuum = args.vacuum or '0.0,2.0,3.0'
        args.pbc = args.pbc or '0,1'
        args.mode = args.mode or 'pw,lcao,fd'
        args.code = args.code or 'new,old'
        args.ncores = args.ncores or '1,2,3,4'
        args.kpts = args.kpts or '2.0,3.0'
    else:
        system_names = args.system
        args.repeat = args.repeat or '1x1x1'
        args.vacuum = args.vacuum or '0.0'
        args.pbc = args.pbc or '0'
        args.mode = args.mode or 'pw'
        args.code = args.code or 'new'
        args.ncores = args.ncores or '1'
        args.kpts = args.kpts or '2.0'

    repeats = [[int(r) for r in rrr.split('x')]
               for rrr in args.repeat.split(',')]
    vacuums = [float(v) for v in args.vacuum.split(',')]
    pbcs = [bool(int(p)) for p in args.pbc.split(',')]

    magmoms = None if args.magmoms is None else [
        float(m) for m in args.magmoms.split(',')]

    modes = args.mode.split(',')
    codes = args.code.split(',')
    ncores_all = [int(c) for c in args.ncores.split(',')]
    kpts = [[int(k) for k in kpt.split(',')] if ',' in kpt else
            float(kpt)
            for kpt in args.kpts.split(',')]

    # 'force_complex_dtype': args.complex},
    # spinpol

    if args.fuzz:
        def pick(choises):
            return [random.choice(choises)]
    else:
        def pick(choises):
            return choises

    count = 0
    calculations = {}
    while True:
        for atoms, atag in create_systems(system_names,
                                          repeats,
                                          vacuums,
                                          pbcs,
                                          magmoms,
                                          pick):
            for params, ptag in create_parameters(modes,
                                                  kpts,
                                                  pick):
                tag = atag + ' ' + ptag

                for extra, xtag in create_extra_parameters(codes,
                                                           ncores_all,
                                                           pick):
                    params.update(extra)

                    if not args.count:
                        result = run(atoms,
                                     params,
                                     tag + ' ' + xtag,
                                     args.ignore_cache)
                        check(tag, result, calculations)

                    count += 1

        if not args.fuzz:
            break


def run(atoms, params, tag, ignore_cache=False):
    params = params.copy()
    name, tag = tag.split(' -', 1)
    print(f'{name:3} {tag}:', end='', flush=True)
    tag = tag.replace(' ', '')
    folder = Path(name)
    if not folder.is_dir():
        folder.mkdir()
    result_file = folder / f'{tag}.json'
    if not result_file.is_file() or ignore_cache:
        print(' ...', end='')
        ncores = params.pop('ncores')
        if ncores == world.size:
            result = run2(atoms, params, result_file)
        else:
            pckl_file = result_file.with_suffix('.pckl')
            pckl_file.write_bytes(pickle.dumps((atoms, params, result_file)))
            subprocess.run(
                ['mpiexec', '-np', str(ncores),
                 sys.executable, '-m', 'gpaw.test.fuzz',
                 '--pickle', str(pckl_file)],
                check=True,
                env=os.environ)
            result, _ = json.loads(result_file.read_text())
    else:
        print('    ', end='')
        result, _ = json.loads(result_file.read_text())
    print(f' {result["energy"]:14.6f} eV, {result["time"]:9.3f} s')
    return result


def run2(atoms, params, result_file):
    params = params.copy()
    params['txt'] = str(result_file.with_suffix('.txt'))
    code = params.pop('code')
    if code == 'new':
        calc = NewGPAW(**params)
    else:
        calc = OldGPAW(**params)

    atoms.calc = calc

    t1 = time()
    energy = atoms.get_potential_energy()
    try:
        forces = atoms.get_forces()
    except NotImplementedError:
        forces = None

    t2 = time()

    result = {'time': t2 - t1,
              'energy': energy,
              'forces': None if forces is None else forces.tolist()}

    gpw_file = result_file.with_suffix('.gpw')
    calc.write(gpw_file, mode='all')

    calculation = NewGPAW(gpw_file).calculation

    energy2 = calculation.results['energy'] * Ha
    assert abs(energy2 - energy) < 1e-14, (energy2, energy)

    if forces is not None:
        forces2 = calculation.results['forces'] * Ha / Bohr
        assert abs(forces2 - forces).max() < 1e-14

    # ibz_index = atoms.calc.wfs.kd.bz2ibz_k[p.kpt]
    # eigs = atoms.calc.get_eigenvalues(ibz_index, p.spin)

    if world.rank == 0:
        result_file.write_text(json.dumps([result, params], indent=2))

    return result


def check(tag, result, calculations):
    if tag not in calculations:
        calculations[tag] = result
        return

    result0 = calculations[tag]
    e0 = result0['energy']
    f0 = result0['forces']
    e = result['energy']
    f = result['forces']
    error = e - e0
    if abs(error) > 0.0005:
        print('Energy error:', e, e0, error)
        return
    if f0 is None:
        if f is not None:
            calculations[tag]['forces'] = f
        return
    if f is not None:
        error = abs(np.array(f) - f0).max()
        if error > 0.0005:
            print('Force error:', error)


def create_systems(system_names,
                   repeats,
                   vacuums,
                   pbcs,
                   magmoms,
                   pick) -> tuple[Atoms, dict[str, Any]]:
    for name in pick(system_names):
        atoms = systems[name]()
        for repeat in pick(repeats):
            if any(not p and r > 1 for p, r in zip(atoms.pbc, repeat)):
                continue
            ratoms = atoms.repeat(repeat)
            for vacuum in pick(vacuums):
                if vacuum:
                    vatoms = ratoms.copy()
                    axes = [a for a, p in enumerate(atoms.pbc) if not p]
                    if axes:
                        vatoms.center(vacuum=vacuum, axis=axes)
                    else:
                        continue
                else:
                    vatoms = atoms
                for pbc in pick(pbcs):
                    if pbc:
                        if atoms.pbc.all():
                            continue
                        patoms = atoms.copy()
                        patoms.pbc = pbc
                    else:
                        patoms = atoms

                    if magmoms is not None:
                        patoms.set_initial_magnetic_moments(
                            magmoms * (len(patoms) // len(magmoms)))

                    tag = (f'{name} '
                           f'-r{"x".join(str(r) for r in repeat)} '
                           f'-v{vacuum:.1f} '
                           f'-p{int(pbc)}')
                    yield patoms, tag


def create_parameters(modes,
                      kpts_all,
                      pick) -> tuple[dict[str, Any], str]:
    for mode in pick(modes):
        for kpt in pick(kpts_all):
            if isinstance(kpt, float):
                kpts = {'density': kpt}
                ktag = f'-k{kpt:.1f}'
            else:
                kpts = kpt
                ktag = f'-k{"x".join(str(k) for k in kpt)}'
            yield {'mode': mode,
                   'kpts': kpts}, f'-m{mode} {ktag}'


def create_extra_parameters(codes,
                            ncores_all,
                            # symmetries,
                            pick) -> dict[str, Any]:
    for code in pick(codes):
        for ncores in pick(ncores_all):
            yield {'code': code,
                   'ncores': ncores}, f'-c{code} -n{ncores}'

    # if symmetry != 'all':
    #    parameters['symmetry'] = 'off'


systems = {}


def system(func):
    systems[func.__name__] = func
    return func


@system
def h():
    atoms = Atoms('H', magmoms=[1])
    atoms.center(vacuum=2.0)
    return atoms


@system
def h2():
    atoms = Atoms('H2', [(0, 0, 0), (0, 0.75, 0)])
    atoms.center(vacuum=2.0)
    return atoms


@system
def si():
    atoms = bulk('Si', a=5.4)
    return atoms


@system
def fe():
    atoms = bulk('Fe')
    atoms.set_initial_magnetic_moments([2.3])
    return atoms


@system
def li():
    L = 5.0
    atoms = Atoms('Li', cell=[L, L, 1.5], pbc=(0, 0, 1))
    atoms.center()
    return atoms


if __name__ == '__main__':
    raise SystemExit(main())
