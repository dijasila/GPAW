from __future__ import annotations

import random
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import time
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.units import Ha, Bohr
from gpaw.calculator import GPAW as OldGPAW
from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW as NewGPAW


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repeat', default='1,1,1')
    parser.add_argument('-p', '--pbc')
    parser.add_argument('-V', '--vacuum', default=0.0, type=float)
    parser.add_argument('-M', '--magmoms')
    parser.add_argument('-k', '--kpts', default='2.0')
    parser.add_argument('-m', '--mode', default='pw')
    parser.add_argument('-C', '--code', default='new')
    parser.add_argument('-c', '--cores', default='1')
    parser.add_argument('-s', '--symmetry', default='all')
    parser.add_argument('-f', '--complex')
    parser.add_argument('-i', '--ignore-cache', action='store_true')
    parser.add_argument('--hex', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--fuzz', action='store_true')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('system', nargs='*')
    args = parser.parse_intermixed_args()

    if args.hex:
        run(json.loads(bytes.fromhex(args.system)))
        return 0

    many = args.all or args.fuzz

    if many:
        system_names = args.system or list(systems)
        args.repeats = args.repeat or '1x1x1,2x1x1'
        args.vacuums = args.vacuum or '0.0,2.0,3.0'
        args.pbc = args.pbc or '0x0x0,1x1x1'
        args.mode = args.mode or 'pw,lcao,fd'
        args.code = args.code or 'new,old'
        args.core = args.core or '1,2,3,4'
        args.kpts = args.kpts or '2.0,3.0'
    else:
        system_names = args.system
        args.repeats = args.repeat or '1x1x1'
        args.vacuums = args.vacuum or '0.0'
        args.pbc = args.pbc or '0x0x0'
        args.mode = args.mode or 'pw'
        args.code = args.code or 'new'
        args.core = args.core or '1'
        args.kpts = args.kpts or '2.0'

    repeats = [[int(r) for r in rrr.split('x')]
               for rrr in repeats.split(',')]
    vacuums = [float(v) for v in vacuums.split(',')]
    pbcs = [[bool(int(p)) for p in ppp.split('x')]
            for ppp in pbc.split(',')]

    magmoms = None if args.magmoms is None else [
        float(m) for m in args.magmoms.split(',')]

    modes = args.modes.splir(',')
    codes = args.codes.split(',')
    cores = [int(c) for c in args.codes.split(',')]
    kpts = [[int(k) for k in args.kpts.split(',')]
            if ',' in args.kpts else float(args.kpts)]

    # 'force_complex_dtype': args.complex},

    if args.fuzz:
        def pick(choises):
            return [random.choice(choises)]
    else:
        def pick(choises):
            return choises

    count = 0
    calculations = {}
    while True:
        for atoms, dct in create_systems(system_names,
                                         repeats,
                                         vacuums,
                                         pbcs,
                                         magmoms,
                                         pick):
            for params in create_parameters(atoms,
                                            modes,
                                            kpts,
                                            codes,
                                            pick):
                input = [dct, params]
                hash = hashlib.md5(json.dumps(input).encode()).hexdigest()[:8]

                for extra in create_extra_parematers(atoms,
                                                     cores,
                                                     pick):
                    cores = extra['cores']
                    tag = f'{hash}-{cores}'
                    params.update(extra)

                    if not args.count:
                        continue

                    run(atoms, **params, tag=tag)

                    count += 1

        if not args.fuzz:
            break



                calculations.append(result)
                print(hash, result['energy'])

    e0 = None
    f0 = None
    for result in calculations:
        e = result['energy']
        f = result['forces']
        if e0 is None:
            e0 = e
        if f0 is None and f is not None:
            f0 = np.array(f)
        de = result['energy'] - e0
        if f is None or f0 is None:
            df = '?'
        else:
            error = abs(result['forces'] - f0).max()
            df = f'{error:10.6f}'
        cores = 'x'.join(f'{c}' for c in result['cores'])
        code = result['code']
        print(f'{cores:6} {code} {de:10.6f} {df}')

    return 0


def run(input):
    ncores = np.prod(input['cores'])
    if ncores == world.size:
        result = run_system(**input)
        result = {**result, **input}
        if world.rank == 0:
            result_file = Path(f'{input["system"]}/{input["hash"]}.json')
            result_file.write_text(json.dumps(result, indent=2))
    else:
        hex = json.dumps(input).encode().hex()
        subprocess.run(
            f'mpiexec -np {ncores} {sys.executable} -m gpaw.utilities.check '
            f'--hex {hex}',
            shell=True,
            check=True,
            env=os.environ)
    folder = Path(args.system)
    if not folder.is_dir():
        folder.mkdir()

                result_file = folder / f'{hash}.json'
                if not result_file.is_file() or args.ignore_cache:
                    input['hash'] = hash
                    result = run(input)

                result = json.loads(result_file.read_text())


def create_systems(system_names,
                   repeats,
                   vacuums,
                   pbcs,
                   magmoms: list[float] = None,
                   pick=pick_all) -> tuple[Atoms, dict[str, Any]]:
    for name in pick(system_names):
        atoms = systems[name]()
        for repeat in pick(repeats):
            if sum(repeat) > 3:
                atoms = atoms.repeat()
            for vacuum in pick(vacuums):
                if vacuum:
                    atoms = atoms.copy()
                    atoms.center(vacuum=vacuum,
                                 axis=[a
                                       for a, p in enumerate(atoms.pbc)
                                       if not p])
                for pbc in pick(pbcs):
                    if pbc:
                        atoms = atoms.copy()
                        atoms.pbc = pbc
                    if magmoms is not None:
                        atoms.set_initial_magnetic_moments(
                            magmoms * (len(atoms) // len(magmoms)))
                    dct = {'name': name,
                           'repeat': repeat,
                           'vacuum': vacuum,
                           'pbc': pbc}
                    yield atoms, dct


def create_parameters(atoms: Atoms,
                      mode: str = 'pw',
                      kpts: Sequence[int] | float = 2.0,
                      code: str = 'new',
                      cores: Sequence[int] = None,
                      symmetry: str = 'all',
                      pick_random: bool = False) -> dct[str, Any]:

    parameters: dict[str, Any] = {
        'mode': mode,
        'kpts': kpts if not isinstance(kpts, float) else {
            'density': kpts},
        'txt': f'{tag}.txt'}
    if symmetry != 'all':
        parameters['symmetry'] = 'off'


def run(atoms, paramaters):
    if code == 'new':
        calc = NewGPAW(**parameters)
    else:
        calc = OldGPAW(**parameters)

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

    calc.write(f'{tag}.gpw', mode='all')

    calculation = NewGPAW(f'{tag}.gpw').calculation

    energy2 = calculation.results['energy'] * Ha
    assert abs(energy2 - energy) < 1e-14, (energy2, energy)

    if forces is not None:
        forces2 = calculation.results['forces'] * Ha / Bohr
        assert abs(forces2 - forces).max() < 1e-14

    # ibz_index = atoms.calc.wfs.kd.bz2ibz_k[p.kpt]
    # eigs = atoms.calc.get_eigenvalues(ibz_index, p.spin)

    return result


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
