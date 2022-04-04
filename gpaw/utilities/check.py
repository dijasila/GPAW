from __future__ import annotations

import argparse
import binascii
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
from gpaw.calculator import GPAW as OldGPAW
from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW as NewGPAW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repeat', default='1,1,1')
    parser.add_argument('-p', '--pbc')
    parser.add_argument('-V', '--vacuum', default=0.0, type=float)
    parser.add_argument('-M', '--magmoms')
    parser.add_argument('-k', '--kpts', default='2.0')
    parser.add_argument('-m', '--mode', default='pw')
    parser.add_argument('-C', '--code', default='new')
    parser.add_argument('-c', '--cores', default='1')
    parser.add_argument('--hex', action='store_true')
    parser.add_argument('system')
    args = parser.parse_intermixed_args()

    if args.hex:
        run(json.loads(binascii.unhexlify(args.system)))
        return 0

    params = {
        'system': args.system,
        'repeat': [int(r) for r in args.repeat.split(',')],
        'pbc': None if args.pbc is None else [
            bool(int(p)) for p in args.pbc.split(',')],
        'vacuum': args.vacuum,
        'magmoms': None if args.magmoms is None else [
            float(m) for m in args.magmoms.split(',')],
        'mode': args.mode,
        'kpts': ([int(k) for k in args.kpts.split(args.kpts)]
                 if ',' in args.kpts else float(args.kpts))}

    folder = Path(args.system)
    if not folder.is_dir():
        folder.mkdir()

    calculations = []
    for code in args.code.split(','):
        for cores in args.cores.split(','):
            input = {**params,
                     'cores': [int(c) for c in cores.split(',')],
                     'code': code}
            hash = hashlib.md5(json.dumps(input).encode()).hexdigest()[:8]
            result_file = folder / f'{hash}.json'
            if not result_file.is_file():
                input['hash'] = hash
                result = run(input)

            result = json.loads(result_file.read_text())

            calculations.append(result)
            print(hash, result['energy'])

    e0 = None
    f0 = None
    for result in calculations:
        e = result['energy']
        f = np.array(result['forces'])
        if e0 is None:
            e0 = e
            f0 = f
        de = result['energy'] - e0
        df = abs(result['forces'] - f0).max()
        cores = 'x'.join(f'{c}' for c in result['cores'])
        code = result['code']
        print(f'{cores:6} {code} {de:10.6f} {df:10.6f}')

    # 'force_complex_dtype': True,
    # kwargs['symmetry'] = {'point_group': False}
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


def run_system(system: str,
               repeat: Sequence[int] = (1, 1, 1),
               pbc: Sequence[bool] = None,
               vacuum: float = 0.0,
               magmoms: list[float] = None,
               mode: str = 'pw',
               kpts: Sequence[int] | float = 2.0,
               code: str = 'new',
               cores: Sequence[int] = None,
               parameters: dict[str, Any] = None,
               hash='42') -> dict[str, Any]:
    atoms = systems[system]()
    tag = f'{system}/{hash}'

    atoms = atoms.copy()
    if sum(repeat) > 3:
        atoms = atoms.repeat()
    if pbc is not None:
        atoms.pbc = pbc
    if vacuum:
        atoms.center(vacuum=vacuum, axes=[a
                                          for a, p in enumerate(atoms.pbc)
                                          if not p])
    if magmoms is not None:
        atoms.set_initial_magnetic_moments(
            magmoms * (len(atoms) // len(magmoms)))

    parameters = {'mode': mode,
                  'kpts': kpts if not isinstance(kpts, float) else {
                      'density': kpts},
                  'txt': f'{tag}.txt'}
    if code == 'new':
        calc = NewGPAW(**parameters)
    else:
        calc = OldGPAW(**parameters)

    atoms.calc = calc

    t1 = time()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    t2 = time()

    result = {'time': t2 - t1,
              'energy': energy,
              'forces': forces.tolist()}

    calc.write(f'{tag}.gpw', mode='all')

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
