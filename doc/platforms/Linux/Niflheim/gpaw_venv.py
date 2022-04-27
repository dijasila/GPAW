#!/usr/bin/env python3
"""Install gpaw on Niflheim in a virtual environment.

Also installs ase, ase-ext, spglib, sklearn and myqueue.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

module_cmds_all = """\
module purge
unset PYTHONPATH
module load GPAW-setups/0.9.20000
module load matplotlib/3.3.3-{tchain}-2020b
module load spglib-python/1.16.0-{tchain}-2020b
module load scikit-learn/0.23.2-{tchain}-2020b
module load pytest-xdist/2.1.0-GCCcore-10.2.0
module load Wannier90/3.1.0-{tchain}-2020b
"""

module_cmds_tc = {
    'foss': """\
module load libxc/4.3.4-GCC-10.2.0
module load libvdwxc/0.4.0-foss-2020b
""",
    'intel': """\
module load libxc/4.3.4-iccifort-2020.4.304
"""}

activate_extra = """
export GPAW_SETUP_PATH=$GPAW_SETUP_PATH:{venv}/gpaw-basis-pvalence-0.9.20000

# Set matplotlib backend:
if [[ $SLURM_SUBMIT_DIR ]]; then
    export MPLBACKEND=Agg
else
    export MPLBACKEND=TkAgg
"""

dftd3 = """\
mkdir {venv}/DFTD3
cd {venv}/DFTD3
wget chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/dftd3.tgz
tar -xf dftd3.tgz
ssh thul "cd {venv}/DFTD3 && make"
ln -s {venv}/DFTD3/dftd3 {venv}/bin
"""


def run(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    print(cmd)
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def compile_gpaw_c_code(gpaw: Path, activate: Path) -> None:
    # xeon16, xeon24, xeon40:
    for host in ['thul', 'sylg', 'svol', 'surt']:
        run(f'ssh {host} ". {activate} && pip install -q -e {gpaw}"')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('venv', description=__doc__)
    parser.add_argument('--toolchain', default='foss',
                        choices=['foss', 'intel'],
                        help='Default is foss.')
    parser.add_argument('--dftd3', action='store_true',
                        help='Also build DFTD3.')
    parser.add_argument('--recompile', action='store_true',
                        help='Recompile the GPAW C-extensions in an '
                        'exisint venv.')
    args = parser.parse_args()

    venv = Path(args.venv).absolute()
    activate = venv / 'bin/activate'
    gpaw = venv / 'gpaw'

    if args.recompile:
        compile_gpaw_c_code(gpaw, activate)
        return 0

    module_cmds = module_cmds_all.format(tchain=args.toolchain)
    module_cmds += module_cmds_tc[args.toolchain]

    cmds = (' && '.join(module_cmds.splitlines()) +
            f' && python3 -m venv --system-site-packages {args.venv}')
    run(cmds)

    os.chdir(venv)

    activate.write_text(module_cmds +
                        activate.read_text())

    run(f'. {activate} && pip install --upgrade pip -q')

    packages = ['myqueue',
                'graphviz',
                'qeh']
    run(f'. {activate} && pip install -q ' + ' '.join(packages))

    for name in ['ase', 'gpaw']:
        run(f'git clone -q https://gitlab.com/{name}/{name}')

    run(f'. {activate} && pip install -q -e ase/')

    if args.dftd3:
        run(' && '.join(dftd3.format(venv=venv).splitlines()))

    # Compile ase-exc C-extension of old thul so that it works on
    # newer architectures
    run(f'ssh thul ". {activate} && pip install -q ase-ext"')

    # Install GPAW:
    siteconfig = Path(
        f'gpaw/doc/platforms/Linux/Niflheim/siteconfig-{args.toolchain}.py')
    Path('gpaw/siteconfig.py').write_text(siteconfig.read_text())

    compile_gpaw_c_code(gpaw, activate)

    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    for fro, to in [('ivybridge', 'sandybridge'),
                    ('nahelem', 'icelake')]:
        f = gpaw / f'build/lib.linux-x86_64-{fro}-{version}'
        t = gpaw / f'build/lib.linux-x86_64-{to}-{version}'
        f.symlink_to(t)

    for path in gpaw.glob('build/temp.linux-x86_64-*'):
        shutil.rmtree(path)
    for path in gpaw.glob('_gpaw.*.so'):
        path.unlink()

    # Create .pth file to load correct .so file:
    pth = ('import sys, os; '
           'arch = os.environ["CPU_ARCH"]; '
           f"path = f'{venv}/gpaw/build/lib.linux-x86_64-{{arch}}-{version}'; "
           'sys.path.append(path)\n')
    Path(f'lib/python{version}/site-packages/niflheim.pth').write_text(pth)

    # Install extra basis-functions:
    run(f'. {activate} && gpaw install-data --basis --version=20000 '
        f'{venv} --no-register')

    extra = activate_extra.format(venv=venv)
    # Tab completion:
    for cmd in ['ase', 'gpaw', 'mq', 'pip']:
        txt = run(f'{cmd} completion', capture_output=True).stdout
        extra += txt
    activate.write_text(activate.read_text() + extra)

    # Run tests:
    run('mq info && ase info && gpaw test')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
