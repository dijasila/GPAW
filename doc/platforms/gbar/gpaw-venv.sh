#!/usr/bin/bash
# Install gpaw, ase, ase-ext, spglib, sklearn and myqueue on Niflheim in a venv

set -e  # stop if there are errors

NAME=$1
FOLDER=$PWD

# Some modules are not yet loaded.  Once they are installed, this
# script is changed to load them instead of pip-installing them.

echo "
module purge
unset PYTHONPATH
module load dcc-setup/2020-aug
module load python/3.8.5
module load fftw/3.3.8 libxc/4.3.4
module load scalapack/2.1.0
module load openblas/0.3.10
unset CC
" > modules.sh

. modules.sh

# Create venv:
echo "Creating virtual environment $NAME"
python3 -m venv --system-site-packages $NAME
cd $NAME
VENV=$PWD
. bin/activate
PIP="python3 -m pip"
$PIP install --upgrade pip -qq

# Load modules in activate script:
mv bin/activate old
mv ../modules.sh bin/activate
cat old >> bin/activate
rm old

# Install ASE from git:
git clone https://gitlab.com/ase/ase.git
$PIP install -e ase/

$PIP install myqueue graphviz ase-ext

# Install GPAW:
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
echo "
from pathlib import Path
from os import environ as env
scalapack = True
fftw = True
libraries = ['xc', 'openblas', 'fftw3', 'scalapack']
base = Path(env['DCC_SW_PATH']) / env['DCC_SW_CPUTYPE'] / env['DCC_SW_COMPILER']

for p in ['fftw/3.3.8', 'libxc/4.3.4', 'scalapack/2.1.0', 'openblas/0.3.10']:
    lib = base / p / 'lib'
    library_dirs.append(lib)
    extra_link_args.append(f'-Wl,-rpath={lib}')
    include_dirs.append(f'{base / p}/include')
" > siteconfig.py
pip install -e . -v > gpaw.out 2>&1

# Install extra basis-functions:
cd $VENV
export GPAW_SETUP_PATH=~jjmo/PAW/gpaw-setups-0.9.20000/
echo "export GPAW_SETUP_PATH=$GPAW_SETUP_PATH" >> bin/activate

# Tab completion:
ase completion >> bin/activate
gpaw completion >> bin/activate
mq completion >> bin/activate

# Run tests:
mq --version
ase info
gpaw test
