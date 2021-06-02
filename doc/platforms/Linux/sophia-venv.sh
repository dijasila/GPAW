#!/usr/bin/bash
set -e  # stop if there are errors
NAME=$1
FOLDER=$PWD

echo '
export EASYBUILD_PREFIX=/groups/physics/modules
module use $EASYBUILD_PREFIX/modules/all
module purge
unset PYTHONPATH
module load matplotlib
module load spglib-python
module load libxc
module load libvdwxc
module load ScaLAPACK
module load GPAW-setups
' > modules.sh
. modules.sh

python3 -m venv $NAME
cd $NAME
VENV=$PWD
. bin/activate
PIP="python3 -m pip"
$PIP install --upgrade pip -qq

mv bin/activate old
mv ../modules.sh bin/activate
cat old >> bin/activate
rm old

git clone https://gitlab.com/ase/ase.git
$PIP install -e ase

git clone https://gitlab.com/asr-dev/asr.git
$PIP install -e asr

git clone https://gitlab.com/gpaw/gpaw.git
echo "
from os import environ
from pathlib import Path
scalapack = True
fftw = True
libraries = ['openblas', 'fftw3', 'readline', 'gfortran',
             'scalapack', 'xc', 'vdwxc']
libxc = Path(environ['EBROOTLIBXC'])
include_dirs.append(libxc / 'include')
libvdwxc = Path(environ['EBROOTLIBVDWXC'])
include_dirs.append(libvdwxc / 'include')
library_dirs = environ['LD_LIBRARY_PATH'].split(':')
" > gpaw/siteconfig.py
pip install -e gpaw

$PIP install myqueue graphviz qeh ase-ext

gpaw install-data --basis --version=20000 . --no-register
export GPAW_SETUP_PATH=$GPAW_SETUP_PATH:$VENV/gpaw-basis-pvalence-0.9.20000
echo "export GPAW_SETUP_PATH=$GPAW_SETUP_PATH" >> bin/activate

ase completion >> bin/activate
gpaw completion >> bin/activate
mq completion >> bin/activate
$PIP completion --bash >> bin/activate

echo '
if [[ $SLURM_SUBMIT_DIR ]]; then
    export MPLBACKEND=Agg
else
    export MPLBACKEND=TkAgg
fi' >> bin/activate

mq --version
ase info
gpaw test
