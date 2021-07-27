#!/usr/bin/bash
# Install gpaw, ase, ase-ext, spglib, sklearn and myqueue on Juwels in a venv

set -e  # stop if there are errors

NAME=$1
USAGE="Usage: $0 foldername [intel]"
FOLDER=$PWD
ASE_REPO=https://gitlab.com/ase/ase.git
GPAW_REPO=https://gitlab.com/gpaw/gpaw.git

if [[ $# -ne 2 && $# -ne 1 ]]; then
    echo "Wrong number of arguments, expected 1 or 2, got $#"
    echo $USAGE
    exit 1
fi


# Some modules are not yet loaded.  Once they are installed, this
# script is changed to load them instead of pip-installing them.

echo "
module purge
module  load StdEnv/2020
module load Python/3.8.5
module load SciPy-Stack/2020-Python-3.8.5
module load intel-para
module load FFTW/3.3.8
module load libxc/4.3.4
module load ELPA/2020.05.001
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
mv $FOLDER/modules.sh bin/activate
cat old >> bin/activate
rm old

# Install ASE from git:
git clone $ASE_REPO
$PIP install -e ase/

$PIP install myqueue graphviz qeh

CMD="cd $VENV &&
     . bin/activate &&
     pip install ase-ext"
echo $CMD

# Install GPAW:
git clone $GPAW_REPO
cd gpaw
cp doc/platforms/Linux/Juwels/siteconfig_juwel.py siteconfig.py
cd $VENV
. bin/activate
pip install -e gpaw -v > compilation.out

# Install extra basis-functions:
cd $VENV
gpaw install-data .
gpaw install-data --basis --version=20000 . --no-register
export GPAW_SETUP_PATH=$GPAW_SETUP_PATH:$VENV/gpaw-basis-pvalence-0.9.20000
echo "export GPAW_SETUP_PATH=$GPAW_SETUP_PATH" >> bin/activate

# Tab completion:
ase completion >> bin/activate
gpaw completion >> bin/activate
mq completion >> bin/activate
$PIP completion --bash >> bin/activate

# Run tests:
mq --version
ase info
gpaw test
