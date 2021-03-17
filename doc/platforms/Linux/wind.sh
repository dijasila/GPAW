#!/usr/bin/bash
set -e  # stop if there are errors
NAME=$1
FOLDER=$PWD

echo "
module purge
unset PYTHONPATH
module load Python/3.8.2-GCCcore-9.3.0
module load ScaLAPACK
module load FFTW
" > modules.sh
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

# git clone https://gitlab.com/ase/ase.git
$PIP install -e ~/ase

$PIP install myqueue graphviz qeh ase-ext

# git clone https://gitlab.com/gpaw/gpaw.git
pip install -e ~/gpaw

cd $VENV
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
