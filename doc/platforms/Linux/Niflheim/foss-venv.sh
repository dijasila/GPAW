#!/usr/bin/bash
# Install gpaw, ase, ase_ext, spglib, and myqueue on Niflheim in a venv

######deactivate > /dev/null 2>&1
set -e  # stop if there are errors

FOLDERNAME=$1
mkdir -p $FOLDERNAME
cd $FOLDERNAME

module purge
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4
module load libxc/4.3.4-GCC-8.3.0
module load libvdwxc/0.4.0-foss-2019b

# Create venv:
python3 -m venv venv --prompt=$FOLDERNAME --system-site-packages
cd venv
VENV=$PWD
. bin/activate
PIP="python3 -m pip"
$PIP install --upgrade pip -qq

# Load modules in activate script:
mv bin/activate .
echo "module purge" > bin/activate
echo "module load matplotlib/3.1.1-foss-2019b-Python-3.7.4" >> bin/activate
echo "module load libxc/4.3.4-GCC-8.3.0" >> bin/activate
echo "module load libvdwxc/0.4.0-foss-2019b" >> bin/activate
echo "module load GPAW-setups/0.9.20000" >> bin/activate

cat activate >> bin/activate
rm activate

# Install MyQueue:
$PIP install myqueue

# Install ASE:
git clone https://gitlab.com/ase/ase.git
$PIP install -e ase/

# Install ASE C-extension (ase_ext):
URL="git+https://gitlab.com/ase/ase_ext.git@master"
CMD="cd $VENV && . bin/activate && pip install $URL -v > ase_ext.out 2>&1"
echo $CMD
ssh fjorm $CMD

# Install spglib:
CMD="cd $VENV && . bin/activate && pip install spglib"
echo $CMD
ssh fjorm $CMD

# Install GPAW:
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp doc/platforms/Linux/Niflheim/el7-foss.py siteconfig.py
for HOST in fjorm svol thul slid
do
  CMD="cd $VENV && . bin/activate && pip install -e gpaw -v > $HOST.out 2>&1"
  echo Compiling GPAW on $HOST
  echo $CMD
  ssh $HOST $CMD
done
(cd build && ln -sf lib.linux-x86_64-{sandybridge,ivybridge}-3.6)
rm -r build/temp.linux-x86_64-*
rm _gpaw.*.so
cd ..

# Tab completion:
ase completion >> bin/activate
gpaw completion >> bin/activate
mq completion >> bin/activate

# Set matlplotlib backend:
echo "export MPLBACKEND=Agg" >> bin/activate

# DFTD3:
# export ASE_DFTD3_COMMAND=~/source/dftd3/dftd3

# Run tests:
mq --version
ase info
gpaw test
