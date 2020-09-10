#!/usr/bin/bash
# Install gpaw, ase, ase-ext, spglib, sklearn and myqueue on Niflheim in a venv

set -e  # stop if there are errors

FOLDERNAME=$1
mkdir -p $FOLDERNAME
cd $FOLDERNAME
FOLDER=$PWD

echo "module purge
module load GPAW-setups/0.9.20000
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4
module load libxc/4.3.4-GCC-8.3.0
module load libvdwxc/0.4.0-foss-2019b" > modules.sh
. modules.sh

# Create venv:
python3 -m venv venv --prompt=$FOLDERNAME --system-site-packages
cd venv
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
cd $FOLDER
git clone https://gitlab.com/ase/ase.git
$PIP install -e ase/

$PIP install myqueue graphviz pytest-xdist

CMD="cd $VENV &&
     . bin/activate &&
     pip install spglib sklearn ase-ext"
echo $CMD
ssh fjorm $CMD

# Install GPAW:
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp doc/platforms/Linux/Niflheim/siteconfig-$TOOLCHAIN.py siteconfig.py
for HOST in fjorm svol thul slid
do
  CMD="cd $FOLDER &&
       . venv/bin/activate &&
       pip install -e gpaw -v > $HOST.out 2>&1"
  echo Compiling GPAW on $HOST
  echo $CMD
  ssh $HOST $CMD
done
(cd build && ln -sf lib.linux-x86_64-{sandybridge,ivybridge}-3.7)
rm -r build/temp.linux-x86_64-*
rm _gpaw.*.so
cd ..

# Install extra basis-functions:
gpaw install-data --basis --version=20000 . --no-register
export GPAW_SETUP_PATH=$GPAW_SETUP_PATH:$FOLDER/gpaw-basis-pvalence-0.9.20000
cd $VENV
echo "export GPAW_SETUP_PATH=$GPAW_SETUP_PATH" >> bin/activate

# Tab completion:
ase completion >> bin/activate
gpaw completion >> bin/activate
mq completion >> bin/activate

# Set matlplotlib backend:
echo "unset MPLBACKEND" >> bin/activate

# Run tests:
mq --version
ase info
gpaw test
