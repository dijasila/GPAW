#!/usr/bin/bash
# Install gpaw, ase, ase-ext, spglib, sklearn and myqueue on Niflheim in a venv

set -e  # stop if there are errors

FOLDERNAME=$1
TOOLCHAIN=foss
mkdir -p $FOLDERNAME
cd $FOLDERNAME
FOLDER=$PWD

echo "module purge" > modules.sh
echo "module load GPAW-setups/0.9.20000" >> modules.sh

if [ "$TOOLCHAIN" = foss ]; then
    echo "module load matplotlib/3.1.1-foss-2019b-Python-3.7.4" >> modules.sh
    echo "module load libxc/4.3.4-GCC-8.3.0" >> modules.sh
    echo "module load libvdwxc/0.4.0-foss-2019b" >> modules.sh
else
    echo "module load matplotlib/3.1.1-intel-2019b-Python-3.7.4" >> modules.sh
    echo "module load libxc/4.3.4-iccifort-2019.5.281" >> modules.sh
fi

. modules.sh

# Create venv:
python3 -m venv venv --prompt=$FOLDERNAME --system-site-packages
cd venv
VENV=$PWD
. bin/activate
PIP="python3 -m pip"
$PIP install --upgrade pip -qq

# Load modules in activate script:
mv bin/activate .
mv ../modules.sh bin/activate
cat activate >> bin/activate
rm activate

cd $FOLDER

# Install MyQueue:
$PIP install myqueue

# Install ASE:
git clone https://gitlab.com/ase/ase.git
$PIP install -e ase/

# Install ase-ext, spglib and sklearn:
CMD="cd $VENV &&
     . bin/activate &&
     pip install spglib sklearn ase-ext pytest-xdist"
echo $CMD
ssh fjorm $CMD

# Install GPAW:
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp doc/platforms/Linux/Niflheim/el7-$TOOLCHAIN.py siteconfig.py
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
echo "export MPLBACKEND=Agg" >> bin/activate

# DFTD3:
# export ASE_DFTD3_COMMAND=~/source/dftd3/dftd3

# Run tests:
mq --version
ase info
gpaw test
