#!/usr/bin/bash
# Install gpaw, ase, ase-ext, spglib, sklearn and myqueue on Niflheim in a venv

set -e  # stop if there are errors

NAME=$1
USAGE="Usage: $0 foldername [intel]"

if [[ $# -ne 2 && $# -ne 1 ]]; then
    echo "Wrong number of arguments, expected 1 or 2, got $#"
    echo $USAGE
    exit 1
fi

if [[ "$2" == "intel" || "$2" == "Intel" ]]; then
    TCHAIN=intel
    echo "Using Intel toolchain."
else
    if [[ $# -eq 2 ]]; then
	echo "Illegal second argument, only 'intel' is allowed.  Got $2"
	echo $USAGE
	exit 1
    fi
    TCHAIN=foss
fi

FOLDER=$PWD

# Some modules are not yet loaded.  Once they are installed, this
# script is changed to load them instead of pip-installing them.

echo "
module purge
unset PYTHONPATH
module load GPAW-setups/0.9.20000
module load matplotlib/3.3.3-${TCHAIN}-2020b
module load spglib-python/1.16.0-${TCHAIN}-2020b
module load scikit-learn/0.23.2-${TCHAIN}-2020b
module load pytest-xdist/2.1.0-GCCcore-10.2.0
module load Wannier90/3.1.0-${TCHAIN}-2020b
" > modules.sh

if [[ "$TCHAIN" == "intel" ]]; then
echo "module load libxc/4.3.4-iccifort-2020.4.304" >> modules.sh
else
echo "module load libxc/4.3.4-GCC-10.2.0" >> modules.sh
echo "module load libvdwxc/0.4.0-foss-2020b" >> modules.sh
fi

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

$PIP install myqueue graphviz

CMD="cd $VENV &&
     . bin/activate &&
     pip install ase-ext"
echo $CMD
ssh fjorm $CMD

# Install GPAW:
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp doc/platforms/Linux/Niflheim/siteconfig-${TCHAIN}.py siteconfig.py
for HOST in fjorm svol thul slid
do
  CMD="cd $VENV &&
       . bin/activate &&
       pip install -e gpaw -v > $HOST.out 2>&1"
  echo Compiling GPAW on $HOST
  echo $CMD
  ssh $HOST $CMD
done
(cd build && ln -sf lib.linux-x86_64-{sandybridge,ivybridge}-3.8)
rm -r build/temp.linux-x86_64-*
rm _gpaw.*.so

# Install extra basis-functions:
cd $VENV
gpaw install-data --basis --version=20000 . --no-register
export GPAW_SETUP_PATH=$GPAW_SETUP_PATH:$VENV/gpaw-basis-pvalence-0.9.20000
echo "export GPAW_SETUP_PATH=$GPAW_SETUP_PATH" >> bin/activate

# Tab completion:
ase completion >> bin/activate
gpaw completion >> bin/activate
mq completion >> bin/activate

# Set matplotlib backend:
echo "export MPLBACKEND=TkAgg" >> bin/activate

# Run tests:
mq --version
ase info
gpaw test
