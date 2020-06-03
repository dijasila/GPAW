export GPAW_TOOLCHAIN=intel

if [ -z $GPAW ]; then
    GPAW=~/gpaw
fi
if [ -z $ASE ]; then
    ASE=~/ase
fi

if [[ -z $MYPYTHON ]]; then
    MYPYTHON=${GPAW_TOOLCHAIN}-2019b-Python-3.7.4
fi

# Load libxc unless another version is already loaded
if [[ -z "$EBROOTLIBXC" ]]; then
   module load libxc/4.3.4-iccifort-2019.5.281
fi

# Load Python and matplotlib
module load matplotlib/3.1.1-$MYPYTHON

# Load default setups, unless GPAW_SETUP_PATH is already set
if [[ -z $GPAW_SETUP_PATH ]]; then
    module load GPAW-setups
fi

export PATH=$GPAW/tools:$PATH
export PYTHONPATH=$GPAW:$PYTHONPATH
export PATH=$ASE/bin:$PATH
export PYTHONPATH=$ASE:$PYTHONPATH
