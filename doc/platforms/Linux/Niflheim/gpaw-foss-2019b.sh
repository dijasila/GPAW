export GPAW_TOOLCHAIN=foss

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
   module load libxc/4.3.4-GCC-8.3.0
fi

# Load Python and matplotlib
module load matplotlib/3.1.1-$MYPYTHON

# Load default setups, unless GPAW_SETUP_PATH is already set
if [[ -z $GPAW_SETUP_PATH ]]; then
    module load GPAW-setups
fi

export GPAW_MPI_OPTIONS=""
PLATFORM=linux-x86_64-$CPU_ARCH-el7-3.7
export PATH=$GPAW/tools:$GPAW/build/bin.$PLATFORM:$PATH
export PYTHONPATH=$GPAW:$GPAW/build/lib.$PLATFORM:$PYTHONPATH
export PATH=$ASE/bin:$PATH
export PYTHONPATH=$ASE:$PYTHONPATH
