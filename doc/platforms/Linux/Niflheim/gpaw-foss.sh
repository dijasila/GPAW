

#if [ -f /etc/bashrc ]; then
#    . /etc/bashrc
#fi

# Inform the compile script of which toolchain to use.
export GPAW_TOOLCHAIN=foss

if [ -z $GPAW ]; then
    GPAW=~/gpaw
fi
if [ -z $ASE ]; then
    ASE=~/ase
fi

if [[ -z $MYPYTHON ]]; then
    MYPYTHON=${GPAW_TOOLCHAIN}-2018b-Python-3.6.6
fi

# Load libxc unless another version is already loaded
if [[ -z "$EBROOTLIBXC" ]]; then
   module load libxc/3.0.1-${GPAW_TOOLCHAIN}-2018b
fi

# Load Python and matplotlib
module load matplotlib/3.0.0-$MYPYTHON

# Load default setups, unless GPAW_SETUP_PATH is already set
if [[ -z $GPAW_SETUP_PATH ]]; then
    module load GPAW-setups
fi


PLATFORM=linux-x86_64-$CPU_ARCH-el7-3.6
if [ $CPU_ARCH = broadwell ]; then
    export GPAW_MPI_OPTIONS="-mca pml cm -mca mtl psm2"
elif [ $CPU_ARCH = sandybridge ]; then
    export GPAW_MPI_OPTIONS=""
elif [ $CPU_ARCH = ivybridge ]; then
    export GPAW_MPI_OPTIONS=""
elif [ $CPU_ARCH = nehalem ]; then
    export GPAW_MPI_OPTIONS=""
fi

export PATH=$GPAW/tools:$GPAW/build/bin.$PLATFORM:$PATH
export PYTHONPATH=$GPAW:$GPAW/build/lib.$PLATFORM:$PYTHONPATH
export PATH=$ASE/bin:$PATH
export PYTHONPATH=$ASE:$PYTHONPATH
