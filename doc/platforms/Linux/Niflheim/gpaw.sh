if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

if [ -z $GPAW ]; then
    GPAW=~/gpaw
fi
if [ -z $ASE ]; then
    ASE=~/ase
fi

module load GPAW
module unload ASE

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
