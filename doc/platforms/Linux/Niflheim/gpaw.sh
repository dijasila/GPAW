if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

GPAW=~/gpaw
ASE=~/ase

module load GPAW
module load matplotlib
PLATFORM=linux-x86_64-$CPU_ARCH-el7-3.5
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
export PATH=$ASE/tools:$PATH
export PYTHONPATH=$ASE:$PYTHONPATH
