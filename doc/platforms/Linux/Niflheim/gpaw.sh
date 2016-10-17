if [ -n "$FYS_PLATFORM" ]
then
    cd $PBS_O_WORKDIR
    source /home/opt/modulefiles/modulefiles_el6.sh
    module load GPAW
    module load NUMPY/1.7.1-1
    module load SCIPY/0.12.0-1
    module load fftw
    PLATFORM=linux-x86_64-$FYS_PLATFORM-2.6
    GPAW=~/gpaw
    ASE=~/ase
    alias gpaw-qsub=$GPAW/doc/platforms/Linux/Niflheim/gpaw-qsub.py
else
    module load GPAW
    PLATFORM=linux-x86_64-el7-3.5
    GPAW=~/gpaw7
    ASE=~/ase7
    alias gpaw-sbatch=$GPAW/doc/platforms/Linux/Niflheim/gpaw-sbatch.py
fi

export PATH=$GPAW/tools:$GPAW/build/bin.$PLATFORM:$PATH
export PYTHONPATH=$GPAW:$GPAW/build/lib.$PLATFORM:$PYTHONPATH
export PATH=$ASE/tools:$PATH
export PYTHONPATH=$ASE:$PYTHONPATH
