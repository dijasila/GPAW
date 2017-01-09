#!/bin/sh

if [ -r "/home/camp/modulefiles.sh" ]; then
    source /home/camp/modulefiles.sh
fi
if [ -r "/home/opt/modulefiles/modulefiles_el6.sh" ]; then
    source /home/opt/modulefiles/modulefiles_el6.sh
fi

if [ "`echo $FYS_PLATFORM`" == "AMD-Opteron-el5" ]; then # fjorm
    module load GPAW
    export GPAW_PLATFORM="linux-x86_64-opteron-2.4"
fi
if test -n "`echo $FYS_PLATFORM | grep el6`"; then
    module load GPAW
    export GPAW_PLATFORM="linux-x86_64-`echo $FYS_PLATFORM | sed 's/-el6//'`-2.6"
fi
# GPAW_HOME must be set after loading the GPAW module!
export ASE_HOME=/home/niflheim/mogje/ase_dists/ase_trunk
export GPAW_HOME=/home/niflheim/mogje/gpaw_dists/gpaw_trunk_latest/
export PATH=${GPAW_HOME}/build/bin.${GPAW_PLATFORM}:${PATH}
export PATH=${GPAW_HOME}/tools:${PATH}
export PYTHONPATH=${GPAW_HOME}:${PYTHONPATH}
export PYTHONPATH=${GPAW_HOME}/build/lib.${GPAW_PLATFORM}:${PYTHONPATH}

export ASE_HOME=/home/niflheim/$USER/ase_dists/ase_trunk_latest
export PYTHONPATH=/${ASE_HOME}:$PYTHONPATH
export PATH=/${ASE_HOME}/tools:$PATH

if test -n "`echo $FYS_PLATFORM | grep el6`"; then
# http://docs.python.org/2/using/cmdline.html#envvar-PYTHONDONTWRITEBYTECODE
    export PYTHONDONTWRITEBYTECODE=1  # disable creation of pyc files
    module load NUMPY/1.7.1-1
    module load SCIPY/0.12.0-1
fi

mpiexec gpaw-python "$name"