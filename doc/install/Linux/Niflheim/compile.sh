#!/bin/sh

if test -z $GPAW_HOME;
    then
    echo "Error: \$GPAW_HOME variable not set"
    exit 1
fi

rm -rf $GPAW_HOME/build/
echo "source /home/opt/modulefiles/modulefiles_el6.sh&& module purge&& module load NUMPY/1.7.1-1&& module load NVIDIA-Linux-x86_64&& module load cuda&& cd $GPAW_HOME&& rm -f cukernels.o&& nvcc -arch sm_35 -c c/cukernels.cu -Xcompiler -fPIC&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/el6-sl270-gpu-tm-gfortran-openmpi-1.6.3-acml-4.4.0-sl-hdf5-1.8.10.py build_ext 2>&1 | tee el6-sl270-gpu-tm-gfortran-openmpi-1.6.3-acml-4.4.0-sl-hdf5-1.8.10.log" | ssh surt bash

