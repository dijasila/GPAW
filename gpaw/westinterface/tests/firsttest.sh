#!/bin/sh
#PBS -q hpc
#PBS -N ChiTest
#PBS -l nodes=2:ppn=8
#PBS -l walltime=01:00:00
cd $PBS_O_WORKDIR
OMP_NUM_THREADS=1 mpiexec gpaw-python ./../gpawserver.py vext.xml chitestout.xml PBEdef_50.gpw
