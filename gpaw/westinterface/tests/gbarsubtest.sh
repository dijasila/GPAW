#!/bin/sh
#PBS -q hpc
#PBS -N IntegrationTest
#PBS -l nodes=4:ppn=8
#PBS -l walltime=00:05:00
cd $PBS_O_WORKDIR
OMP_NUM_THREADS=1 mpiexec -np 16 gpaw-python /zhome/00/e/76055/gpaw/gpaw/westinterface/gpawserver.py ./submitin.xml ./submitout.xml ./subcalc.gpw &
OMP_NUM_THREADS=1 mpiexec -np 16 gpaw-python /zhome/00/e/76055/gpaw/gpaw/westinterface/westdummycode.py ./submitin.xml ./submitout.xml
wait

exit $?