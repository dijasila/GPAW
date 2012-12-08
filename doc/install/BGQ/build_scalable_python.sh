#!/bin/bash

export CC=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-gcc
export CXX=/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc64-bgq-linux-g++
export MPICC=mpicc
export CCSHARED=-fPIC
export LINKFORSHARED='-Xlinker -export-dynamic -dynamic'
export MPI_LDFLAGS_SHARED='-Xlinker -export-dynamic -dynamic'

./configure --prefix=/soft/apps/python/scalable-python-2.6.7-cnk-gcc --enable-mpi --disable-ipv6   2>&1 | tee loki-conf

make 2>&1 | tee loki-make
make mpi 2>&1 | tee loki-make-mpi

make install 2>&1 | tee loki-inst
make install-mpi 2>&1 | tee loki-inst-mpi
