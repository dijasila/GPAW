####################################################################
#  make include file.                                              #
####################################################################
#
SHELL = /bin/sh

# ----------------------------------------------------------------------
# - gpaw-cuda Directory Structure / gpaw-cuda library --------------------
# ----------------------------------------------------------------------
#
TOPdir       = .
INCdir       = $(TOPdir)
PYTHONINCdir?=/fs/local/linux26_x86_64/opt/python/2.6.5-gcc/include/python2.6/
PYTHONLIBdir?=/fs/local/linux26_x86_64/opt/python/2.6.5-gcc/lib/
NUMPYINCdir?=`python -c "import numpy; print numpy.get_include()"`
MPIINCdir?=/v/linux26_x86_64/opt/mvapich2/1.8a1p1-cuda/include
LIBdir       = $(TOPdir)
CUGPAWLIB    = $(LIBdir)/libgpaw-cuda.a 

#
# ----------------------------------------------------------------------
# - NVIDIA CUDA includes / libraries / specifics -----------------------
# ----------------------------------------------------------------------
CUDAdir       = $(CUDA_INSTALL_PATH)
CUDAINCdir       = $(CUDAdir)/include
CUDALIBdir       = $(CUDAdir)/lib64
CUDA_OPTS     = 

#
# ----------------------------------------------------------------------
# - gpaw-cuda includes / libraries / specifics -------------------------------
# ----------------------------------------------------------------------
#

CUGPAW_INCLUDES =  -I$(INCdir) -I$(CUDAINCdir) -I$(MPIINCdir) -I$(NUMPYINCdir)  -I$(PYTHONINCdir)
CUGPAW_OPTS     = -DPARALLEL=1 -DGPAW_CUDA=1
#CUGPAW_OPTS     = -DPARALLEL=1 -DGPAW_CUDA=1 -DMPI_CUDA=1

#
# ----------------------------------------------------------------------
#

CUGPAW_DEFS     = $(CUGPAW_OPTS) $(CUDA_OPTS)  $(CUGPAW_INCLUDES)

#
# ----------------------------------------------------------------------
# - Compilers / linkers - Optimization flags ---------------------------
# ----------------------------------------------------------------------

CC           = gcc
CCNOOPT      = $(CUGPAW_DEFS)
# CCFLAGS      = $(CUGPAW_DEFS) -fomit-frame-pointer -mfpmath=sse -msse3 -mtune=native -O3 -funroll-loops -W -Wall -m64 -fPIC -std=c99
CCFLAGS      = $(CUGPAW_DEFS) -g -fPIC -std=c99 -m64 -O3

NVCC           = nvcc
NVCCFLAGS      = $(CUGPAW_DEFS) -O3 -g -arch=sm_20 -m64 --compiler-options '-O3 -g -std=c99 -fPIC' 
#NVCCFLAGS      = $(CUGPAW_DEFS) -Xptxas -v -O3 -g -arch=sm_20 -m64 --compiler-options '-O3 -g -std=c99 -fPIC' 
#NVCCFLAGS      = $(CUGPAW_DEFS) -Xptxas -v -O3 -g -G -arch=sm_20 -m64 --compiler-options '-O3 -g -std=c99 -fPIC' 



ARCH     = ar
ARCHFLAGS= cr
RANLIB   = ranlib


