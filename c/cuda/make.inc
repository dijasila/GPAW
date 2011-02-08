####################################################################
#  make include file.                                              #
####################################################################
#
SHELL = /bin/sh

# ----------------------------------------------------------------------
# - gpaw-cuda Directory Structure / gpaw-cuda library ------------------------------
# ----------------------------------------------------------------------
#
TOPdir       = .
INCdir       = $(TOPdir)
PYTHONINCdir = /v/linux26_x86_64/opt/python/2.6.5-gcc/include/python2.6
NUMPYINCdir  = /v/linux26_x86_64/opt/python/2.6.5-gcc/lib/python2.6/site-packages/numpy/core/include
LIBdir       = $(TOPdir)
CUGPAWLIB    = $(LIBdir)/libgpaw-cuda.a 

#
# ----------------------------------------------------------------------
# - NVIDIA CUDA includes / libraries / specifics -----------------------
# ----------------------------------------------------------------------
CUDAdir       = /opt/cuda-3.2.16/cuda
CUDAinc       = -I$(CUDAdir)/include
CUDA_OPTS     = 

# ----------------------------------------------------------------------
# - gpaw-cuda includes / libraries / specifics -------------------------------
# ----------------------------------------------------------------------
#

CUGPAW_INCLUDES = -I$(INCdir) $(CUDAinc) -I$(NUMPYINCdir)  -I$(PYTHONINCdir)
# CUGPAW_OPTS     = -DPARALLEL=0 -DGPAW_CUDA=1
CUGPAW_OPTS     = -DPARALLEL=1 -DGPAW_CUDA=1
#CUGPAW_OPTS     = -DGPAW_CUDA=1
#
# ----------------------------------------------------------------------
#
CUGPAW_DEFS     = $(CUGPAW_OPTS) $(CUDA_OPTS)  $(CUGPAW_INCLUDES)
#
# ----------------------------------------------------------------------
# - Compilers / linkers - Optimization flags ---------------------------
# ----------------------------------------------------------------------
#
CC           = gcc
CCNOOPT      = $(CUGPAW_DEFS)
##CCFLAGS      = $(CUGPAW_DEFS)  -g -mfpmath=sse -msse3 -mtune=native -O3 -funroll-loops -W -Wall -m64
# CCFLAGS      = $(CUGPAW_DEFS) -fomit-frame-pointer -mfpmath=sse -msse3 -mtune=native -O3 -funroll-loops -W -Wall -m64 -fPIC -std=c99
CCFLAGS      = $(CUGPAW_DEFS) -O3 -fPIC -std=c99 

NVCC           = nvcc
# NVCCFLAGS      = $(CUGPAW_DEFS) -arch=sm_20 -O3 --compiler-options '-std=c99 -mfpmath=sse -msse3 -fPIC' --ptxas-options=-v 
NVCCFLAGS      = $(CUGPAW_DEFS) -arch=sm_20 --compiler-options '-std=c99 -fPIC' 


ARCH     = ar
ARCHFLAGS= cr
RANLIB   = ranlib


