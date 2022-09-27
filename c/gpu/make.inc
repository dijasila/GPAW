####################################################################
# Site configuration for make
####################################################################

# ----------------------------------------------------------------------
# Paths to header files to include (if needed)
# ----------------------------------------------------------------------
INCLUDE_PYTHON = $(PYTHONHOME)/include/python3.8/
INCLUDE_NUMPY  = $(shell python3 -c "import numpy; print(numpy.get_include())")
INCLUDE_MPI    =
INCLUDE_CUDA   =

PATHS    = . $(INCLUDE_PYTHON) $(INCLUDE_NUMPY) $(INCLUDE_MPI) $(INCLUDE_CUDA)
INCLUDES = $(addprefix -I,$(PATHS))

# ----------------------------------------------------------------------
# GPAW options to define
# ----------------------------------------------------------------------
DEFINES  = -DPARALLEL=1 -DGPAW_CUDA=1

# ----------------------------------------------------------------------
# Compilers and optimization flags
# ----------------------------------------------------------------------
COMPILE_FLAGS = $(INCLUDES) $(DEFINES)

CC        = gcc
CCFLAGS   = $(COMPILE_FLAGS) -g -fPIC -O3 -mavx512f

NVCC      = nvcc
NVCCFLAGS = $(COMPILE_FLAGS) -g -O3 -gencode arch=compute_70,code=sm_70 --compiler-options '-g -fPIC -O3 -mavx512f'

# ----------------------------------------------------------------------
# Linkers
# ----------------------------------------------------------------------
ARCH     = ar
ARCHFLAGS= cr
RANLIB   = ranlib

