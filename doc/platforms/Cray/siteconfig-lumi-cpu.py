"""Custom GPAW siteconfig for LUMI-C."""

parallel_python_interpreter = True

mpi = True
compiler = 'cc'
compiler_args = []  # remove all default args
libraries = []
library_dirs = []
include_dirs = []
extra_compile_args = [
    '-O3',
    '-fopenmp',
    '-fPIC',
    '-Wall',
    '-Wno-stringop-overflow',  # suppress warnings from MPI_STATUSES_IGNORE
    '-Werror',
    '-g']
extra_link_args = ['-fopenmp']

# FFTW
fftw = True
libraries += ['fftw3']

# ScaLAPACK
# Note: required libraries are linked by compiler wrappers
scalapack = True

# Libxc
libraries += ['xc']

define_macros += [('GPAW_ASYNC', 1)]
