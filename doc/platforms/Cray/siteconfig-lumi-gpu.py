# Custom GPAW siteconfig for LUMI-G

parallel_python_interpreter = True

mpi = True
compiler = 'cc'
libraries = []
library_dirs = []
include_dirs = []
extra_compile_args = [
    '-O3',
    '-march=native',
    '-mtune=native',
    '-mavx2',
    '-fopenmp',
    '-fPIC',
    '-Wall',
    '-Wno-stringop-overflow',  # suppress warnings from MPI_STATUSES_IGNORE
    #'-DNDEBUG',
    '-g',
    ]
extra_link_args = ['-fopenmp']

# hip
gpu = True
gpu_target = 'hip-amd'
gpu_compiler = 'hipcc'
gpu_include_dirs = []
gpu_compile_args = ['--offload-arch=gfx90a', '-O3', '-g']
libraries += ['amdhip64', 'hipblas']
define_macros += [('GPAW_GPU_AWARE_MPI', None)]

# ELPA
elpa = True
libraries += ['elpa']

# FFTW
fftw = True
libraries += ['fftw3']

# ScaLAPACK
# Note: required libraries are linked by compiler wrappers
scalapack = True

# Libxc
libraries += ['xc']
#dpath = '/appl/lumi/spack/22.08/0.18.1/opt/spack/libxc-5.1.7-c55ppsw'
#include_dirs += [f'{dpath}/include']
#library_dirs += [f'{dpath}/lib']
#extra_link_args += [f'-Wl,-rpath,{dpath}/lib']

define_macros += [('GPAW_ASYNC', 1)]
