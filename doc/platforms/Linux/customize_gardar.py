scalapack = True

mklpath ='/global/apps/intel/2013.1/mkl'
omppath ='/global/apps/openmpi/1.6.5/intel/13.1'
lxcpath ='/home/pcje/global/apps/libxc-2.2.1-1'

compiler = 'icc'

libraries = ['xc', 'mpi', 'mkl_scalapack_lp64', 'mkl_lapack95_lp64', 'mkl_intel_lp64', 'mkl_sequential', 'mkl_mc', 'mkl_core', 'mkl_def', 'mkl_intel_thread', 'iomp5']
library_dirs += [f'{omppath}/lib']
library_dirs += [f'{mklpath}/lib/intel64']
library_dirs += [f'{lxcpath}/lib']
include_dirs += [f'{omppath}/include']
include_dirs += [f'{mklpath}/include']
include_dirs += [f'{lxcpath}/include']

extra_link_args += [f'{mklpath}/lib/intel64/libmkl_blacs_openmpi_lp64.a', f'{mklpath}/lib/intel64/libmkl_blas95_lp64.a']

extra_compile_args += ['-O3', '-std=c99', '-fPIC', '-Wall']

define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]

mpicompiler = 'mpicc'
mpilinker = mpicompiler
