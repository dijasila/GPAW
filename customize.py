#User provided customizations for the gpaw setup

define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]


define_macros += [('GPAW_CUDA', '1')]
define_macros += [('PARALLER', '1')]


#Here, one can override the default arguments, or append own
#arguments to default ones
#To override use the form
#     libraries = ['somelib','otherlib']
#To append use the form
#     libraries += ['somelib','otherlib']


# Valid values for scalapack are False, or True:
# False (the default) - no ScaLapack compiled in
# True - ScaLapack compiled in
extra_compile_args = ['-std=c99', '-fPIC', '-g']

extra_link_args += ['-g']

#scalapack = True
scalapack = False


compiler = 'gcc'
#libraries = []
#libraries += []

#libraries = ['gpaw-cuda','cublas','cuda','acml', 'gfortran']
#scalapack = True
#libraries += ['scalapack', 'blacsCinit', 'blacsF77init', 'blacs']


libraries = ['gpaw-cuda','cublas','cuda','xc','c','mkl_scalapack_lp64','mkl_blacs_intelmpi_lp64','mkl_intel_lp64','gfortran','mkl_sequential','mkl_core','pthread']

#libraries = ['gpaw-cuda','cublas','cuda','scalapack', 'blacsCinit', 'blacsF77init', 'blacs','gfortran','mkl_intel_lp64','mkl_sequential','mkl_core','pthread']
#libraries = ['gpaw-cuda','cublas','cuda','mkl_intel_lp64','mkl_sequential','mkl_core','pthread']

#libraries = ['gpaw-cuda','cublas','cuda','mkl_scalapack_lp64','mkl_blacs_lp64','mkl_intel_lp64','mkl_sequential','mkl_core','pthread']

#libraries = ['gpaw-cuda','cublas','cuda','mkl_scalapack_lp64','mkl_blacs_openmpi_lp64','mkl_intel_lp64','mkl_intel_thread','mkl_core','iomp5','pthread']


#library_dirs = []

#library_dirs += ['./c/cuda','/opt/cuda-3.2.16/cuda/lib64']
library_dirs += ['/v/users/shakala/cuda/lib','/usr/lib64','./c/cuda','/v/linux26_x86_64/opt/cuda/toolkit/4.0.17/cuda/lib64','/v/linux26_x86_64/opt/intel/Compiler/11.1/064/mkl/lib/em64t/']
#                 '/v/linux26_x86_64/opt/intel/mkl/10.2.4.032/lib/em64t/']

#                 ,'/v/linux26_x86_64/opt/scalapack/1.8.0/gnu64/4.4.3/mvapich2/lib','/v/linux26_x86_64/opt/scalapack/1.8.0/gnu64/4.4.3/mvapich2/lib/']


#include_dirs = []
include_dirs += ['/v/users/shakala/cuda/include','/v/linux26_x86_64/opt/cuda/toolkit/4.0.17/cuda/include']
#include_dirs += ['/usr/local/cuda/include']


#extra_link_args = []
#extra_link_args += []

#extra_compile_args = []
#extra_compile_args += []

#runtime_library_dirs = []
#runtime_library_dirs += []

#extra_objects = []
#extra_objects += []

#define_macros = []
#define_macros += []

#mpicompiler = None
#mpilinker = None
#mpi_libraries = []
#mpi_libraries += []

#mpi_library_dirs = []
#mpi_library_dirs += []

#mpi_include_dirs = []
#mpi_include_dirs += []

#mpi_runtime_library_dirs = []
#mpi_runtime_library_dirs += []

#mpi_define_macros = []
#mpi_define_macros += []

#platform_id = ''

#hdf5 = True

# Valid values for scalapack are False, or True:
# False (the default) - no ScaLapack compiled in
# True - ScaLapack compiled in
# Warning! At least scalapack 2.0.1 is required!
# See https://trac.fysik.dtu.dk/projects/gpaw/ticket/230
scalapack = False

if scalapack:
    libraries += ['scalapack']
    library_dirs += []
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
