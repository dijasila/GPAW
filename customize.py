#User provided customizations for the gpaw setup

#Here, one can override the default arguments, or append own
#arguments to default ones
#To override use the form
#     libraries = ['somelib','otherlib']
#To append use the form
#     libraries += ['somelib','otherlib']

compiler = 'mpicc'
libraries = []
library_dirs = []

include_dirs = []
#include_dirs += []

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

mpicompiler = 'mpicc'
mpilinker = 'mpicc'
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

hdf5 = True

if hdf5:
    libraries += ['hdf5']
    library_dirs += ['/lap/hdf5/1.8.11/gcc-4.6/openmpi/lib']
    include_dirs += ['/lap/hdf5/1.8.11/gcc-4.6/openmpi/include']

# Valid values for scalapack are False, or True:
# False (the default) - no ScaLapack compiled in
# True - ScaLapack compiled in
scalapack = True

if scalapack:
    libraries += ['scalapack_ompi']
    library_dirs += ['/lap/libscalapack/2.0.2/lib/gcc']
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]

libraries += ['lapack']
libraries += ['openblas']
libraries += ['gfortran']

library_dirs += ['/lap/liblapack/3.4.2/lib/gcc']
library_dirs += ['/lap/openblas/lib/gcc']

# Load external libxc
libraries += ['xc']
LIBXCDIR='/home/r/rasmusk/Public/libxc-2.2.0/install/'
library_dirs += [LIBXCDIR + 'lib']
include_dirs += [LIBXCDIR + 'include']
