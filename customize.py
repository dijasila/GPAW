#User provided customizations for the gpaw setup

#Here, one can override the default arguments, or append own
#arguments to default ones
#To override use the form
#     libraries = ['somelib','otherlib']
#To append use the form
#     libraries += ['somelib','otherlib']

# Valid values for scalapack are False, or True:
# False (the default) - no ScaLapack compiled in
# True - ScaLapack compiled in
scalapack = True

compiler = 'icc'
libraries =['cublas','cudart','cufft','mkl_scalapack_lp64','mkl_blacs_openmpi_lp64','mkl_rt','pthread','m']

library_dirs = ['/nfs/slac/g/suncatfs/sw/rh6/external/openmpi/1.6.3/install/lib','/afs/slac/package/intel_tools/2013u0/mkl/lib/intel64','/opt/cuda-4.2/lib64']
#library_dirs += []

include_dirs += ['/nfs/slac/g/suncatfs/sw/rh6/external/numpy/1.6.1/install/lib64/python2.6/site-packages/numpy/core/include','/opt/cuda-4.2/include']
#include_dirs = []
#include_dirs += []

#extra_link_args = ['-static']
#extra_link_args += []
extra_link_args += ['-fPIC']

#extra_compile_args = ['-I/afs/slac/package/intel_tools/2013u0/mkl/include','-xHOST','-O3','-ipo','-no-prec-div','-static','-std=c99']
#extra_compile_args = ['-I/afs/slac/package/intel_tools/2013u0/mkl/include','-xHOST','-O1','-ipo','-no-prec-div','-static','-std=c99','-fPIC']
extra_compile_args = ['-I/afs/slac/package/intel_tools/2013u0/mkl/include','-xHOST','-ipo','-O1', '-no-prec-div','-static','-std=gnu99','-fPIC']
#extra_compile_args += []

#runtime_library_dirs = []
#runtime_library_dirs += []

extra_objects = []
extra_objects += ['/nfs/slac/g/suncatfs/junyan/download/gpu-dev-cuda4.2/c/cukernels.o']

#define_macros = []
#define_macros += []

mpicompiler = 'mpicc'
mpilinker = mpicompiler
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

if scalapack:
#    libraries += ['scalapack']
    library_dirs += []
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]

# In order to link libxc installed in a non-standard location
# (e.g.: configure --prefix=/home/user/libxc-2.0.1-1), use:
# - static linking:
#include_dirs += ['/home/user/libxc-2.0.1-1/include']
#extra_link_args += ['/home/user/libxc-2.0.1-1/lib/libxc.a']
#if 'xc' in libraries: libraries.remove('xc')
# - dynamic linking (requires also setting LD_LIBRARY_PATH at runtime):
#include_dirs += ['/home/user/libxc-2.0.1-1/include']
#library_dirs += ['/home/user/libxc-2.0.1-1/lib']
#if 'xc' not in libraries: libraries.append('xc')

