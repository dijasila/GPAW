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
libraries =['xc','mkl_scalapack_lp64','mkl_blacs_openmpi_lp64','mpi','mkl_intel_lp64','mkl_sequential','mkl_cdft_core','mkl_core','pthread','m']

library_dirs = ['/nfs/slac/g/suncatfs/sw/external/libxc/1.2.0/install/lib','/nfs/slac/g/suncatfs/sw/external/intel11.1/openmpi/1.4.3/install/lib','/afs/slac/package/intel_tools/compiler11.1/mkl/lib/em64t/','/nfs/slac/g/suncatfs/sw/external/intel11.1/openmpi/1.4.3/install/lib']
#library_dirs += []

include_dirs += ['/nfs/slac/g/suncatfs/sw/external/libxc/1.2.0/install/include','/nfs/slac/g/suncatfs/sw/external/numpy/1.4.1/install/lib64/python2.4/site-packages/numpy/core/include']
#include_dirs = []
#include_dirs += []

#extra_link_args = ['-static']
#extra_link_args += []
extra_link_args += ['-fPIC']

#extra_compile_args = ['-I/afs/slac/package/intel_tools/compiler11.1/mkl/include','-xHOST','-O3','-ipo','-no-prec-div','-static','-std=c99']
extra_compile_args = ['-I/afs/slac/package/intel_tools/compiler11.1/mkl/include','-xHOST','-O0','-g','-ipo','-no-prec-div','-static','-std=c99','-fPIC']
#extra_compile_args = ['-I/afs/slac/package/intel_tools/compiler11.1/mkl/include','-xHOST','-O1','-ipo','-no-prec-div','-static','-std=c99','-fPIC']
#extra_compile_args += []

#runtime_library_dirs = []
#runtime_library_dirs += []

#extra_objects = []
#extra_objects += []

define_macros =[('GPAW_NO_UNDERSCORE_CBLACS', '1'), ('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
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
