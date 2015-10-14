import os
if 'GPAW_MPI' in os.environ:
    # Build MPI-interface into _gpaw.so:
    compiler = 'mpicc'
    define_macros += [('PARALLEL', '1')]
    mpicompiler = None

compiler = 'gcc'

# change /home/fr/fr_fr/fr_tr75/source/libxc-2.2.2 to 
# $MYLIBXCDIR/libxc-2.0.2/install
include_dirs += ['/home/fr/fr_fr/fr_tr75/source/libxc-2.2.2/include']
library_dirs += ['/home/fr/fr_fr/fr_tr75/source/libxc-2.2.2/lib']
if 'xc' not in libraries:
    libraries.append('xc')


extra_link_args += [
     '-Wl,--no-as-needed',
    '-L/opt/bwhpc/common/compiler/intel/compxe.2013.sp1.4.211/mkl/lib/intel64',
    '-lmkl_scalapack_lp64',
    '-lmkl_intel_lp64',
    '-lmkl_core',
    '-lmkl_sequential',
    '-lmkl_blacs_intelmpi_lp64',
    '-lpthread',
    '-lm',
]

#extra_compile_args = []
extra_compile_args += [
    '-m64',
    '-I/opt/bwhpc/common/compiler/intel/compxe.2013.sp1.4.211/mkl/include',
]

