#User provided customizations for the gpaw setup
 
#hdf5 = True
scalapack = True
 
if scalapack:
    define_macros = [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
 
#opt_string = '-prof-use'
 
compiler = 'icc'
linker = 'icc ' #+ opt_string
 
extra_compile_args = [
    '-w',
    '-xhost',
    #'-no-prec-div',
    '-O3',
    '-ipo',
    '-funroll-all-loops',
    '-fPIC',
    #'-static',
    '-std=c99',
    #opt_string,
    ]
 
libraries = [
    'xc',
    'mpi',
    ]
 
mkl_lib_path = '/apps/rhel6/intel/composer_xe_2015.1.133/mkl/lib/intel64/'
 
extra_link_args = [
    '/apps/rhel6/libxc-2.2.0/lib/libxc.a',
    mkl_lib_path + 'libmkl_scalapack_lp64.a',
    '-Wl,--start-group',
    mkl_lib_path + 'libmkl_intel_lp64.a',
    mkl_lib_path + 'libmkl_sequential.a',
    mkl_lib_path + 'libmkl_core.a',
    mkl_lib_path + 'libmkl_blacs_intelmpi_lp64.a',
    '-Wl,--end-group',
    '-lpthread',
    '-lm',
    ]
 
library_dirs = [
    '/apps/rhel6/intel/composer_xe_2015.1.133/mkl/lib/intel64',
    '/apps/rhel6/intel/composer_xe_2015.1.133/mkl/compiler/intel64',
    '/apps/rhel6/intel/impi/5.0.2.044/lib64',
    '/apps/rhel6/libxc-2.2.0/include',
    ]
 
include_dirs = [
    '/apps/rhel6/intel/composer_xe_2015.1.133/mkl/lib/intel64',
    '/apps/rhel6/intel/composer_xe_2015.1.133/mkl/compiler/intel64',
    '/apps/rhel6/intel/impi/5.0.2.044/include64'
    ]
 
mpicompiler = 'mpiicc'
mpilinker = mpicompiler + ' ' #+ opt_string
 
mpi_libraries = [
#'hdf5',
#'z',
]
 
mpi_library_dirs = ['/apps/rhel6/intel/impi/5.0.2.044/lib64']
 
mpi_include_dirs = ['/apps/rhel6/intel/impi/5.0.2.044/include64']
 
define_macros += [('GPAW_MKL','1')] 
