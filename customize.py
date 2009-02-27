# extra_compile_args += ['-O3']
# extra_compile_args += ['-std=c99']
# extra_compile_args += ['-g']
# extra_compile_args += ['-fPIC']

libraries = [
           'hpm',
           'lapack_bgp',
           'scalapack',
           'blacsCinit_MPI-BGP-0',
           'blacs_MPI-BGP-0',
           'lapack_bgp',
           'goto',
           'xlf90_r',
           'xlopt',
           'xl',
           'xlfmath',
           'xlsmp'
           ]

library_dirs = [
           '/soft/apps/UPC/lib',
           '/soft/apps/LAPACK',
           '/soft/apps/LIBGOTO',
           '/soft/apps/BLACS',
           '/soft/apps/SCALAPACK',
           '/opt/ibmcmp/xlf/bg/11.1/bglib',
           '/opt/ibmcmp/xlsmp/bg/1.7/bglib',
           '/bgsys/drivers/ppcfloor/gnu-linux/lib'
           ]

include_dirs += [
    '/home/dulak/numpy-1.0.4-1.optimized/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/numpy/core/include'
    ]

define_macros += [("GPAW_AIX",1)]
define_macros += [("GPAW_MKL",1)]
define_macros += [("GPAW_BGP",1)]
define_macros += [("GPAW_ASYNC",1)]
# define_macros += [('GPAW_BGP_PERF',1)]
# define_macros += [("GPAW_OMP",1)]

scalapack = True

compiler = "bg_gcc.py"
mpicompiler = "bg_gcc.py"
mpilinker   = "mpicc"
