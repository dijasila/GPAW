compiler = './icc.py'
mpicompiler = './icc.py'
mpilinker = 'MPICH_CC=gcc mpicc'

scalapack = True

define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
define_macros += [("GPAW_ASYNC",1)]
define_macros += [("GPAW_MPI2",1)]

scalapack = False

if scalapack:
    libraries += ['scalapack']
    library_dirs += []
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
