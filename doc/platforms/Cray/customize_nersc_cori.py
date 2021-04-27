parallel_python_interpreter = True
compiler = 'cc'
mpicompiler = 'cc'
mpilinker = 'cc'
scalapack = True
libxc = '/usr/common/software/libxc/4.2.3/gnu/haswell'
include_dirs += [libxc + '/include']
library_dirs += [libxc + '/lib']
extra_link_args += [f'-Wl,-rpath={libxc}/lib']
extra_compile_args += ['-O2']
if 'xc' not in libraries:
    libraries.append('xc')
# these are in the cray wrapper
if 'blas' in libraries:
    libraries.remove('blas')
if 'lapack' in libraries:
    libraries.remove('lapack')

if scalapack:
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
