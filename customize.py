"""User provided customizations.

Here, one change the default arguments for compiling _gpaw.so (serial)
and gpaw-python (parallel).

Here are all the lists that can be modified:
    
* libraries
* library_dirs
* include_dirs
* extra_link_args
* extra_compile_args
* runtime_library_dirs
* extra_objects
* define_macros
* mpi_libraries
* mpi_library_dirs
* mpi_include_dirs
* mpi_runtime_library_dirs
* mpi_define_macros

To override use the form:
    
    libraries = ['somelib', 'otherlib']

To append use the form

    libraries += ['somelib', 'otherlib']
"""

# compiler = 'gcc'
# mpicompiler = 'mpicc'  # use None if you don't want to build a gpaw-python
# mpilinker = 'mpicc'
# platform_id = ''
# scalapack = False
# hdf5 = False

# Use ScaLAPACK:
# Warning! At least scalapack 2.0.1 is required!
# See https://trac.fysik.dtu.dk/projects/gpaw/ticket/230
if scalapack:
    libraries += ['scalapack-openmpi',
                  'blacsCinit-openmpi',
                  'blacs-openmpi']
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]


username = 'username'

# -------------------------------------------------------------- #
# LibXC                                                          #
# -------------------------------------------------------------- #
# In order to link libxc installed in a non-standard location
# (e.g.: configure --prefix=/home/user/libxc-2.0.1-1), use:

# - static linking:
if 0:
    include_dirs += ['/home/%s/libxc-2.0.1-1/include' % username]
    extra_link_args += ['/home/%s/libxc-2.0.1-1/lib/libxc.a' % username]
    if 'xc' in libraries:
        libraries.remove('xc')
        
# - dynamic linking (requires also setting LD_LIBRARY_PATH at runtime):
if 0:
    include_dirs += ['/home/%s/libxc-2.0.1-1/include' % username]
    library_dirs += ['/home/%s/libxc-2.0.1-1/lib' % username]
    if 'xc' not in libraries:
        libraries.append('xc')

# -------------------------------------------------------------- #
# libvdwxc                                                       #
# -------------------------------------------------------------- #
# - dynamic linking (requires also setting LD_LIBRARY_PATH at runtime):
if 0:
    include_dirs += ['/home/%s/libvdwxc/include' % username]
    library_dirs += ['/home/%s/libvdwxc/lib' % username]
    libraries += ['vdwxc']
    define_macros += [('GPAW_WITH_LIBVDWXC', '1')]


# Build MPI-interface into _gpaw.so:
if 0:
    compiler = 'mpicc'
    define_macros += [('PARALLEL', '1')]
    mpicompiler = None
