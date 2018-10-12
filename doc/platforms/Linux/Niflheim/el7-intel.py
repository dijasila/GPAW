"""User provided customizations.

Here one changes the default arguments for compiling _gpaw.so (serial)
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

scalapack = True

platform_id = os.environ['CPU_ARCH'] + '-el7'

# Convert static library specs from EasyBuild to GPAW
def static_eblibs_to_gpawlibs(lib_specs):
    return [libfile[3:-2] for libfile in os.getenv(lib_specs).split(',')]

# Clean out any autodetected things, we only want the EasyBuild
# definitions to be used.
libraries = []
mpi_libraries = []
include_dirs = []

# Use Intel MKL
libraries += ['fftw3xc_intel','mkl_intel_lp64','mkl_sequential','mkl_core']

# Use ScaLAPACK:
# Warning! At least scalapack 2.0.1 is required!
# See https://trac.fysik.dtu.dk/projects/gpaw/ticket/230
# Use EasyBuild scalapack from the active toolchain
if scalapack:
    mpi_libraries += ['mkl_scalapack_lp64','mkl_blacs_intelmpi_lp64']
    mpi_define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    mpi_define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]

# LibXC:
# Use EasyBuild libxc
libxc = os.getenv('EBROOTLIBXC')
if libxc:
    include_dirs.append(os.path.join(libxc, 'include'))
    libraries.append('xc')

# libvdwxc:
# Use EasyBuild libvdwxc
# This will only work with the foss toolchain.
libvdwxc = os.getenv('EBROOTLIBVDWXC')
if libvdwxc:
    include_dirs.append(os.path.join(libvdwxc, 'include'))
    libraries.append('vdwxc')

# Now add a EasyBuild "cover-all-bases" library_dirs
library_dirs = os.getenv('LD_LIBRARY_PATH').split(':')

# Build separate gpaw-python
mpicompiler = 'mpiicc'
mpilinker = mpicompiler

