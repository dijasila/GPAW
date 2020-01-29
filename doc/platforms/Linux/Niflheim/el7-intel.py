scalapack = True
fftw = True
mpicompiler = 'mpiicc'

# Use Intel MKL
libraries += ['mkl_sequential', 'mkl_core', 'fftw3xc_intel_pic', 'mkl_rt']

# Use EasyBuild scalapack from the active toolchain
libraries += ['mkl_scalapack_lp64', 'mkl_blacs_intelmpi_lp64']

# Use EasyBuild libxc
libxc = os.getenv('EBROOTLIBXC')
include_dirs.append(os.path.join(libxc, 'include'))

# libvdwxc:
# Use EasyBuild libvdwxc
# This will only work with the foss toolchain.
libvdwxc = os.getenv('EBROOTLIBVDWXC')
if libvdwxc:
    include_dirs.append(os.path.join(libvdwxc, 'include'))
    libraries.append('vdwxc')

# Now add a EasyBuild "cover-all-bases" library_dirs
library_dirs = os.getenv('LD_LIBRARY_PATH').split(':')
