import os

scalapack = True
fftw = True

# Clean out any autodetected things, we only want the EasyBuild
# definitions to be used.
libraries = ['openblas', 'fftw3', 'readline', 'gfortran']
mpi_libraries = []
include_dirs = []

# Use EasyBuild scalapack from the active toolchain
libraries += ['scalapack']

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
