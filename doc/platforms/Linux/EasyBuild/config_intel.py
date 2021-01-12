mpicompiler = 'mpiicc'

# FFTW should be configured from environment variables, but they do
# not report the correct names for a dynamically loaded library.
fftw = True
# Use Intel MKL
libraries += ['mkl_sequential','mkl_core', 'fftw3xc_intel_pic', 'mkl_rt', ]

# Use EasyBuild scalapack from the active toolchain
scalapack = True
libraries += ['mkl_scalapack_lp64', 'mkl_blacs_intelmpi_lp64']

# Use EasyBuild libxc
libxc = os.getenv('EBROOTLIBXC')
include_dirs.append(os.path.join(libxc, 'include'))

# libvdwxc:
# Use EasyBuild libvdwxc
# NOTE: This currenlty does not work together with the Intel MKL, so
# the easyconfig files does not load libvdwxc
libvdwxc = os.getenv('EBROOTLIBVDWXC')
if libvdwxc:
    include_dirs.append(os.path.join(libvdwxc, 'include'))
    libraries.append('vdwxc')

# ELPA:
# Use EasyBuild ELPA if loaded
elpa = os.getenv('EBROOTELPA')
if elpa:
    libraries += ['elpa']
    elpaversion = os.path.basename(elpa).split('-')[0]
    library_dirs.append(os.path.join(elpa, 'lib'))
    extra_link_args += [f'-Wl,-rpath={elpa}/lib']
    include_dirs.append(os.path.join(elpa, 'include', 'elpa-'+elpaversion))

# Now add a EasyBuild "cover-all-bases" library_dirs
library_dirs = os.getenv('LD_LIBRARY_PATH').split(':')
