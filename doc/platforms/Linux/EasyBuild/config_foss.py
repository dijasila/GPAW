# Convert static library specs from EasyBuild to GPAW
def static_eblibs_to_gpawlibs(lib_specs):
    return [libfile[3:-2] for libfile in os.getenv(lib_specs).split(',')]

# Clean out any autodetected things, we only want the EasyBuild
# definitions to be used.
libraries = []
include_dirs = []

# Use EasyBuild fftw from the active toolchain
fftw = os.getenv('FFT_STATIC_LIBS')
if fftw:
    libraries += static_eblibs_to_gpawlibs('FFT_STATIC_LIBS')

# Use ScaLAPACK from the active toolchain
scalapack = os.getenv('SCALAPACK_STATIC_LIBS')
if scalapack:
    libraries += static_eblibs_to_gpawlibs('SCALAPACK_STATIC_LIBS')

# Add EasyBuild LAPACK/BLAS libs
libraries += static_eblibs_to_gpawlibs('LAPACK_STATIC_LIBS')
libraries += static_eblibs_to_gpawlibs('BLAS_STATIC_LIBS')

# LibXC:
# Use EasyBuild libxc
libxc = os.getenv('EBROOTLIBXC')
if libxc:
    include_dirs.append(os.path.join(libxc, 'include'))
    libraries.append('xc')

# libvdwxc:
# Use EasyBuild libvdwxc
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
