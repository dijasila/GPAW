import os

mpicompiler = 'mpicc'

parallel_python_interpreter=True
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
    extra_link_args += ['-Wl,-rpath={}/lib'.format(elpa)]
    include_dirs.append(os.path.join(elpa, 'include', 'elpa-'+elpaversion))

# Now add a EasyBuild "cover-all-bases" library_dirs
library_dirs = os.getenv('LD_LIBRARY_PATH').split(':')



#   scalapack = True
#   fftw = True
#   parallel_python_interpreter=True

#   # Clean out any autodetected things, we only want the EasyBuild
#   # definitions to be used.
#   #libraries = ['openblas', 'fftw3', 'readline', 'gfortran']
#   #libraries = [ 'fftw3', 'readline', 'gfortran']
#   libraries=['xc']
#   #libraries = ['mkl_intel_lp64' ,'mkl_sequential' ,'mkl_core',
#   #             'mkl_lapack',
#   #             'mkl_scalapack_lp64', 'mkl_blacs_intelmpi_lp64',
#   #             'pthread',
#   #             'readline', 'termcap',
#                #'elpa',
#   #             'xc']
#   mpi_libraries = []
#   include_dirs = []

#   # Use EasyBuild scalapack from the active toolchain
#   #libraries += ['scalapack']

#   # Use EasyBuild libxc
#   libxc = os.getenv('EBROOTLIBXC')
#   if libxc:
#       include_dirs.append(os.path.join(libxc, 'include'))
#       libraries.append('xc')

#   # libvdwxc:
#   # Use EasyBuild libvdwxc
#   # This will only work with the foss toolchain.
#   libvdwxc = os.getenv('EBROOTLIBVDWXC')
#   if libvdwxc:
#       include_dirs.append(os.path.join(libvdwxc, 'include'))
#       libraries.append('vdwxc')

#   # ELPA:
#   # Use EasyBuild ELPA if loaded
#   elpa = os.getenv('EBROOTELPA')
#   if elpa:
#       libraries += ['elpa']
#       elpaversion = os.path.basename(elpa).split('-')[0]
#       #print(elpaversion)
#       #das
#       library_dirs = [os.path.join(elpa, 'lib')]
#       extra_link_args = [f'-Wl,-rpath={elpa}/lib']
#       include_dirs.append(os.path.join(elpa, 'include', 'elpa-' + elpaversion))

#   # Now add a EasyBuild "cover-all-bases" library_dirs
#   library_dirs = os.getenv('LD_LIBRARY_PATH').split(':')
