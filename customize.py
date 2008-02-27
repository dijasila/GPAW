#User provided customizations for the gpaw setup

#Here, one can override the default arguments, or append own
#arguments to default ones
#To override use the form
#     libraries = ['somelib',otherlib']
#To append use the form
#     libraries += ['somelib',otherlib']


#compiler = 'mpcc'
#libraries = []
#libraries += []
libraries += [
    'gfortran',
    #'mpiblacsF77init',
    'mpiblacsCinit',
    #    'lapack',
    'acml',
    'mpiblacs',
    'scalapack',
    'mpi_f77']

#library_dirs = []
#library_dirs += []
#library_dirs += ['/usr/local/openmpi-1.2.3-gfortran/lib']
library_dirs += [
    '/opt/acml-4.0.1/gfortran64/lib',
    '/usr/local/openmpi-1.2.3-gfortran/lib64',
    '/usr/lib64/blacs-1.1-24.5',
    '/usr/lib64/scalapack-1.7.5-5'
    #'/usr/lib64/scalapack-1.8.0-1'
    ]

#include_dirs = []
#include_dirs += []

#extra_link_args = []
#extra_link_args += []
extra_link_args += [
    '-Wl,-rpath=/opt/acml-4.0.1/gfortran64/lib,-rpath=/usr/local/openmpi-1.2.3-gfortran/lib64,-rpath=/usr/lib64/blacs-1.1-24.5,-rpath=/usr/lib64/scalapack-1.7.5-5'
    #'-Wl,-rpath=/opt/acml-4.0.1/gfortran64/lib,-rpath=/usr/local/openmpi-1.2.3-gfortran/lib64,-rpath=/usr/lib64/blacs-1.1-24.5,-rpath=/usr/lib64/scalapack-1.8.0-1'
    ]

#extra_compile_args = []
#extra_compile_args += []

#runtime_library_dirs = []
#runtime_library_dirs += []

#extra_objects = []
#extra_objects += []

#define_macros = []
#define_macros += []

#mpicompiler = None
#mpi_libraries = []
#mpi_libraries += []

#mpi_library_dirs = []
#mpi_library_dirs += []
#mpi_library_dirs += [
#    '/usr/local/openmpi-1.2.3-gfortran/lib64'
#    ]

#mpi_include_dirs = []
#mpi_include_dirs += []

#mpi_runtime_library_dirs = []
#mpi_runtime_library_dirs += []
mpi_runtime_library_dirs += [
    '/opt/acml-4.0.1/gfortran64/lib',
    '/usr/local/openmpi-1.2.3-gfortran/lib64'
    '/usr/lib64/blacs-1.1-24.5',
    '/usr/lib64/scalapack-1.7.5-5'
    #'/usr/lib64/scalapack-1.8.0-1'
    ]

#mpi_define_macros = []
#mpi_define_macros += []
