nodetype = 'sl230s'
scalapack = True
compiler = 'gcc'
libraries =[
    'gfortran',
    'scalapack',
    'mpiblacs',
    'mpiblacsCinit',
    'acml',
    'acml_mv',
    'hdf5',
    'xc',
    'mpi',
    'mpi_f77',
    'cuda',  # from NVIDIA-Linux-x86_64
    'cublas', 'cufft', 'cudart',  # from cuda
    ]
library_dirs =[
    '/home/opt/el6/' + nodetype + '/openmpi-1.6.3-' + nodetype + '-tm-gfortran-1/lib',
    '/home/opt/el6/' + nodetype + '/blacs-1.1-' + nodetype + '-tm-gfortran-openmpi-1.6.3-1/lib',
    '/home/opt/el6/' + nodetype + '/scalapack-2.0.2-' + nodetype + '-tm-gfortran-openmpi-1.6.3-acml-4.4.0-1/lib',
    '/home/opt/common/acml-gfortran-64bit-4.4.0/lib',
    '/home/opt/el6/' + nodetype + '/hdf5-1.8.10-' + nodetype + '-tm-gfortran-openmpi-1.6.3-1/lib',
    '/home/opt/el6/' + nodetype + '/libxc-2.0.1-' + nodetype + '-gfortran-1/lib',
    '/home/opt/common/NVIDIA-Linux-x86_64-331.20/lib',
    '/home/opt/el6/common/cuda-5.5.22/lib64',
    ]
include_dirs +=[
    '/home/opt/el6/' + nodetype + '/openmpi-1.6.3-' + nodetype + '-tm-gfortran-1/include',
    '/home/opt/el6/' + nodetype + '/hdf5-1.8.10-' + nodetype + '-tm-gfortran-openmpi-1.6.3-1/include',
    '/home/opt/el6/' + nodetype + '/libxc-2.0.1-' + nodetype + '-gfortran-1/include',
    '/home/opt/common/NVIDIA-Linux-x86_64-331.20/include',
    '/home/opt/el6/common/cuda-5.5.22/include',
    ]
extra_link_args =[
    '-Wl,-rpath=/home/opt/el6/' + nodetype + '/openmpi-1.6.3-' + nodetype + '-tm-gfortran-1/lib'
    ',-rpath=/home/opt/el6/' + nodetype + '/blacs-1.1-' + nodetype + '-tm-gfortran-openmpi-1.6.3-1/lib'
    ',-rpath=/home/opt/el6/' + nodetype + '/scalapack-2.0.2-' + nodetype + '-tm-gfortran-openmpi-1.6.3-acml-4.4.0-1/lib'
    ',-rpath=/home/opt/common/acml-gfortran-64bit-4.4.0/lib'
    ',-rpath=/home/opt/el6/' + nodetype + '/hdf5-1.8.10-' + nodetype + '-tm-gfortran-openmpi-1.6.3-1/lib'
    ',-rpath=/home/opt/el6/' + nodetype + '/libxc-2.0.1-' + nodetype + '-gfortran-1/lib'
    ',-rpath=/home/opt/common/NVIDIA-Linux-x86_64-331.20/lib'
    ',-rpath=/home/opt/el6/common/cuda-5.5.22/lib64'
    ]
extra_compile_args =['-O3', '-std=c99', '-fPIC', '-Wall']
# nvcc -arch sm_35 -c c/cukernels.cu -Xcompiler -fPIC
extra_objects += ['cukernels.o']
define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
define_macros += [('GPAW_CUDA', '1')]
mpicompiler = '/home/opt/el6/' + nodetype + '/openmpi-1.6.3-' + nodetype + '-tm-gfortran-1/bin/mpicc'
mpilinker = mpicompiler
platform_id = 'sl270s'
hdf5 = True
