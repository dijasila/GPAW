# Custom GPAW siteconfig for LUMI

# compiler and linker 
compiler = 'cc'
mpicompiler = 'cc'
mpilinker = 'cc'
extra_compile_args = ['-g', '-fPIC', '-march=native', '-O3', '-fopenmp']
extra_link_args = ['-fopenmp']

# ScaLAPACK 
# required libraries are linked by compiler wrappers 
scalapack = True

# libxc 
xc = '/appl/lumi/spack/22.08/0.18.1/opt/spack/libxc-5.1.7-c55ppsw/'
include_dirs += [xc + 'include']
library_dirs += [xc + 'lib']
# Use rpath to avoid setting LD_LIBRARY_PATH:
extra_link_args += ['-Wl,-rpath={xc}/lib'.format(xc=xc)]
if 'xc' not in libraries:
    libraries.append('xc')
