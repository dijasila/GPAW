# Custom GPAW siteconfig for LUMI
import os
parallel_python_interpreter = False

# compiler and linker
compiler = 'cc'
mpicompiler = 'cc'
mpilinker = 'cc'
extra_compile_args = ['-g', '-fopenmp-simd', '-O3']
extra_link_args = ['-fopenmp']

# libraries
libraries = ['z']

# ScaLAPACK
# required libraries are linked by compiler wrappers
scalapack = True

# libxc
# - dynamic linking (requires rpath or setting LD_LIBRARY_PATH at runtime):
if 1:
    xc = '/appl/lumi/spack/22.08/0.18.1/opt/spack/libxc-5.1.7-c55ppsw/'
    include_dirs += [xc + 'include']
    library_dirs += [xc + 'lib']
    # You can use rpath to avoid changing LD_LIBRARY_PATH:
    extra_link_args += ['-Wl,-rpath={xc}/lib'.format(xc=xc)]
    if 'xc' not in libraries:
        libraries.append('xc')
