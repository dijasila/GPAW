#!/usr/bin/env python

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import distutils
import distutils.util
import os
import re
import sys

from distutils.command.build_scripts import build_scripts as _build_scripts
from distutils.command.sdist import sdist as _sdist
from distutils.core import setup, Extension
from glob import glob
from os.path import join

from config import (check_packages, get_system_config, get_parallel_config,
                    check_dependencies,
                    write_configuration, build_interpreter, get_config_vars)

# Get the current version number:
with open('gpaw/__init__.py') as fd:
    version = re.search("__version__ = '(.*)'", fd.read()).group(1)
 
long_description = """\
A grid-based real-space Projector Augmented Wave (PAW) method Density
Functional Theory (DFT) code featuring: Flexible boundary conditions,
k-points and gradient corrected exchange-correlation functionals."""

msg = [' ']

libraries = []
library_dirs = []
include_dirs = []
extra_link_args = []
extra_compile_args = []
runtime_library_dirs = []
extra_objects = []
define_macros = [('NPY_NO_DEPRECATED_API', 7)]
undef_macros = []

mpi_libraries = []
mpi_library_dirs = []
mpi_include_dirs = []
mpi_runtime_library_dirs = []
mpi_define_macros = []

platform_id = ''


packages = []
for dirname, dirnames, filenames in os.walk('gpaw'):
        if '__init__.py' in filenames:
            packages.append(dirname.replace('/', '.'))

include_ase = False
if '--include-ase' in sys.argv:
    include_ase = True
    sys.argv.remove('--include-ase')

import_numpy = True
if '--ignore-numpy' in sys.argv:
    import_numpy = False
    sys.argv.remove('--ignore-numpy')

remove_default_flags = False
if '--remove-default-flags' in sys.argv:
    remove_default_flags = True
    sys.argv.remove('--remove-default-flags')

customize = 'customize.py'
for i, arg in enumerate(sys.argv):
    if arg.startswith('--customize'):
        customize = sys.argv.pop(i).split('=')[1]
        break

check_packages(packages, msg, include_ase, import_numpy)

get_system_config(define_macros, undef_macros,
                  include_dirs, libraries, library_dirs,
                  extra_link_args, extra_compile_args,
                  runtime_library_dirs, extra_objects, msg,
                  import_numpy)

mpicompiler = get_parallel_config(mpi_libraries,
                                  mpi_library_dirs,
                                  mpi_include_dirs,
                                  mpi_runtime_library_dirs,
                                  mpi_define_macros)

mpilinker = mpicompiler
compiler = None

scalapack = False
libvdwxc = False
hdf5 = False

# User provided customizations:
exec(open(customize).read())

if platform_id != '':
    my_platform = distutils.util.get_platform() + '-' + platform_id

    def my_get_platform():
        return my_platform
    distutils.util.get_platform = my_get_platform

if compiler is not None:
    msg += ['* Compiling gpaw with %s' % compiler]
    # A hack to change the used compiler and linker:
    vars = get_config_vars()
    if remove_default_flags:
        for key in ['BASECFLAGS', 'CFLAGS', 'OPT', 'PY_CFLAGS',
                    'CCSHARED', 'CFLAGSFORSHARED', 'LINKFORSHARED',
                    'LIBS', 'SHLIBS']:
            if key in vars:
                value = vars[key].split()
                # remove all gcc flags (causing problems with other compilers)
                for v in list(value):
                    value.remove(v)
                vars[key] = ' '.join(value)
    for key in ['CC', 'LDSHARED']:
        if key in vars:
            value = vars[key].split()
            # first argument is the compiler/linker.  Replace with mpicompiler:
            value[0] = compiler
            vars[key] = ' '.join(value)

custom_interpreter = False
# Check the command line so that custom interpreter is build only with
# 'build', 'build_ext', or 'install':
if mpicompiler is not None:
    for cmd in ['build', 'build_ext', 'install']:
        if cmd in sys.argv:
            custom_interpreter = True
            break


if scalapack:
    define_macros.append(('GPAW_WITH_SL', '1'))
    msg.append('* Compiling with ScaLapack')


if libvdwxc:
    define_macros.append(('GPAW_WITH_LIBVDWXC', '1'))
    msg.append('* Compiling with libvdwxc')


# distutils clean does not remove the _gpaw.so library and gpaw-python
# binary so do it here:
plat = distutils.util.get_platform()
msg += ['* Architecture: ' + plat]
plat = plat + '-' + sys.version[0:3]
gpawso = 'build/lib.%s/' % plat + '_gpaw.so'
gpawbin = 'build/bin.%s/' % plat + 'gpaw-python'
if 'clean' in sys.argv:
    if os.path.isfile(gpawso):
        print('removing ', gpawso)
        os.remove(gpawso)
    if os.path.isfile(gpawbin):
        print('removing ', gpawbin)
        os.remove(gpawbin)

sources = glob('c/*.c') + ['c/bmgs/bmgs.c']
sources = sources + glob('c/xc/*.c')
# Make build process deterministic (for "reproducible build" in debian)
sources.sort()

check_dependencies(sources)

extension = Extension('_gpaw',
                      sources,
                      libraries=libraries,
                      library_dirs=library_dirs,
                      include_dirs=include_dirs,
                      define_macros=define_macros,
                      undef_macros=undef_macros,
                      extra_link_args=extra_link_args,
                      extra_compile_args=extra_compile_args,
                      runtime_library_dirs=runtime_library_dirs,
                      extra_objects=extra_objects)

extensions = [extension]

if hdf5:
    hdf5_sources = ['c/hdf5.c']
    define_macros.append(('GPAW_WITH_HDF5', '1'))
    msg.append('* Compiling with HDF5')

    hdf5_extension = Extension('_gpaw_hdf5',
                               hdf5_sources,
                               libraries=libraries,
                               library_dirs=library_dirs,
                               include_dirs=include_dirs,
                               define_macros=define_macros,
                               undef_macros=undef_macros,
                               extra_link_args=extra_link_args,
                               extra_compile_args=extra_compile_args,
                               runtime_library_dirs=runtime_library_dirs,
                               extra_objects=extra_objects)
    extensions.append(hdf5_extension)

files = ['gpaw-analyse-basis', 'gpaw-basis', 'gpaw-install-setups',
         'gpaw-mpisim', 'gpaw-plot-parallel-timings', 'gpaw-runscript',
         'gpaw-setup', 'gpaw-test', 'gpaw-upfplot', 'gpaw']
scripts = [join('tools', script) for script in files]

write_configuration(define_macros, include_dirs, libraries, library_dirs,
                    extra_link_args, extra_compile_args,
                    runtime_library_dirs, extra_objects, mpicompiler,
                    mpi_libraries, mpi_library_dirs, mpi_include_dirs,
                    mpi_runtime_library_dirs, mpi_define_macros)

description = 'An electronic structure code based on the PAW method'

kwargs = dict(
    name='gpaw',
    version=version,
    description=description,
    maintainer='GPAW-community',
    maintainer_email='gpaw-developers@listserv.fysik.dtu.dk',
    url='http://wiki.fysik.dtu.dk/gpaw',
    license='GPLv3+',
    platforms=['unix'],
    packages=packages,
    long_description=long_description)


class sdist(_sdist):
    """Fix distutils.
    
    Distutils insists that there should be a README or README.txt,
    but GitLab.com needs README.rst in order to parse it as reStructureText."""
    
    def warn(self, msg):
        if msg.startswith('standard file not found: should have one of'):
            self.filelist.append('README.rst')
        else:
            _sdist.warn(self, msg)

            
setup(ext_modules=extensions,
      scripts=scripts,
      cmdclass={'sdist': sdist},
      **kwargs)
      

class build_scripts(_build_scripts):
    # Python 3's version will try to read the first line of gpaw-python
    # because it thinks it is a Python script that need an adjustment of
    # the Python version.  We skip this in our version.
    def copy_scripts(self):
        self.mkpath(self.build_dir)
        outfiles = []
        updated_files = []
        script = self.scripts[0]
        outfile = os.path.join(self.build_dir, os.path.basename(script))
        outfiles.append(outfile)
        updated_files.append(outfile)
        self.copy_file(script, outfile)
        return outfiles, updated_files
        
        
if custom_interpreter:
    scripts = ['build/bin.%s/' % plat + 'gpaw-python']
    error, par_msg = build_interpreter(
        define_macros, include_dirs, libraries,
        library_dirs, extra_link_args, extra_compile_args,
        runtime_library_dirs, extra_objects,
        mpicompiler, mpilinker, mpi_libraries,
        mpi_library_dirs,
        mpi_include_dirs,
        mpi_runtime_library_dirs, mpi_define_macros)
    msg += par_msg
    # install also gpaw-python
    if 'install' in sys.argv and error == 0:
        setup(cmdclass={'build_scripts': build_scripts},
              ext_modules=[extension],
              scripts=scripts,
              **kwargs)

else:
    msg += ['* Only a serial version of gpaw was built!']

# Messages make sense only when building
if 'build' in sys.argv or 'build_ext' in sys.argv or 'install' in sys.argv:
    for line in msg:
        print(line)
