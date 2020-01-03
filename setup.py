#!/usr/bin/env python
# Copyright (C) 2003-2016  CAMP
# Please see the accompanying LICENSE file for further information.

import distutils.util
from distutils.sysconfig import get_config_vars
import os
import re
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import subprocess
import sys
from pathlib import Path

from config import check_dependencies, write_configuration, build_interpreter


assert sys.version_info >= (3, 5)

# Get the current version number:
txt = Path('gpaw/__init__.py').read_text()
version = re.search("__version__ = '(.*)'", txt).group(1)

description = 'GPAW: DFT and beyond within the projector-augmented wave method'
long_description = """\
GPAW is a density-functional theory (DFT) Python code based on the
projector-augmented wave (PAW) method and the atomic simulation environment
(ASE). It uses plane-waves, atom-centered basis-functions or real-space
uniform grids combined with multigrid methods."""

remove_default_flags = False
if '--remove-default-flags' in sys.argv:
    remove_default_flags = True
    sys.argv.remove('--remove-default-flags')

for i, arg in enumerate(sys.argv):
    if arg.startswith('--customize='):
        custom = arg.split('=')[1]
        raise DeprecationWarning(
            'Please set GPAW_CONFIG={custom} or place {custom} in '
            '~/.gpaw/siteconfig.py'.format(custom=custom))

libraries = ['xc']
library_dirs = []
include_dirs = []
extra_link_args = []
extra_compile_args = ['-Wall', '-Wno-unknown-pragmas', '-std=c99']
runtime_library_dirs = []
extra_objects = []
define_macros = [('NPY_NO_DEPRECATED_API', '7'),
                 ('GPAW_NO_UNDERSCORE_CBLACS', '1'),
                 ('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
undef_macros = ['NDEBUG']

mpi_libraries = []
mpi_library_dirs = []
mpi_include_dirs = []
mpi_runtime_library_dirs = []
mpi_define_macros = []

parallel_python_interpreter = False
compiler = None
noblas = False
fftw = False
scalapack = False
libvdwxc = False
elpa = False
mpicompiler = 'mpicc'
mpilinker = 'mpicc'

error = subprocess.call(['which', 'mpicc'], stdout=subprocess.PIPE)
if error:
    mpicompiler = None
    mpilinker = None

# Search and store current git hash if possible
try:
    from ase.utils import search_current_git_hash
    githash = search_current_git_hash('gpaw')
    if githash is not None:
        define_macros += [('GPAW_GITHASH', githash)]
    else:
        print('.git directory not found. GPAW git hash not written.')
except ImportError:
    print('ASE not found. GPAW git hash not written.')

# User provided customizations:
for siteconfig in [os.environ.get('GPAW_CONFIG'),
                   'siteconfig.py',
                   '~/.gpaw/siteconfig.py']:
    if siteconfig is not None:
        path = Path(siteconfig).expanduser()
        if path.is_file():
            print('Reading configuration from', path)
            exec(path.read_text())
            break
else:
    if not noblas:
        libraries.append('blas')

if not parallel_python_interpreter and mpicompiler:
    # Build MPI-interface into _gpaw.so:
    compiler = mpicompiler
    define_macros += [('PARALLEL', '1')]

plat = distutils.util.get_platform()
platform_id = os.getenv('CPU_ARCH')
if platform_id:
    plat += '-' + platform_id

    def my_get_platform():
        return plat

    distutils.util.get_platform = my_get_platform

if compiler is not None:
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

for flag, name in [(noblas, 'GPAW_WITHOUT_BLAS'),
                   (fftw, 'GPAW_WITH_FFTW'),
                   (scalapack, 'GPAW_WITH_SL'),
                   (libvdwxc, 'GPAW_WITH_LIBVDWXC'),
                   (elpa, 'GPAW_WITH_ELPA')]:
    if flag:
        define_macros.append((name, '1'))

sources = [Path('c/bmgs/bmgs.c')]
sources += Path('c').glob('*.c')
sources += Path('c/xc').glob('*.c')
# Make build process deterministic (for "reproducible build")
sources = [str(source) for source in sources]
sources.sort()

check_dependencies(sources)

extensions = [Extension('_gpaw',
                        sources,
                        libraries=libraries,
                        library_dirs=library_dirs,
                        include_dirs=include_dirs,
                        define_macros=define_macros,
                        undef_macros=undef_macros,
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                        runtime_library_dirs=runtime_library_dirs,
                        extra_objects=extra_objects)]

write_configuration(define_macros, include_dirs, libraries, library_dirs,
                    extra_link_args, extra_compile_args,
                    runtime_library_dirs, extra_objects, mpicompiler,
                    mpi_libraries, mpi_library_dirs, mpi_include_dirs,
                    mpi_runtime_library_dirs, mpi_define_macros)


class build_ext(_build_ext):
    def run(self):
        import numpy as np
        self.include_dirs.append(np.get_include())

        _build_ext.run(self)

        if parallel_python_interpreter:
            # Also build gpaw-python:
            error = build_interpreter(
                define_macros, include_dirs, libraries,
                library_dirs, extra_link_args, extra_compile_args,
                runtime_library_dirs, extra_objects,
                mpicompiler, mpilinker, mpi_libraries,
                mpi_library_dirs,
                mpi_include_dirs,
                mpi_runtime_library_dirs, mpi_define_macros)
            assert error == 0


files = ['gpaw-analyse-basis', 'gpaw-basis',
         'gpaw-mpisim', 'gpaw-plot-parallel-timings', 'gpaw-runscript',
         'gpaw-setup', 'gpaw-upfplot']
scripts = [str(Path('tools') / script) for script in files]

if parallel_python_interpreter:
    major, minor = sys.version_info[:2]
    scripts.append('build/bin.{plat}-{major}.{minor}/gpaw-python'
                   .format(plat=plat, major=major, minor=minor))

setup(name='gpaw',
      version=version,
      description=description,
      long_description=long_description,
      maintainer='GPAW-community',
      maintainer_email='gpaw-users@listserv.fysik.dtu.dk',
      url='https://wiki.fysik.dtu.dk/gpaw',
      license='GPLv3+',
      platforms=['unix'],
      packages=find_packages(),
      entry_points={'console_scripts': ['gpaw = gpaw.cli.main:main']},
      setup_requires=['numpy'],
      install_requires=['ase>=3.18.0'],
      ext_modules=extensions,
      scripts=scripts,
      cmdclass={'build_ext': build_ext},
      classifiers=[
          'Development Status :: 6 - Mature',
          'License :: OSI Approved :: '
          'GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Physics'])
