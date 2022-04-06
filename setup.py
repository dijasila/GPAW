#!/usr/bin/env python
# Copyright (C) 2003-2020  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import re
import sys
from pathlib import Path
from subprocess import PIPE, run
from sysconfig import get_platform

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install

from config import build_interpreter, check_dependencies, write_configuration

assert sys.version_info >= (3, 7)

# Get the current version number:
txt = Path('gpaw/__init__.py').read_text()
version = re.search("__version__ = '(.*)'", txt)[1]
ase_version_required = re.search("__ase_version_required__ = '(.*)'", txt)[1]

description = 'GPAW: DFT and beyond within the projector-augmented wave method'
long_description = Path('README.rst').read_text()

for i, arg in enumerate(sys.argv):
    if arg.startswith('--customize='):
        custom = arg.split('=')[1]
        raise DeprecationWarning(
            f'Please set GPAW_CONFIG={custom} or place {custom} in ' +
            '~/.gpaw/siteconfig.py')

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
nolibxc = False
fftw = False
scalapack = False
libvdwxc = False
elpa = False

if os.name != 'nt' and run(['which', 'mpicc'], stdout=PIPE).returncode == 0:
    mpicompiler = 'mpicc'
else:
    mpicompiler = None

mpilinker = mpicompiler

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
gpaw_config = os.environ.get('GPAW_CONFIG')
if gpaw_config and not Path(gpaw_config).is_file():
    raise FileNotFoundError(gpaw_config)
for siteconfig in [gpaw_config,
                   'siteconfig.py',
                   '~/.gpaw/siteconfig.py']:
    if siteconfig is not None:
        path = Path(siteconfig).expanduser()
        if path.is_file():
            print('Reading configuration from', path)
            exec(path.read_text())
            break
else:  # no break
    if not noblas:
        libraries.append('blas')

if not parallel_python_interpreter and mpicompiler:
    # Build MPI-interface into _gpaw.so:
    compiler = mpicompiler
    define_macros += [('PARALLEL', '1')]

platform_id = os.getenv('CPU_ARCH')
if platform_id:
    os.environ['_PYTHON_HOST_PLATFORM'] = get_platform() + '-' + platform_id

if compiler is not None:
    # A hack to change the used compiler and linker:
    try:
        # distutils is deprecated and will be removed in 3.12
        from distutils.sysconfig import get_config_vars
    except ImportError:
        from sysconfig import get_config_vars

    # If CC is set then the following hack will not work
    assert not os.environ.get('CC'), 'Please unset CC'

    vars = get_config_vars()
    for key in ['CC', 'LDSHARED']:
        if key in vars:
            value = vars[key].split()
            # first argument is the compiler/linker.  Replace with mpicompiler:
            value[0] = compiler
            vars[key] = ' '.join(value)

for flag, name in [(noblas, 'GPAW_WITHOUT_BLAS'),
                   (nolibxc, 'GPAW_WITHOUT_LIBXC'),
                   (fftw, 'GPAW_WITH_FFTW'),
                   (scalapack, 'GPAW_WITH_SL'),
                   (libvdwxc, 'GPAW_WITH_LIBVDWXC'),
                   (elpa, 'GPAW_WITH_ELPA')]:
    if flag:
        define_macros.append((name, '1'))

sources = [Path('c/bmgs/bmgs.c')]
sources += Path('c').glob('*.c')
sources += Path('c/xc').glob('*.c')
if nolibxc:
    for name in ['libxc.c', 'm06l.c',
                 'tpss.c', 'revtpss.c', 'revtpss_c_pbe.c',
                 'xc_mgga.c']:
        sources.remove(Path(f'c/xc/{name}'))
    if 'xc' in libraries:
        libraries.remove('xc')

# Make build process deterministic (for "reproducible build")
sources = [str(source) for source in sources]
sources.sort()

check_dependencies(sources)

# Convert Path objects to str:
library_dirs = [str(dir) for dir in library_dirs]
include_dirs = [str(dir) for dir in include_dirs]

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
            include_dirs.append(np.get_include())
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


def copy_gpaw_python(cmd, dir: str) -> None:
    major, minor = sys.version_info[:2]
    plat = get_platform() + f'-{major}.{minor}'
    source = f'build/bin.{plat}/gpaw-python'
    target = os.path.join(dir, 'gpaw-python')
    cmd.copy_file(source, target)


class install(_install):
    def run(self):
        _install.run(self)
        copy_gpaw_python(self, self.install_scripts)


class develop(_develop):
    def run(self):
        _develop.run(self)
        copy_gpaw_python(self, self.script_dir)


cmdclass = {'build_ext': build_ext}
if parallel_python_interpreter:
    cmdclass['install'] = install
    cmdclass['develop'] = develop

files = ['gpaw-analyse-basis', 'gpaw-basis',
         'gpaw-plot-parallel-timings', 'gpaw-runscript',
         'gpaw-setup', 'gpaw-upfplot']
scripts = [str(Path('tools') / script) for script in files]


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
      install_requires=[f'ase>={ase_version_required}',
                        'scipy>=1.2.0'],
      ext_modules=extensions,
      scripts=scripts,
      cmdclass=cmdclass,
      classifiers=[
          'Development Status :: 6 - Mature',
          'License :: OSI Approved :: '
          'GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Scientific/Engineering :: Physics'])
