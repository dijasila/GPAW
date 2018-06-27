#!/usr/bin/env python
# Copyright (C) 2003-2017 CAMd
# Please see the accompanying LICENSE file for further information.

import distutils
import distutils.util
import os
import os.path as op
import re
import shutil
import sys
from glob import glob

from distutils.command.build_scripts import build_scripts as _build_scripts
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.develop import develop as _develop
from setuptools.command.easy_install import easy_install as _easy_install

from config import (get_system_config, get_parallel_config,
                    check_dependencies,
                    write_configuration, build_interpreter, get_config_vars)


assert sys.version_info >= (2, 7)

# Get the current version number:
with open('gpaw/__init__.py', 'rb') as fd:
    txt = fd.read().decode('UTF-8')
version = re.search("__version__ = '(.*)'", txt).group(1)

description = 'GPAW: DFT and beyond within the projector-augmented wave method'
long_description = """\
GPAW is a density-functional theory (DFT) Python code based on the
projector-augmented wave (PAW) method and the atomic simulation environment
(ASE). It uses plane-waves, atom-centered basis-functions or real-space
uniform grids combined with multigrid methods."""

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

platform_id = ''

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

get_system_config(define_macros, undef_macros,
                  include_dirs, libraries, library_dirs,
                  extra_link_args, extra_compile_args,
                  runtime_library_dirs, extra_objects,
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

# User provided customizations:
exec(open(customize).read())

if platform_id != '':
    my_platform = distutils.util.get_platform() + '-' + platform_id

    def my_get_platform():
        return my_platform
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

if scalapack:
    define_macros.append(('GPAW_WITH_SL', '1'))

if libvdwxc:
    define_macros.append(('GPAW_WITH_LIBVDWXC', '1'))

# distutils clean does not remove the _gpaw.so library and gpaw-python
# binary so do it here:
plat = distutils.util.get_platform()
plat = plat + '-' + sys.version[0:3]
gpawso = 'build/lib.%s/' % plat + '_gpaw.so'
gpawbin = 'build/bin.%s/' % plat + 'gpaw-python'
if 'clean' in sys.argv:
    if op.isfile(gpawso):
        print('removing ', gpawso)
        os.remove(gpawso)
    if op.isfile(gpawbin):
        print('removing ', gpawbin)
        os.remove(gpawbin)

sources = glob('c/*.c') + ['c/bmgs/bmgs.c']
sources = sources + glob('c/xc/*.c')
# Make build process deterministic (for "reproducible build" in debian)
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

files = ['gpaw-analyse-basis', 'gpaw-basis',
         'gpaw-mpisim', 'gpaw-plot-parallel-timings', 'gpaw-runscript',
         'gpaw-setup', 'gpaw-upfplot']
scripts = [op.join('tools', script) for script in files]
if mpicompiler:
    scripts.append('build/bin.' + plat + '/gpaw-python')

write_configuration(define_macros, include_dirs, libraries, library_dirs,
                    extra_link_args, extra_compile_args,
                    runtime_library_dirs, extra_objects, mpicompiler,
                    mpi_libraries, mpi_library_dirs, mpi_include_dirs,
                    mpi_runtime_library_dirs, mpi_define_macros)


class build_ext(_build_ext):
    def run(self):
        _build_ext.run(self)
        if mpicompiler:
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


class develop(_develop):
    def install_egg_scripts(self, dist):
        if (self.distribution.scripts[-1].endswith('gpaw-python') and
            str(dist).startswith('gpaw')):
            script = self.distribution.scripts.pop()
        else:
            script = None
        _develop.install_egg_scripts(self, dist)
        if script:
            distutils.log.info('Installing gpaw-python to %s', self.script_dir)
            dst = op.join(self.script_dir, 'gpaw-python')
            if op.isfile(dst):
                os.remove(dst)
            shutil.copy(script, dst)
            self.distribution.scripts.append(script)


_install_egg_scripts = _easy_install.install_egg_scripts


def install_egg_scripts(self, dist):
    if not self.exclude_scripts and dist.metadata_isdir('scripts'):
        for script_name in dist.metadata_listdir('scripts'):
            if script_name == 'gpaw-python':
                distutils.log.info('Installing gpaw-python to %s',
                                   self.script_dir)
                src = 'build/bin.' + plat + '/gpaw-python'
                dst = op.join(self.script_dir, 'gpaw-python')
                if op.isfile(dst):
                    os.remove(dst)
                shutil.copy(src, dst)
                continue
            if dist.metadata_isdir('scripts/' + script_name):
                # The "script" is a directory, likely a Python 3
                # __pycache__ directory, so skip it.
                continue
            self.install_script(
                dist, script_name,
                dist.get_metadata('scripts/' + script_name))
    self.install_wrapper_scripts(dist)


_easy_install.install_egg_scripts = install_egg_scripts


class build_scripts(_build_scripts):
    def copy_scripts(self):
        if self.scripts[-1].endswith('gpaw-python'):
            script = self.scripts.pop()
        else:
            script = None
        outfiles, updated_files = _build_scripts.copy_scripts(self)
        if script:
            outfile = op.join(self.build_dir, op.basename(script))
            outfiles.append(outfile)
            updated_files.append(outfile)
            self.copy_file(script, outfile)
            self.scripts.append(script)
        return outfiles, updated_files


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
      entry_points={'console_scripts': ['gpaw=gpaw.cli.main:main']},
      install_requires=['ase>=3.16.0'],
      ext_modules=extensions,
      scripts=scripts,
      cmdclass={'build_ext': build_ext,
                'build_scripts': build_scripts,
                'develop': develop},
      classifiers=[
          'Development Status :: 6 - Mature',
          'License :: OSI Approved :: '
          'GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Physics'])
