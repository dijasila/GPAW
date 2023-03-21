# Copyright (C) 2006 CSC-Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.
import os
import sys
import re
import shlex
from subprocess import run
from sysconfig import get_config_vars, get_platform
from glob import glob
from pathlib import Path
from stat import ST_MTIME


def mtime(path, name, mtimes):
    """Return modification time.

    The modification time of a source file is returned.  If one of its
    dependencies is newer, the mtime of that file is returned.
    This function fails if two include files with the same name
    are present in different directories."""

    include = re.compile(r'^#\s*include "(\S+)"', re.MULTILINE)

    if name in mtimes:
        return mtimes[name]
    t = os.stat(os.path.join(path, name))[ST_MTIME]
    for name2 in include.findall(open(os.path.join(path, name)).read()):
        path2, name22 = os.path.split(name2)
        if name22 != name:
            t = max(t, mtime(os.path.join(path, path2), name22, mtimes))
    mtimes[name] = t
    return t


def check_dependencies(sources):
    # Distutils does not do deep dependencies correctly.  We take care of
    # that here so that "python setup.py build_ext" always does the right
    # thing!
    mtimes = {}  # modification times

    # Remove object files if any dependencies have changed:
    plat = get_platform() + '-{maj}.{min}'.format(maj=sys.version_info[0],
                                                  min=sys.version_info[1])
    remove = False
    for source in sources:
        path, name = os.path.split(source)
        t = mtime(path + '/', name, mtimes)
        o = 'build/temp.%s/%s.o' % (plat, source[:-2])  # object file
        if os.path.exists(o) and t > os.stat(o)[ST_MTIME]:
            print('removing', o)
            os.remove(o)
            remove = True

    so = 'build/lib.{}/_gpaw.so'.format(plat)
    if os.path.exists(so) and remove:
        # Remove shared object C-extension:
        # print 'removing', so
        os.remove(so)


def write_configuration(define_macros, include_dirs, libraries, library_dirs,
                        extra_link_args, extra_compile_args,
                        runtime_library_dirs, extra_objects, mpicompiler,
                        mpi_libraries, mpi_library_dirs, mpi_include_dirs,
                        mpi_runtime_library_dirs, mpi_define_macros):

    # Write the compilation configuration into a file
    try:
        out = open('configuration.log', 'w')
    except IOError as x:
        print(x)
        return
    print("Current configuration", file=out)
    print("libraries", libraries, file=out)
    print("library_dirs", library_dirs, file=out)
    print("include_dirs", include_dirs, file=out)
    print("define_macros", define_macros, file=out)
    print("extra_link_args", extra_link_args, file=out)
    print("extra_compile_args", extra_compile_args, file=out)
    print("runtime_library_dirs", runtime_library_dirs, file=out)
    print("extra_objects", extra_objects, file=out)
    if mpicompiler is not None:
        print(file=out)
        print("Parallel configuration", file=out)
        print("mpicompiler", mpicompiler, file=out)
        print("mpi_libraries", mpi_libraries, file=out)
        print("mpi_library_dirs", mpi_library_dirs, file=out)
        print("mpi_include_dirs", mpi_include_dirs, file=out)
        print("mpi_define_macros", mpi_define_macros, file=out)
        print("mpi_runtime_library_dirs", mpi_runtime_library_dirs, file=out)
    out.close()


def build_interpreter(define_macros, include_dirs, libraries, library_dirs,
                      extra_link_args, extra_compile_args,
                      runtime_library_dirs, objects,
                      build_temp, build_bin,
                      mpicompiler, mpilinker, mpi_libraries, mpi_library_dirs,
                      mpi_include_dirs, mpi_runtime_library_dirs,
                      mpi_define_macros):
    exename = 'gpaw-python'
    print(f'building {repr(exename)} interpreter', flush=True)

    # Create bin build directory
    if not build_bin.exists():
        print(f'creating {build_bin}', flush=True)
        build_bin.mkdir(parents=True)

    exefile = build_bin / exename

    libraries += mpi_libraries
    library_dirs += mpi_library_dirs
    define_macros += mpi_define_macros
    include_dirs += mpi_include_dirs
    runtime_library_dirs += mpi_runtime_library_dirs

    define_macros.append(('GPAW_INTERPRETER', '1'))

    cfgDict = get_config_vars()

    libs = [f'-l{lib}' for lib in libraries if lib.strip()]
    # LIBDIR/INSTSONAME will point at the static library if that is how
    # Python was compiled:
    lib = Path(cfgDict['LIBDIR']) / cfgDict['INSTSONAME']
    if lib.is_file():
        libs += [lib]
    else:
        libs += [cfgDict.get('BLDLIBRARY',
                             '-lpython{}'.format(cfgDict['VERSION']))]
    libs += shlex.split(cfgDict['LIBS'])
    libs += shlex.split(cfgDict['LIBM'])

    # Hack taken from distutils to determine option for runtime_libary_dirs
    if sys.platform[:6] == 'darwin':
        # MacOSX's linker doesn't understand the -R flag at all
        runtime_lib_option = '-L'
    elif sys.platform[:5] == 'hp-ux':
        runtime_lib_option = '+s -L'
    elif os.popen('mpicc --showme 2> /dev/null', 'r').read()[:3] == 'gcc':
        runtime_lib_option = '-Wl,-R'
    elif os.popen('mpicc -show 2> /dev/null', 'r').read()[:3] == 'gcc':
        runtime_lib_option = '-Wl,-R'
    else:
        runtime_lib_option = '-R'

    runtime_libs = [runtime_lib_option + lib
                    for lib in runtime_library_dirs]
    extra_link_args += shlex.split(cfgDict['LDFLAGS'])

    if sys.platform in ['aix5', 'aix6']:
        extra_link_args += shlex.split(cfgDict['LINKFORSHARED'].replace(
                                       'Modules', cfgDict['LIBPL']))
    elif sys.platform == 'darwin':
        # On a Mac, it is important to preserve the original compile args.
        # This should probably always be done ?!?
        extra_compile_args += shlex.split(cfgDict['CFLAGS'])
        extra_link_args += shlex.split(cfgDict['LINKFORSHARED'])
    else:
        extra_link_args += shlex.split(cfgDict['LINKFORSHARED'])

    extra_compile_args.append('-fPIC')

    # Compile the sources that define GPAW_INTERPRETER
    sources = [Path('c/_gpaw.c')]
    for src in sources:
        obj = build_temp / src.with_suffix('.o')
        run_args = [mpicompiler]
        run_args += [f'-D{name}={value}' for (name, value) in define_macros]
        run_args += [f'-I{dpath}' for dpath in include_dirs]
        run_args += ['-c', str(src)]
        run_args += ['-o', str(obj)]
        run_args += extra_compile_args
        print(' '.join(run_args), flush=True)
        p = run(run_args, check=False, shell=False)
        if p.returncode != 0:
            print(f'error: command {repr(mpicompiler)} failed '
                  f'with exit code {p.returncode}',
                  file=sys.stderr, flush=True)
            sys.exit(1)

    # Link the custom interpreter
    run_args = [mpilinker]
    run_args += extra_link_args
    run_args += objects
    run_args += [f'-L{dpath}' for dpath in library_dirs]
    run_args += [f'{lib}' for lib in libs + runtime_libs]
    run_args += ['-o', str(exefile)]
    print(' '.join(run_args), flush=True)
    p = run(run_args, check=False, shell=False)
    if p.returncode != 0:
        print(f'error: command {repr(mpilinker)} failed '
              f'with exit code {p.returncode}',
              file=sys.stderr, flush=True)
        sys.exit(1)

    return exefile
