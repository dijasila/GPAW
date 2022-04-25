# Copyright (C) 2006 CSC-Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.
import os
import sys
import re
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
                      runtime_library_dirs, extra_objects,
                      mpicompiler, mpilinker, mpi_libraries, mpi_library_dirs,
                      mpi_include_dirs, mpi_runtime_library_dirs,
                      mpi_define_macros):

    # Build custom interpreter which is used for parallel calculations

    cfgDict = get_config_vars()
    plat = get_platform() + '-{maj}.{min}'.format(maj=sys.version_info[0],
                                                  min=sys.version_info[1])

    cfiles = glob('c/[a-zA-Z_]*.c') + ['c/bmgs/bmgs.c']
    cfiles += glob('c/xc/*.c')
    # Make build process deterministic (for "reproducible build" in debian)
    # XXX some of this is duplicated in setup.py!  Why do the same thing twice?
    cfiles.sort()

    sources = ['c/bc.c', 'c/mpi.c', 'c/_gpaw.c',
               'c/operators.c', 'c/woperators.c', 'c/transformers.c',
               'c/elpa.c',
               'c/blacs.c', 'c/utilities.c', 'c/xc/libvdwxc.c']
    objects = ' '.join(['build/temp.%s/' % plat + x[:-1] + 'o'
                        for x in cfiles])

    if not os.path.isdir('build/bin.{}/'.format(plat)):
        os.makedirs('build/bin.{}/'.format(plat))
    exefile = 'build/bin.{}/gpaw-python'.format(plat)

    libraries += mpi_libraries
    library_dirs += mpi_library_dirs
    define_macros += mpi_define_macros
    include_dirs += mpi_include_dirs
    runtime_library_dirs += mpi_runtime_library_dirs

    define_macros.append(('PARALLEL', '1'))
    define_macros.append(('GPAW_INTERPRETER', '1'))
    macros = ' '.join(['-D%s=%s' % x for x in define_macros if x[0].strip()])

    include_dirs.append(cfgDict['INCLUDEPY'])
    include_dirs.append(cfgDict['CONFINCLUDEPY'])
    includes = ' '.join(['-I' + incdir for incdir in include_dirs])

    library_dirs.append(cfgDict['LIBPL'])
    lib_dirs = ' '.join(['-L' + lib for lib in library_dirs])

    libs = ' '.join(['-l' + lib for lib in libraries if lib.strip()])
    # LIBDIR/INSTSONAME will point at the static library if that is how
    # Python was compiled:
    lib = Path(cfgDict['LIBDIR']) / cfgDict['INSTSONAME']
    if lib.is_file():
        libs += ' {}'.format(lib)
    else:
        libs += ' ' + cfgDict.get('BLDLIBRARY',
                                  '-lpython{}'.format(cfgDict['VERSION']))
    libs = ' '.join([libs, cfgDict['LIBS'], cfgDict['LIBM']])

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

    runtime_libs = ' '.join([runtime_lib_option + lib
                             for lib in runtime_library_dirs])

    extra_link_args.append(cfgDict['LDFLAGS'])

    if sys.platform in ['aix5', 'aix6']:
        extra_link_args.append(cfgDict['LINKFORSHARED'].replace(
            'Modules', cfgDict['LIBPL']))
    elif sys.platform == 'darwin':
        # On a Mac, it is important to preserve the original compile args.
        # This should probably always be done ?!?
        extra_compile_args.append(cfgDict['CFLAGS'])
        extra_link_args.append(cfgDict['LINKFORSHARED'])
    else:
        extra_link_args.append(cfgDict['LINKFORSHARED'])

    extra_compile_args.append('-fPIC')

    # Compile the parallel sources
    for src in sources:
        obj = 'build/temp.{}/'.format(plat) + src[:-1] + 'o'
        cmd = ('{} {} {} {} -o {} -c {} ').format(
               mpicompiler,
               macros,
               ' '.join(extra_compile_args),
               includes,
               obj,
               src)
        print(cmd)
        error = os.system(cmd)
        if error != 0:
            return error

    # Link the custom interpreter
    cmd = ('{} -o {} {} {} {} {} {} {}').format(
           mpilinker,
           exefile,
           objects,
           ' '.join(extra_objects),
           lib_dirs,
           libs,
           runtime_libs,
           ' '.join(extra_link_args))

    print(cmd)
    error = os.system(cmd)
    return error
