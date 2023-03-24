# Copyright (C) 2006 CSC-Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.
import os
import sys
import re
import shlex
from sysconfig import get_config_var, get_platform
from stat import ST_MTIME


def config_args(key):
    return shlex.split(get_config_var(key))


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
                        runtime_library_dirs, extra_objects, compiler):

    # Write the compilation configuration into a file
    try:
        out = open('configuration.log', 'w')
    except IOError as x:
        print(x)
        return
    print("Current configuration", file=out)
    print("compiler", compiler, file=out)
    print("libraries", libraries, file=out)
    print("library_dirs", library_dirs, file=out)
    print("include_dirs", include_dirs, file=out)
    print("define_macros", define_macros, file=out)
    print("extra_link_args", extra_link_args, file=out)
    print("extra_compile_args", extra_compile_args, file=out)
    print("runtime_library_dirs", runtime_library_dirs, file=out)
    print("extra_objects", extra_objects, file=out)
    out.close()


def build_interpreter(
        compiler, extension, extension_objects, *,
        build_temp, build_bin, debug):
    exename = compiler.executable_filename('gpaw-python')
    print(f'building {repr(exename)} executable', flush=True)

    macros = extension.define_macros.copy()
    macros.append(('GPAW_INTERPRETER', '1'))
    for undef in extension.undef_macros:
        macros.append((undef,))

    # Compile the sources that define GPAW_INTERPRETER
    sources = ['c/_gpaw.c']
    objects = compiler.compile(
        sources,
        output_dir=str(build_temp),
        macros=macros,
        include_dirs=extension.include_dirs,
        debug=debug,
        extra_postargs=extension.extra_compile_args)
    # Note: we recompiled _gpaw.o
    # This object file is already included in extra_objects
    objects = extension_objects

    # Note: LDFLAGS and LIBS go together, but depending on the platform,
    # it might be unnecessary to include them
    extra_preargs = config_args('LDFLAGS')
    extra_postargs = (config_args('BLDLIBRARY')
                      + config_args('LIBS')
                      + config_args('LIBM')
                      + extension.extra_link_args
                      + config_args('LINKFORSHARED'))

    # Link the custom interpreter
    compiler.link_executable(
            objects, exename,
            output_dir=str(build_bin),
            extra_preargs=extra_preargs,
            libraries=extension.libraries,
            library_dirs=extension.library_dirs,
            runtime_library_dirs=extension.runtime_library_dirs,
            extra_postargs=extra_postargs,
            debug=debug,
            target_lang=extension.language)

    return build_bin / exename
