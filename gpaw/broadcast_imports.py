"""Provide mechanism to broadcast imports from master to other processes.

This reduces file system strain.

Use:

  with globally_broadcast_imports():
      <execute import statements>

This temporarily overrides the Python import mechanism so that

  1) master executes and caches import metadata and code
  2) import metadata and code are broadcast to all processes
  3) other processes execute the import statements from memory

"""

import sys
import importlib
import importlib.util
from importlib.machinery import PathFinder, ModuleSpec
import marshal
import types

try:
    import _gpaw
except ImportError:
    we_are_gpaw_python = False
else:
    we_are_gpaw_python = hasattr(_gpaw, 'Communicator')

if we_are_gpaw_python:
    world = _gpaw.Communicator()
else:
    world = None


paths = {}
sources = {}


def broadcast(obj):
    if world.rank == 0:
        buf = marshal.dumps(obj)
    else:
        assert obj is None
        buf = None

    buf = _gpaw.globally_broadcast_bytes(buf)
    newobj = marshal.loads(buf)
    return newobj


class BroadcastLoader:
    def __init__(self, spec, module_cache):
        self.module_cache = module_cache
        self.spec = spec

    def load_module(self, fullname):
        if world.rank == 0:
            spec = self.spec

            # Load from file and store in cache:
            code = spec.loader.get_code(fullname)
            searchloc = spec.submodule_search_locations
            metadata = (searchloc, spec.origin)
            self.module_cache[fullname] = (metadata, code)

            metadata, code = self.module_cache[fullname]
            searchloc = metadata[0]
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            exec(code, module.__dict__)
            return module
        else:
            # This is mostly the same as above, but when debugging
            # it is nice to be able to modify master/others separately
            metadata, code = self.module_cache[fullname]
            origin = metadata[1]
            module = importlib.util.module_from_spec(self.spec)
            sys.modules[fullname] = module
            module.__file__ = origin
            exec(code, module.__dict__)
            return module

class BroadcastImporter:
    def __init__(self):
        self.module_cache = {}

    def find_spec(self, fullname, path=None, target=None):
        if world.rank == 0:

            spec = PathFinder.find_spec(fullname, path, target)
            if spec is None:
                return None

            loader = spec.loader
            code = loader.get_code(fullname)
            if code is None:  # C extensions
                return None

            loader = BroadcastLoader(spec, self.module_cache)
            assert fullname == spec.name

            searchloc = spec.submodule_search_locations
            spec = ModuleSpec(fullname, loader, origin=spec.origin,
                              is_package=searchloc is not None)
            if searchloc is not None:
                spec.submodule_search_locations += searchloc
            return spec
        else:
            if not fullname in self.module_cache:
                return PathFinder.find_spec(fullname, path, target)

            searchloc, origin = self.module_cache[fullname][0]
            loader = BroadcastLoader(None, self.module_cache)
            spec = ModuleSpec(fullname, loader, origin=origin,
                              is_package=searchloc is not None)
            if searchloc is not None:
                spec.submodule_search_locations += searchloc
            loader.spec = spec  # XXX loader.loader is still None
            return spec

    def broadcast(self):
        if world.rank == 0:
            print('bcast {} modules'.format(len(self.module_cache)))
            broadcast(self.module_cache)
        else:
            self.module_cache = broadcast(None)
            print('recv {} modules'.format(len(self.module_cache)))

    def __enter__(self):
        sys.meta_path.insert(0, self)
        if world.rank != 0:
            self.broadcast()

    def __exit__(self, *args):
        if world.rank == 0:
            self.broadcast()
        self.module_cache = {}
        myself = sys.meta_path.pop(0)
        assert myself is self


globally_broadcast_imports = BroadcastImporter()


if 0:
    from __future__ import print_function
    import sys
    import marshal
    import imp
    import pickle


    # When running in parallel, the _gpaw module exists right from the
    # start.  Else it may not yet be defined.  So if we have _gpaw, *and*
    # _gpaw defines the Communicator, then this is truly parallel.
    # Otherwise, nothing matters anymore.

    try:
        import _gpaw
    except ImportError:
        we_are_gpaw_python = False
    else:
        we_are_gpaw_python = hasattr(_gpaw, 'Communicator')

    if we_are_gpaw_python:
        world = _gpaw.Communicator()
    else:
        world = None


    def broadcast(obj):
        if world.rank == 0:
            buf = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        else:
            assert obj is None
            buf = None

        buf = _gpaw.globally_broadcast_bytes(buf)
        newobj = pickle.loads(buf)
        return newobj


    class ModuleData:
        def __init__(self, vars, code):
            self.vars = vars
            self.code = code


    class OurFinder:
        def __init__(self, module_cache, module_findcache):
            self.module_cache = module_cache
            self.module_findcache = module_findcache

        def find_module(self, fullname, path):
            if world.rank == 0:
                moduleinfo = imp.find_module(fullname.split('.')[-1], path)
                #print(type(fullname), type(path))
                #print(moduleinfo)
                #if path is not None:
                #    path = tuple(path)
                #self.module_findcache[(fullname, path)] = moduleinfo
            else:
                #if path is not None:
                #    path = tuple(path)
                #moduleinfo = self.module_findcache[(fullname, path)]
                moduleinfo = None
            return OurLoader(moduleinfo, self.module_cache)


    class OurLoader:
        def __init__(self, moduleinfo, module_cache):
            self.moduleinfo = moduleinfo
            self.module_cache = module_cache

        def load_module(self, name):
            if name in sys.modules:
                return sys.modules[name]

            if world.rank == 0:
                return self.load_and_cache(name)
            else:
                return self.load_from_cache(name)

        def load_and_cache(self, name):
            module = self.load_as_normal(name)

            # Some data, like __path__, is not included in the code.
            # We must manually handle these:
            module_vars = {}

            # XXX is this guaranteed to be complete?
            for var in ['__path__', '__package__', '__file__', '__cached__']:
                if hasattr(module, var):
                    module_vars[var] = getattr(module, var)

            # Load module code, if the module actually comes from a file, and
            # the file is a python-file (not e.g. C extensions like .so):
            code = None
            if hasattr(module, '__file__'):
                filename = module.__file__
                if any(filename.endswith(extension)
                       for extension in ['.py', '.pyc', 'pyo']):
                    with open(filename, 'rb') as fd:
                        code = fd.read()

            self.module_cache[name] = ModuleData(module_vars, code)
            return module

        def load_from_cache(self, name):
            module_data = self.module_cache[name]

            if module_data.code is None:
                return self.load_as_normal(name)

            if sys.version_info[0] == 2:
                # Load, ignoring checksum and time stamp.
                # 8 is header length (12 in py3, if we need that someday)
                code = marshal.loads(module_data.code[8:])
            else:
                code = module_data.code

            imp.acquire_lock()  # Required in threaded applications

            print('{} load {}'.format(world.rank, name))
            module = imp.new_module(name)

            # To properly handle circular and submodule imports, we must
            # add the module before executing its code:
            sys.modules[name] = module

            # Set data like __path__, __file__ etc. which are defined
            # by the loader and not the code itself:
            for var in module_data.vars:
                setattr(module, var, module_data.vars[var])

            exec(code, module.__dict__)

            imp.release_lock()
            return module

        def load_as_normal(self, name):
            module = imp.load_module(name, *self.moduleinfo)
            sys.modules[name] = module
            return module


    class BroadCaster:
        def __init__(self):
            self.oldmetapath = None
            self.module_cache = {} if world.rank == 0 else None
            self.module_findcache = {} if world.rank == 0 else None

        def __enter__(self):
            #assert self.oldmetapath is None, self.oldmetapath
            self.oldmetapath = sys.meta_path
            if world.rank != 0:
                # Here we wait for the master process to finish all its
                # imports; the master process will broadcast its results
                # from its __exit__ method, but we slaves need to receive
                # those data in order to start importing.
                self.broadcast()

            # Override standard import finder/loader:
            sys.meta_path = [OurFinder(self.module_cache, self.module_findcache)]

        def __exit__(self, *args):
            assert len(sys.meta_path) == 1
            sys.meta_path = self.oldmetapath

            # Restore import loader to its former glory:
            if world.rank == 0:
                self.broadcast()

            self.module_cache = {} if world.rank == 0 else None
            self.module_findcache = {} if world.rank == 0 else None

        def broadcast(self):
            if world.rank == 0:
                print('0 send {} modules'.format(len(self.module_cache)))
            else:
                print('rank', world.rank, 'recv', self.module_cache)
            self.module_cache = broadcast(self.module_cache)
            self.module_findcache = broadcast(self.module_findcache)
            if world.rank != 0:
                print('recvd {} modules'.format(len(self.module_cache)))


    class NullBroadCaster:
        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


    if world is None:
        globally_broadcast_imports = NullBroadCaster()
    else:
        globally_broadcast_imports = BroadCaster()


def main():
    b = BroadcastImporter()
    import sys
    print(sys.modules)
    b.__enter__()
    print(sys.meta_path)
    print('import')
    import pyg3t
    assert 'pyg3t' in sys.modules
    b.__exit__()

if __name__ == '__main__':
    main()
