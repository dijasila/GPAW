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
from importlib.machinery import PathFinder, ModuleSpec
import pickle
import marshal

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
        buf = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
    else:
        assert obj is None
        buf = None

    buf = _gpaw.globally_broadcast_bytes(buf)
    newobj = pickle.loads(buf)
    return newobj


class BroadcastLoader:
    def __init__(self, loader, code_cache):
        self.loader = loader
        self.code_cache = code_cache

    def load_module(self, fullname):
        if world.rank == 0:
            print('loadmod0', fullname)
            code = self.loader.get_code(fullname)
            self.code_cache[fullname] = marshal.dumps(code)
            mod = self.loader.load_module(fullname)
            assert fullname in sys.modules
            return mod
        else:
            print('loadmod1', fullname)
            code = marshal.loads(self.code_cache[fullname])
            print('code ok')
            module = type(sys)(fullname)
            print('mod ok')
            exec(code, module.__dict__)
            print('exec ok')
            sys.modules[fullname] = module
            print('mod ok', fullname)
            return module

class BroadcastImporter:
    def __init__(self):
        self.code_cache = {}
        #self.origins = {}

    def find_spec(self, fullname, path=None, target=None):
        if world.rank == 0:
            print('findspec0', fullname)
            spec = PathFinder.find_spec(fullname, path, target)
            if spec is None:
                return None
            spec.loader = BroadcastLoader(spec.loader, self.code_cache)
            return spec
        else:
            print('findspec1', fullname)
            loader = BroadcastLoader(None, self.code_cache)
            refspec = PathFinder.find_spec(fullname, path, target)  # XXX
            retval = ModuleSpec(fullname, loader)
            print(refspec)
            print(retval)
            return retval

    def broadcast(self):
        if world.rank == 0:
            print('bcast {} modules'.format(len(self.code_cache)))
            #print(list(sorted(self.code_cache)))
            broadcast(self.code_cache)
        else:
            self.code_cache = broadcast(None)
            print('recv {} modules'.format(len(self.code_cache)))

    def __enter__(self):
        sys.meta_path.insert(0, self)
        if world.rank != 0:
            self.broadcast()

    def __exit__(self):
        if world.rank == 0:
            self.broadcast()
        self.code_cache = {}
        myself = sys.meta_path.pop(0)
        assert myself is self


globally_broadcast_imports = BroadcastImporter()
#sys.meta_path.insert(0, importer)

#if world.rank != 0:
#    importer.broadcast()

import pyg3t
#import pyg3t
#from pyg3t import util
#import ase

#if world.rank == 0:
#    importer.broadcast()

#import ase.data


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
