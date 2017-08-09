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


from __future__ import print_function
import sys
import marshal
import imp
import pickle

try:
    # If we start from gpaw-python, we already have _gpaw.
    # Else, _gpaw is only defined later
    import _gpaw
except ImportError:
    world = None
else:
    world = _gpaw.Communicator()


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
    def __init__(self, module_cache):
        self.module_cache = module_cache

    def find_module(self, fullname, path):
        moduleinfo = imp.find_module(fullname.split('.')[-1], path)
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
        sys.meta_path = [OurFinder(self.module_cache)]

    def __exit__(self, *args):
        assert len(sys.meta_path) == 1
        sys.meta_path = self.oldmetapath

        # Restore import loader to its former glory:
        if world.rank == 0:
            self.broadcast()

        self.module_cache = {} if world.rank == 0 else None

    def broadcast(self):
        if world.rank == 0:
            print('0 send {} modules'.format(len(self.module_cache)))
        else:
            print('rank', world.rank, 'recv', self.module_cache)
        self.module_cache = broadcast(self.module_cache)
        if world.rank != 0:
            print('recvd {} modules'.format(len(self.module_cache)))


class NoBroadCaster:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


if world is None:
    globally_broadcast_imports = NoBroadCaster()
else:
    globally_broadcast_imports = BroadCaster()
