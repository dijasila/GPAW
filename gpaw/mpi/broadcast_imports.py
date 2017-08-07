import sys
import marshal
import imp

from gpaw.mpi import world, broadcast


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
            #print('had {} already'.format(name))
            return sys.modules[name]


        if world.rank == 0:
            return self.load_and_cache(name)
        else:
            return self.load_from_cache(name)

        return module

    def load_and_cache(self, name):
        module = self.load_as_normal(name)

        # Some data, like __path__, is not included in the code.
        # We must manually handle these:
        module_vars = {}
        for var in ['__path__', '__package__', '__file__']:
            if hasattr(module, var):
                module_vars[var] = getattr(module, var)

        # Load module code, if the module actually comes from a file:
        if hasattr(module, '__file__'):
            with open(module.__file__, 'rb') as fd:
                code = fd.read()
        else:
            # This is a built-in, probably, and the other process will
            # can load it itself
            code = None

        self.module_cache[name] = ModuleData(module_vars, code)
        return module

    def load_from_cache(self, name):
        module_data = self.module_cache[name]

        if '__file__' not in module_data.vars:
            module = self.load_as_normal(name)
            return module

        #print("I'm rank {}. Making new module {}.".
        #      format(world.rank, name))

        # Load, ignoring checksum and time stamp:
        try:
            code = marshal.loads(module_data.code[8:])
        except ValueError as err:
            # Loading Python extensions this way is not
            # supported -- only pyc files.
            # We fall back to the standard loading mechanism.
            if str(err).startswith('bad marshal data'):
                assert name.startswith('_')  # C extensions
                module = self.load_as_normal(name)
                return module

        imp.acquire_lock()  # Required in threaded applications

        module = imp.new_module(name)

        # To properly handle circular and submodule imports, we must
        # add the module before executing its code:
        sys.modules[name] = module

        # Set data like __path__, __file__ etc. which are defined
        # by the loader and not the code itself:
        for var in module_data.vars:
            setattr(module, var, module_data.vars[var])
            exec code in module.__dict__
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
        assert self.oldmetapath is None
        self.oldmetapath = sys.meta_path
        if world.rank != 0:
            self.broadcast()

        sys.meta_path = [OurFinder(self.module_cache)]

    def __exit__(self, *args):
        assert len(sys.meta_path) == 1
        sys.meta_path = self.oldmetapath
        if world.rank == 0:
            self.broadcast()

    def broadcast(self):
        self.module_cache = broadcast(self.module_cache, root=0, comm=world)


def globally_broadcast_imports():
    return BroadCaster()
