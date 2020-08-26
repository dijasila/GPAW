import importlib
from gpaw.mpi import world, broadcast

objects = {}


def start():
    if world.rank == 0:
        return
    while True:
        stuff = broadcast(None)
        if stuff == -1:
            raise SystemExit
        call(*stuff)


def stop():
    broadcast(-1)


def create(id):
    return objects[id]


class ParallelObject:
    def __init__(self, obj, id):
        self._obj = obj
        self._id = id

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if hasattr(attr, '__call__'):
            return ParallelFunction(self, name)
        assert 0, name

    def __reduce__(self):
        return (create, (self._id,))


class ParallelFunction:
    def __init__(self, pobj, name):
        self.pobj = pobj
        self.name = name

    def __call__(self, *args, **kwargs):
        stuff = (self.pobj._id, self.name, args, kwargs)
        broadcast(stuff)
        return call(*stuff)


def parallel(func, *args, **kwargs):
    stuff = (func.__module__, func.__name__, args, kwargs)
    broadcast(stuff)
    return call(*stuff)


def call(module, name, args, kwargs):
    if isinstance(module, str):
        attr = getattr(importlib.import_module(module), name)
    else:
        attr = getattr(objects[module], name)
    obj = attr(*args, **kwargs)
    if name == 'GPAW':
        id = len(objects)
        objects[id] = obj
        if world.rank != 0:
            return
        pobj = ParallelObject(obj, id)
        return pobj
    return obj
