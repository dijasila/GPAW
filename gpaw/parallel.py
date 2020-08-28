import atexit
import importlib
from typing import Dict, Any
from gpaw.mpi import world, broadcast

objects: Dict[int, Any] = {}


def start():
    if world.rank == 0:
        atexit.register(broadcast, -1)
        import ase.parallel
        ase.parallel.world = ase.parallel.DummyMPI()
        return
    while True:
        stuff = broadcast(None)
        if stuff == -1:
            raise SystemExit
        if isinstance(stuff, int):
            del objects[stuff]
        else:
            call(*stuff)


def create(id: int) -> Any:
    return objects[id]


class ParallelObject:
    def __init__(self, obj, id: int):
        self._obj = obj
        self._id = id

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if hasattr(attr, '__call__'):
            return ParallelFunction(self, name)
        assert 0, name

    def __reduce__(self):
        return (create, (self._id,))

    def __del__(self):
        broadcast(self._id)


class ParallelFunction:
    def __init__(self, pobj: ParallelObject, name: str):
        self.pobj = pobj
        self.name = name

    def __call__(self, *args, **kwargs):
        stuff = (self.pobj._id, self.name, args, kwargs)
        broadcast(stuff)
        return call(*stuff)


def parallel(func, *args, **kwargs) -> ParallelObject:
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
