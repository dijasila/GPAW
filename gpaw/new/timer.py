from contextlib import contextmanager
from functools import wraps
from io import StringIO


class GlobalTimer:
    def __init__(self):
        self._timers = []

    @contextmanager
    def context(self, timer):
        self._timers.append(timer)
        yield
        # XXX We need to decide what "timer contexts" are,
        # at least that will be necessary if we want a default
        # behaviour which would then be in effect during runs of the
        # test suite.

    def start(self, name):
        self._timers[-1].start(name)

    def stop(self, name=None):
        self._timers[-1].stop(name=name)

    def tostring(self):
        buf = StringIO()
        self._timers[-1].write(out=buf)
        return buf.getvalue()


def trace(meth):
    """Decorator for telling global timer to trace a function or method."""

    modname = meth.__module__
    methname = meth.__qualname__
    name = f'{modname}.{methname}'

    @wraps(meth)
    def wrapper(*args, **kwargs):
        global_timer.start(name)
        try:
            return meth(*args, **kwargs)
        finally:
            global_timer.stop()

    return wrapper


global_timer = GlobalTimer()
