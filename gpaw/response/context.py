from time import ctime

from inspect import isgeneratorfunction
from functools import wraps
from pathlib import Path

from ase.utils import IOContext
from ase.utils.timing import Timer

from gpaw import disable_dry_run
from gpaw import GPAW
import gpaw.mpi as mpi


def calc_and_context(calc, txt, world, timer):
    context = ResponseContext(txt=txt, world=world, timer=timer)
    with context.timer('Read ground state'):
        try:
            path = Path(calc)
        except TypeError:
            pass
        else:
            print('Reading ground state calculation:\n  %s' % path,
                  file=context.fd)
            with disable_dry_run():
                calc = GPAW(path, communicator=mpi.serial_comm)

    assert calc.wfs.world.size == 1
    return calc, context


class ResponseContext:
    def __init__(self, txt='-', timer=None, world=mpi.world):
        self.world = world
        self.open(txt)
        self.set_timer(timer)

    def open(self, txt):
        self.iocontext = IOContext()
        self.fd = self.iocontext.openfile(txt, self.world)

    def set_timer(self, timer):
        self.timer = timer or Timer()

    def close(self):
        self.iocontext.close()

    def __del__(self):
        self.close()

    def with_txt(self, txt):
        return ResponseContext(txt=txt, world=self.world, timer=self.timer)

    def print(self, *args, flush=True):
        print(*args, file=self.fd, flush=flush)

    def new_txt_and_timer(self, txt, timer=None):
        self.write_timer()
        # Close old output file and create a new
        self.print('Closing')
        self.close()
        self.open(txt)
        self.set_timer(timer)

    def write_timer(self):
        self.timer.write(self.fd)
        self.print('\n%s' % ctime())


class timer:
    """Decorator for timing a method call.
    NB: Includes copy-paste from ase, which is suboptimal...

    Example::

        from gpaw.response.context import timer

        class A:
            def __init__(self, context):
                self.context = context

            @timer('Add two numbers')
            def add(self, x, y):
                return x + y

        """
    def __init__(self, name):
        self.name = name

    def __call__(self, method):
        if isgeneratorfunction(method):
            @wraps(method)
            def new_method(slf, *args, **kwargs):
                gen = method(slf, *args, **kwargs)
                while True:
                    slf.context.timer.start(self.name)
                    try:
                        x = next(gen)
                    except StopIteration:
                        break
                    finally:
                        slf.context.timer.stop()
                    yield x
        else:
            @wraps(method)
            def new_method(slf, *args, **kwargs):
                slf.context.timer.start(self.name)
                x = method(slf, *args, **kwargs)
                try:
                    slf.context.timer.stop()
                except IndexError:
                    pass
                return x
        return new_method
