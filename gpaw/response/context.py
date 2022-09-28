from pathlib import Path
from ase.utils import IOContext
from ase.utils.timing import Timer

from gpaw import disable_dry_run
from gpaw import GPAW
import gpaw.mpi as mpi


def new_context(txt, world, timer):
    timer = timer or Timer()
    return ResponseContext(txt=txt, timer=timer, world=world)


def calc_and_context(calc, txt, world, timer):
    context = new_context(txt, world, timer)
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
    def __init__(self, txt, timer, world):
        self.iocontext = IOContext()
        self.fd = self.iocontext.openfile(txt, world)
        self.timer = timer
        self.world = world

    def close(self):
        self.iocontext.close()

    def __del__(self):
        self.close()

    def with_txt(self, txt):
        return new_context(txt=txt, world=self.world, timer=self.timer)

    def print(self, *args, flush=True):
        print(*args, file=self.fd, flush=flush)
