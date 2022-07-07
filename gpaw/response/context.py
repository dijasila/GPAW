from ase.utils import IOContext
from ase.utils.timing import Timer

from gpaw import disable_dry_run
from gpaw.calculator import GPAW
import gpaw.mpi as mpi


def new_context(txt, world, timer):
    timer = timer or Timer()
    return ResponseContext(txt=txt, timer=timer, world=world)


def calc_and_context(calc, txt, world, timer):
    context = new_context(txt, world, timer)
    with context.timer('Read ground state'):
        if not isinstance(calc, GPAW):
            print('Reading ground state calculation:\n  %s' % calc,
                  file=context.fd)
            with disable_dry_run():
                calc = GPAW(calc, communicator=mpi.serial_comm)
        else:
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
