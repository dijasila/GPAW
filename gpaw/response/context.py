from ase.utils import IOContext
from ase.utils.timing import Timer


def new_context(txt, world, timer):
    timer = timer or Timer()
    return ResponseContext(txt=txt, timer=timer, world=world)


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
