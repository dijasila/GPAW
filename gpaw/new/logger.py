from gpaw.mpi import MPIComm, world
import sys
import os
from pathlib import Path
from gpaw.utilities.memory import maxrss


class Logger:
    def __init__(self,
                 filename='-',
                 comm: MPIComm = None):
        comm = comm or world

        if comm.rank > 0 or filename is None:
            self.fd = open(os.devnull, 'w')
            self.close_fd = True
        elif filename == '-':
            self.fd = sys.stdout
            self.close_fd = False
        elif isinstance(filename, (str, Path)):
            self.fd = open(filename, 'w')
            self.close_fd = True
        else:
            self.fd = filename
            self.close_fd = False

    def __del__(self) -> None:
        try:
            mib = maxrss() / 1024**2
        except (NameError, LookupError):
            pass
        else:
            self.fd.write(f'\nMax RSS: {mib:.3f} MiB\n')
        if self.close_fd:
            self.fd.close()

    def __call__(self, *args, **kwargs) -> None:
        print(*args, **kwargs, file=self.fd)
