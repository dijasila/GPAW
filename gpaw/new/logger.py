from gpaw.mpi import MPIComm, world
import sys
import os
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager


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

        self._indent = ''

    def __del__(self) -> None:
        if self.close_fd:
            self.fd.close()

    def __call__(self, *args) -> None:
        self.fd.write(self._indent)
        print(*args, file=self.fd)

    def comment(self, text):
        print('# ' + '\n# '.join(text.splitlines()), file=self.fd)

    @contextmanager
    def indent(self, text: str) -> Iterator:
        self(text)
        self._indent += '  '
        yield
        self._indent = self._indent[:-2]
