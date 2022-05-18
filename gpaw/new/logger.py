from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import IO

from gpaw.mpi import MPIComm, world
from gpaw.yml import obj2yaml as o2y


class Logger:
    def __init__(self,
                 filename: str | Path | IO[str] | None = '-',
                 comm: MPIComm = None):
        comm = comm or world

        self.fd: IO[str]

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

        self.indentation = ''

    def __del__(self) -> None:
        if self.close_fd:
            self.fd.close()

    @contextlib.contextmanager
    def indent(self, text):
        self(text)
        self.indentation += '  '
        yield
        self.indentation = self.indentation[2:]

    @contextlib.contextmanager
    def comment(self):
        self.indentation += '# '
        yield
        self.indentation = self.indentation[2:]

    def __call__(self, *args, **kwargs) -> None:
        if not self.fd.closed:
            if kwargs:
                for kw, arg in kwargs.items():
                    assert kw not in ['end', 'sep', 'flush', 'file'], kw
                    print(f'{self.indentation}{kw}: {o2y(arg)}',
                          file=self.fd)
            else:
                text = ' '.join(str(arg) for arg in args)
                if self.indentation:
                    text = self.indentation + text.replace(
                        '\n', '\n' + self.indentation)
                print(text, file=self.fd)
