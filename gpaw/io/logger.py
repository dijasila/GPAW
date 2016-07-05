from __future__ import print_function
import os
import sys
import time
import platform

import numpy as np
import ase
from ase import __version__ as ase_version
from ase.utils import convert_string_to_fd

import gpaw
import _gpaw
from gpaw.utilities.memory import maxrss
from gpaw import dry_run, extra_parameters


class GPAWLogger(object):
    """Class for handling all text output."""
    def __init__(self, world):
        self.world = world
        
        self.verbose = False
        self._fd = None
        self.oldfd = 42

    @property
    def fd(self):
        return self._fd
        
    @fd.setter
    def fd(self, fd):
        """Set the stream for text output.

        If `txt` is not a stream-object, then it must be one of:

        * None:  Throw output away.
        * '-':  Use stdout (``sys.stdout``) on master, elsewhere throw away.
        * A filename:  Open a new file on master, elsewhere throw away.
        """
        if fd == self.oldfd:
            return
        self.oldfd = fd
        self._fd = convert_string_to_fd(fd, self.world)
        self.header()
        
    def __call__(self, *args, **kwargs):
        flush = kwargs.pop('flush', False)
        print(*args, file=self._fd, **kwargs)
        if flush:
            self._fd.flush()

    def header(self):
        self()
        self('  ___ ___ ___ _ _ _  ')
        self(' |   |   |_  | | | | ')
        self(' | | | | | . | | | | ')
        self(' |__ |  _|___|_____| ', gpaw.__version__)
        self(' |___|_|             ')
        self()

        uname = platform.uname()
        self('User:  ', os.getenv('USER', '???') + '@' + uname[1])
        self('Date:  ', time.asctime())
        self('Arch:  ', uname[4])
        self('Pid:   ', os.getpid())
        self('Python: {0}.{1}.{2}'.format(*sys.version_info[:3]))
        self('gpaw:  ', os.path.dirname(gpaw.__file__))
        
        # Find C-code:
        c = getattr(_gpaw, '__file__', None)
        if not c:
            c = sys.executable
        self('_gpaw: ', os.path.normpath(c))
                  
        self('ase:    %s (version %s)' %
             (os.path.dirname(ase.__file__), ase_version))
        self('numpy:  %s (version %s)' %
             (os.path.dirname(np.__file__), np.version.version))
        try:
            import scipy as sp
            self('scipy:  %s (version %s)' %
                 (os.path.dirname(sp.__file__), sp.version.version))
            # Explicitly deleting SciPy seems to remove garbage collection
            # problem of unknown cause
            del sp
        except ImportError:
            self('scipy:  Not available')
        self('units:  Angstrom and eV')
        self('cores: ', self.world.size)

        if gpaw.debug:
            self('DEBUG MODE')

        if extra_parameters:
            self('Extra parameters:', extra_parameters)

    def __del__(self):
        """Destructor:  Write timing output before closing."""
        if dry_run:
            return
            
        try:
            mr = maxrss()
        except (LookupError, TypeError, NameError):
            # Thing can get weird during interpreter shutdown ...
            mr = 0

        if mr > 0:
            if mr < 1024.0**3:
                self('Memory usage: %.2f MiB' % (mr / 1024.0**2))
            else:
                self('Memory usage: %.2f GiB' % (mr / 1024.0**3))
