"""Test of H5PY built-in interface to HDF5.
"""

import sys
import numpy as np

from gpaw.mpi import world
from gpaw.utilities import devnull, hdf5

if __name__ in ['__main__', '__builtin__']:
    if not hdf5(True):
        print('Not built with H5PY. Test does not apply.')
    else:
        if __name__ == '__builtin__':
            stream = open('ut_h5py.log', 'w', buffering=0)
        elif world.rank == 0:
            stream = sys.stdout
        else:
            stream = devnull

        from gpaw.h5py.tests import runtests
        testresult = runtests(stream=stream, verbosity=2)

        # Provide feedback on failed tests if imported by test.py
        if __name__ == '__builtin__' and not testresult.wasSuccessful():
            raise SystemExit('Test failed. Check ut_h5py.log for details.')

