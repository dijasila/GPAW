
import numpy as np
import tempfile
import gpaw.h5py as h5py
from gpaw.h5py import tests

class Base(tests.HTest):

    def setUp(self):
        self.f = h5py.File(tempfile.mktemp(), 'w', driver='core', backing_store=False)

    def tearDown(self):
        self.f.close()

