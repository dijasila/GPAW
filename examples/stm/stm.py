# -*- coding: utf-8 -*-
from ase import *
from gpaw import *

atoms = Calculator('Al100.gpw', txt=None).get_atoms()
stm = STM(atoms, symmetries=[0, 1, 2])
c = stm.get_averaged_current(2.5)
h = stm.scan(c)
print u'Min: %.2f Å, Max: %.2f Å' % (h.min(), h.max())
import pylab as p
p.contour(h)
p.show()

