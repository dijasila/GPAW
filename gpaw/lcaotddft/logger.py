from time import localtime
from math import log as ln

from ase.utils.timing import timer

from gpaw.analyse.observers import Observer
from gpaw.tddft.units import autime_to_attosec


class Logger(Observer):

    def __init__(self, paw):
        Observer.__init__(self)
        assert hasattr(paw, 'time') and hasattr(paw, 'niter'), 'Use TDDFT'
        paw.attach(self, 1, paw)

    def update(self, paw):
        density = paw.density
        norm = density.finegd.integrate(density.rhot_g)
        T = localtime()
        paw.log('iter: %3d  %02d:%02d:%02d %11.2f   %9.1f' %
                (paw.niter, T[3], T[4], T[5],
                 paw.time * autime_to_attosec,
                 ln(abs(norm) + 1e-16) / ln(10)))
