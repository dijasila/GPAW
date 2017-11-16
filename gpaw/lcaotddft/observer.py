from gpaw.analyse.observers import Observer


class TDDFTObserver(Observer):

    def __init__(self, paw, interval):
        Observer.__init__(self, interval)
        assert hasattr(paw, 'time') and hasattr(paw, 'niter'), 'Use TDDFT'
        self.timer = paw.timer
        paw.attach(self, interval, paw)

    def update(self, paw):
        self.timer.start('%s update' % self.__class__.__name__)
        self._update(paw)
        self.timer.stop('%s update' % self.__class__.__name__)

    def _update(self, paw):
        raise RuntimeError('Virtual member function called')
