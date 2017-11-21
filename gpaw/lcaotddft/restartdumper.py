from gpaw.lcaotddft.observer import TDDFTObserver


class RestartDumper(TDDFTObserver):
    def __init__(self, restart_filename, paw, interval=100):
        TDDFTObserver.__init__(self, paw, interval)
        self.restart_filename = restart_filename

    def _update(self, paw):
        paw.log('%s:' % self.__class__.__name__)
        for obs, n, args, kwargs in paw.observers:
            if (isinstance(obs, TDDFTObserver) and
                hasattr(obs, 'write_restart')):
                paw.log('Write restart for %s' % obs)
                obs.write_restart()
        paw.log('Write GPAW restart')
        paw.write(self.restart_filename, mode='all')
