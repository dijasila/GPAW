from gpaw.io import Writer

from gpaw.lcaotddft.observer import TDDFTObserver


class WaveFunctionWriter(TDDFTObserver):
    version = 1
    ulmtag = 'WFW'

    def __init__(self, paw, filename, interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        if paw.niter == 0:
            self.writer = Writer(filename, paw.world, mode='w',
                                 tag=self.__class__.ulmtag)
            self.writer.write(version=self.__class__.version)
            self.writer.sync()
        else:
            self.writer = Writer(filename, paw.world, mode='a',
                                 tag=self.__class__.ulmtag)

    def _update(self, paw):
        self.writer.write(time=paw.time, action=paw.action)
        if paw.action == 'kick':
            self.writer.write(kick_strength=paw.kick_strength)
        w = self.writer.child('wave_functions')
        paw.wfs.write_wave_functions(w)
        paw.wfs.write_occupations(w)
        self.writer.sync()

    def __del__(self):
        self.writer.close()
        TDDFTObserver.__del__(self)
