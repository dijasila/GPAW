from gpaw.io import Writer

from gpaw.lcaotddft.observer import TDDFTObserver


class WaveFunctionWriter(TDDFTObserver):
    version = 1
    ulmtag = 'WF'

    def __init__(self, filename, paw, interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        if len(paw.wfs.kpt_u) != 1:
            raise NotImplementedError('K-points not tested!')
        if paw.wfs.bd.comm.size != 1:
            raise NotImplementedError('Band parallelization not tested!')

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
        w = self.writer.child('wave_functions')
        paw.wfs.write_wave_functions(w)
        paw.wfs.write_occupations(w)
        self.writer.sync()

    def __del__(self):
        self.writer.close()
        TDDFTObserver.__del__(self)
