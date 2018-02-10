from gpaw.io import Reader
from gpaw.io import Writer

from gpaw.lcaotddft.observer import TDDFTObserver


class WaveFunctionWriter(TDDFTObserver):
    version = 2
    ulmtag = 'WFW'
    ulmtag_split = ulmtag + 'split'

    def __init__(self, paw, filename, split=False, interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.split = split
        if paw.niter == 0:
            self.writer = Writer(filename, paw.world, mode='w',
                                 tag=self.__class__.ulmtag)
            self.writer.write(version=self.__class__.version)
            self.writer.write(split=self.split)
            self.writer.sync()
        else:
            # Check the earlier file
            reader = Reader(filename)
            assert reader.get_tag() == self.__class__.ulmtag
            assert reader.version == self.__class__.version
            self.split = reader.split  # Use the earlier split value
            reader.close()

            # Append to earlier file
            self.writer = Writer(filename, paw.world, mode='a',
                                 tag=self.__class__.ulmtag)
        if self.split:
            name, ext = tuple(filename.rsplit('.', 1))
            self.split_filename_fmt = name + '-%06d-%s.' + ext

    def _update(self, paw):
        # Write metadata to main writer
        self.writer.write(niter=paw.niter, time=paw.time, action=paw.action)
        if paw.action == 'kick':
            self.writer.write(kick_strength=paw.kick_strength)

        if self.split:
            # Use separate writer for actual data
            filename = self.split_filename_fmt % (paw.niter, paw.action)
            writer = Writer(filename, paw.world, mode='w',
                            tag=self.__class__.ulmtag_split)
        else:
            # Use the same writer for actual data
            writer = self.writer
        w = writer.child('wave_functions')
        paw.wfs.write_wave_functions(w)
        paw.wfs.write_occupations(w)
        if self.split:
            writer.close()
        # Sync the main writer
        self.writer.sync()

    def __del__(self):
        self.writer.close()
        TDDFTObserver.__del__(self)
