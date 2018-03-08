import ase.io.ulm as ulm

from gpaw.io import Writer

from gpaw.lcaotddft.observer import TDDFTObserver


class WaveFunctionReader(object):
    def __init__(self, filename, reader=None, wfreader=None):
        if reader is None:
            self.reader = ulm.Reader(filename)
            tag = self.reader.get_tag()
            if tag != WaveFunctionWriter.ulmtag:
                raise RuntimeError('Unknown tag %s' % tag)
            self.filename = filename
        else:
            self.reader = reader
            self.version = wfreader.version
            self.split = wfreader.split
            self.filename = wfreader.filename
        if self.split:
            name, ext = tuple(self.filename.rsplit('.', 1))
            self.split_filename_fmt = name + '-%06d-%s.' + ext

    def __getattr__(self, attr):
        try:
            return getattr(self.reader, attr)
        except KeyError:
            pass

        # Split reader handling
        if attr == 'wave_functions' and self.split:
            if not hasattr(self, 'splitreader'):
                filename = self.split_filename_fmt % (self.niter, self.action)
                self.splitreader = ulm.Reader(filename)
                tag = self.splitreader.get_tag()
                assert tag == WaveFunctionWriter.ulmtag_split
            return getattr(self.splitreader, attr)

        # Compatibility for older versions
        if attr == 'split':
            return False

        raise AttributeError('Attribute %s not defined in version %s' %
                             (repr(attr), repr(self.version)))

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        return WaveFunctionReader(None, self.reader[index], self)

    def close(self):
        if hasattr(self, 'splitreader'):
            self.splitreader.close()
            del self.splitreader
        self.reader.close()

    def __del__(self):
        if hasattr(self, 'splitreader'):
            self.splitreader.close()
            del self.splitreader


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
            reader = WaveFunctionReader(filename)
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
