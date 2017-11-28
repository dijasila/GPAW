import numpy as np

from gpaw.io import Reader
from gpaw.io import Writer

from gpaw.lcaotddft.frequency import Frequency
from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import read_wuMM
from gpaw.lcaotddft.utilities import write_uMM
from gpaw.lcaotddft.utilities import write_wuMM
from gpaw.tddft.units import eV_to_au


class FrequencyDensityMatrix(TDDFTObserver):
    version = 1
    ulmtag = 'FDM'

    def __init__(self,
                 paw,
                 filename=None,
                 frequencies=None,
                 restart_filename=None,
                 interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.has_initialized = False
        self.filename = filename
        self.restart_filename = restart_filename
        self.world = paw.world
        self.wfs = paw.wfs
        self.log = paw.log
        self.using_blacs = self.wfs.ksl.using_blacs
        if self.using_blacs:
            ksl_comm = self.wfs.ksl.block_comm
            kd_comm = self.wfs.kd.comm
            assert self.world.size == ksl_comm.size * kd_comm.size

        assert self.world.rank == self.wfs.world.rank

        if filename is not None:
            self.read(filename)
            return

        self.time = paw.time
        if isinstance(frequencies, Frequency):
            frequencies = [frequencies]
        self.frequency_w = frequencies

    def initialize(self):
        if self.has_initialized:
            return

        if self.wfs.gd.pbc_c.any():
            self.rho0_dtype = complex
        else:
            self.rho0_dtype = float

        def zeros(dtype):
            ksl = self.wfs.ksl
            if self.using_blacs:
                return ksl.mmdescriptor.zeros(dtype=dtype)
            else:
                return np.zeros((ksl.mynao, ksl.nao), dtype=dtype)

        self.rho0_uMM = []
        for kpt in self.wfs.kpt_u:
            self.rho0_uMM.append(zeros(self.rho0_dtype))
        self.FReDrho_wuMM = []
        self.FImDrho_wuMM = []
        for freq in self.frequency_w:
            self.FReDrho_wuMM.append([])
            self.FImDrho_wuMM.append([])
            for kpt in self.wfs.kpt_u:
                self.FReDrho_wuMM[-1].append(zeros(complex))
                self.FImDrho_wuMM[-1].append(zeros(complex))
        self.has_initialized = True

    def _get_density_matrix(self, paw, kpt):
        wfs = paw.wfs
        if self.using_blacs:
            ksl = wfs.ksl
            rho_MM = ksl.calculate_blocked_density_matrix(kpt.f_n, kpt.C_nM)
        else:
            rho_MM = wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM)
            paw.wfs.bd.comm.sum(rho_MM, root=0)
            # TODO: should the sum over bands be moved to
            # OrbitalLayouts.calculate_density_matrix()
        return rho_MM

    def _update(self, paw):
        if paw.action == 'init':
            if self.time != paw.time:
                raise RuntimeError('Timestamp do not match with the calculator')
            self.initialize()
            if paw.niter == 0:
                for u, kpt in enumerate(self.wfs.kpt_u):
                    rho_MM = self._get_density_matrix(paw, kpt)
                    if self.rho0_dtype == float:
                        assert np.max(np.absolute(rho_MM.imag)) == 0.0
                        rho_MM = rho_MM.real
                    self.rho0_uMM[u][:] = rho_MM
            return

        if paw.action == 'kick':
            return

        assert paw.action == 'propagate'

        time_step = paw.time - self.time
        self.time = paw.time

        exp_w = []
        for w, freq in enumerate(self.frequency_w):
            # Complex exponential with envelope
            exp = (np.exp(1.0j * freq.frequency * self.time) *
                   freq.envelope(self.time) * time_step)
            exp_w.append(exp)

        for u, kpt in enumerate(self.wfs.kpt_u):
            rho_MM = self._get_density_matrix(paw, kpt)
            Drho_MM = rho_MM - self.rho0_uMM[u]
            for w, freq in enumerate(self.frequency_w):
                # Update Fourier transforms
                self.FReDrho_wuMM[w][u] += Drho_MM.real * exp_w[w]
                self.FImDrho_wuMM[w][u] += Drho_MM.imag * exp_w[w]

    def write_restart(self):
        if self.restart_filename is None:
            return
        self.write(self.restart_filename)

    def write(self, filename):
        self.log('%s: Writing to %s' % (self.__class__.__name__, filename))
        writer = Writer(filename, self.world, mode='w',
                        tag=self.__class__.ulmtag)
        writer.write(version=self.__class__.version)
        writer.write(time=self.time)
        writer.write(frequency_w=[f.todict() for f in self.frequency_w])
        wfs = self.wfs
        write_uMM(wfs, writer, 'rho0_skMM', self.rho0_uMM)
        wlist = range(len(self.frequency_w))
        write_wuMM(wfs, writer, 'FReDrho_wskMM', self.FReDrho_wuMM, wlist)
        write_wuMM(wfs, writer, 'FImDrho_wskMM', self.FImDrho_wuMM, wlist)
        writer.close()

    def read(self, filename):
        reader = Reader(filename)
        tag = reader.get_tag()
        if tag != self.__class__.ulmtag:
            raise RuntimeError('Unknown tag %s' % tag)
        version = reader.version
        if version != self.__class__.version:
            raise RuntimeError('Unknown version %s' % version)
        self.time = reader.time
        self.frequency_w = [Frequency(**f) for f in reader.frequency_w]
        wfs = self.wfs
        self.rho0_uMM = read_uMM(wfs, reader, 'rho0_skMM')
        self.rho0_dtype = self.rho0_uMM[0].dtype
        wlist = range(len(self.frequency_w))
        self.FReDrho_wuMM = read_wuMM(wfs, reader, 'FReDrho_wskMM', wlist)
        self.FImDrho_wuMM = read_wuMM(wfs, reader, 'FImDrho_wskMM', wlist)
        reader.close()
        self.has_initialized = True
