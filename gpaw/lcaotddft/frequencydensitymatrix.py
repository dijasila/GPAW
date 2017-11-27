import numpy as np

from gpaw.io import Reader
from gpaw.io import Writer

from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import read_uwMM
from gpaw.lcaotddft.utilities import write_uMM
from gpaw.lcaotddft.utilities import write_uwMM
from gpaw.tddft.units import eV_to_au


class FrequencyDensityMatrix(TDDFTObserver):
    version = 1
    ulmtag = 'FDM'

    def __init__(self,
                 paw,
                 filename=None,
                 frequencies=None,
                 folding=None,
                 width=None,
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

        folding = 'Gauss'
        width = 0.08
        frequencies = [1.0]
        self.time = paw.time
        self.set_folding(folding, width * eV_to_au)

        self.omega_w = np.array(frequencies, dtype=float) * eV_to_au

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
        self.FReDrho_uwMM = []
        self.FImDrho_uwMM = []
        for kpt in self.wfs.kpt_u:
            self.rho0_uMM.append(zeros(self.rho0_dtype))
            self.FReDrho_uwMM.append([])
            self.FImDrho_uwMM.append([])
            for omega in self.omega_w:
                self.FReDrho_uwMM[-1].append(zeros(complex))
                self.FImDrho_uwMM[-1].append(zeros(complex))
        self.has_initialized = True

    def set_folding(self, folding, width):
        if width is None:
            folding = None

        self.folding = folding
        if self.folding is None:
            self.width = None
        else:
            self.width = width

        if self.folding is None:
            self.envelope = lambda t: 1.0
        elif self.folding == 'Gauss':
            self.envelope = lambda t: np.exp(- 0.5 * self.width**2 * t**2)
        elif self.folding == 'Lorentz':
            self.envelope = lambda t: np.exp(- self.width * t)
        else:
            raise RuntimeError('Unknown folding: %s' % self.folding)

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

        # Complex exponential with envelope
        f_w = (np.exp(1.0j * self.omega_w * self.time) *
               self.envelope(self.time) * time_step)

        for u, kpt in enumerate(self.wfs.kpt_u):
            rho_MM = self._get_density_matrix(paw, kpt)
            Drho_MM = rho_MM - self.rho0_uMM[u]
            # Update Fourier transforms
            for w, omega in enumerate(self.omega_w):
                self.FReDrho_uwMM[u][w] += Drho_MM.real * f_w[w]
                self.FImDrho_uwMM[u][w] += Drho_MM.imag * f_w[w]

    def write_restart(self):
        if self.restart_filename is None:
            return
        self.write(self.restart_filename)

    def write(self, filename):
        self.log('%s: Writing to %s' % (self.__class__.__name__, filename))
        writer = Writer(filename, self.world, mode='w',
                        tag=self.__class__.ulmtag)
        writer.write(version=self.__class__.version)
        for arg in ['time', 'omega_w', 'folding', 'width']:
            writer.write(arg, getattr(self, arg))
        wfs = self.wfs
        write_uMM(wfs, writer, 'rho0_uMM', self.rho0_uMM)
        wlist = range(len(self.omega_w))
        write_uwMM(wfs, writer, 'FReDrho_uwMM', self.FReDrho_uwMM, wlist)
        write_uwMM(wfs, writer, 'FImDrho_uwMM', self.FImDrho_uwMM, wlist)
        writer.close()

    def read(self, filename):
        reader = Reader(filename)
        tag = reader.get_tag()
        if tag != self.__class__.ulmtag:
            raise RuntimeError('Unknown tag %s' % tag)
        version = reader.version
        if version != self.__class__.version:
            raise RuntimeError('Unknown version %s' % version)
        for arg in ['time', 'omega_w', 'folding', 'width']:
            setattr(self, arg, getattr(reader, arg))
        self.set_folding(self.folding, self.width)
        wfs = self.wfs
        self.rho0_uMM = read_uMM(wfs, reader, 'rho0_uMM')
        self.rho0_dtype = self.rho0_uMM[0].dtype
        wlist = range(len(self.omega_w))
        self.FReDrho_uwMM = read_uwMM(wfs, reader, 'FReDrho_uwMM', wlist)
        self.FImDrho_uwMM = read_uwMM(wfs, reader, 'FImDrho_uwMM', wlist)
        reader.close()
        self.has_initialized = True
