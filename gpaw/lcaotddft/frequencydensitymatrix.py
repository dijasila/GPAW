import numpy as np

from gpaw.io import Reader
from gpaw.io import Writer

from gpaw.blacs import BlacsGrid
from gpaw.blacs import Redistributor
from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.tddft.units import eV_to_au


class FrequencyDensityMatrix(TDDFTObserver):
    version = 1
    ulmtag = 'FDM'

    def __init__(self, paw, interval=1,
                 frequencies=None,
                 folding=None,
                 width=None):
        TDDFTObserver.__init__(self, paw, interval)
        folding = 'Gauss'
        width = 0.08
        frequencies = [1.0]
        self.world = paw.world
        self.wfs = paw.wfs
        kpt_u = self.wfs.kpt_u
        ksl = self.wfs.ksl

        self.set_folding(folding, width)
        self.using_blacs = ksl.using_blacs
        if self.using_blacs:
            ksl_comm = ksl.block_comm
            kd_comm = self.wfs.kd.comm
            assert self.world.size == ksl_comm.size * kd_comm.size

        self.omega_w = np.array(frequencies, dtype=float) * eV_to_au

        def zeros(dtype):
            if self.using_blacs:
                return ksl.mmdescriptor.zeros(dtype=dtype)
            else:
                return np.zeros((ksl.mynao, ksl.nao), dtype=dtype)

        if self.wfs.gd.pbc_c.any():
            self.rho0_dtype = complex
        else:
            self.rho0_dtype = float

        self.rho0_uMM = []
        self.FReDrho_uwMM = []
        self.FImDrho_uwMM = []
        for u, kpt in enumerate(kpt_u):
            self.rho0_uMM.append(zeros(self.rho0_dtype))
            self.FReDrho_uwMM.append([])
            self.FImDrho_uwMM.append([])
            for w, omega in enumerate(self.omega_w):
                self.FReDrho_uwMM[-1].append(zeros(complex))
                self.FImDrho_uwMM[-1].append(zeros(complex))

    def set_folding(self, folding, width):
        if width is None:
            folding = None

        self.folding = folding
        if self.folding is None:
            self.width = None
        else:
            self.width = width * eV_to_au

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
            self.time = paw.time
            if paw.niter == 0:
                for u, kpt in enumerate(self.wfs.kpt_u):
                    rho_MM = self._get_density_matrix(paw, kpt)
                    if self.rho0_dtype == float:
                        assert np.max(np.absolute(rho_MM.imag)) == 0.0
                        rho_MM = rho_MM.real
                    # print '_get %s %s %s' % (self.ranks(), rho_MM.shape, rho_MM.dtype)
                    # print '_get %s %s %s\n%s' % (self.ranks(), rho_MM.shape, rho_MM.dtype, rho_MM)
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
        pass
        # self.write(self.restart_filename)

    def write(self, filename):
        writer = Writer(filename, self.world, mode='w',
                        tag=self.__class__.ulmtag)
        for arg in ['time', 'omega_w', 'folding', 'width']:
            writer.write(arg, getattr(self, arg))
        self.write_uwMM(writer, 'rho0_uMM', no_w=True)
        self.write_uwMM(writer, 'FReDrho_uwMM')
        self.write_uwMM(writer, 'FImDrho_uwMM')
        writer.close()

    def write_uwMM(self, writer, name, no_w=False):
        wfs = self.wfs
        NM = wfs.ksl.nao
        a_uwMM = getattr(self, name)
        dtype = a_uwMM[0][0].dtype
        if no_w:
            shape = (NM, NM)
            w_i = [None]
        else:
            shape = (len(self.omega_w), NM, NM)
            w_i = range(len(self.omega_w))

        writer.add_array(name,
                         (wfs.nspins, wfs.kd.nibzkpts) + shape, dtype=dtype)
        for s in range(wfs.nspins):
            for k in range(wfs.kd.nibzkpts):
                for w in w_i:
                    a_MM = self.collect(a_uwMM, s, k, w)
                    writer.fill(a_MM)

    def ranks(self):
        # TODO: this is a debug function
        import time
        time.sleep(self.world.rank * 0.1)
        wfs = self.wfs
        txt = ''
        comm_i = [self.world, wfs.world, wfs.gd.comm, wfs.kd.comm,
                  wfs.bd.comm, wfs.ksl.block_comm]
        for comm in comm_i:
            txt += '%2d/%2d ' % (comm.rank, comm.size)
        return txt

    def collect(self, a_uwMM, s, k, w=None):
        # This function is based on
        # gpaw/wavefunctions/base.py: WaveFunctions.collect_auxiliary()

        dtype = a_uwMM[0][0].dtype

        wfs = self.wfs
        ksl = wfs.ksl
        NM = ksl.nao
        kpt_rank, u = wfs.kd.get_rank_and_index(s, k)

        ksl_comm = ksl.block_comm

        if wfs.kd.comm.rank == kpt_rank:
            if w is None:
                a_MM = a_uwMM[u]
            else:
                a_MM = a_uwMM[u][w]

            # Collect within blacs grid
            if self.using_blacs:
                a_mm = a_MM
                grid = BlacsGrid(ksl_comm, 1, 1)
                MM_descriptor = grid.new_descriptor(NM, NM, NM, NM)
                mm2MM = Redistributor(ksl_comm,
                                      ksl.mmdescriptor,
                                      MM_descriptor)

                a_MM = MM_descriptor.empty(dtype=dtype)
                mm2MM.redistribute(a_mm, a_MM)

            # Domain master send a_MM to the global master
            if ksl_comm.rank == 0:
                if kpt_rank == 0:
                    assert self.world.rank == 0
                    return a_MM
                else:
                    wfs.kd.comm.send(a_MM, 0, 2017)
                    return None
        elif ksl_comm.rank == 0 and kpt_rank != 0:
            assert self.world.rank == 0
            a_MM = np.empty((NM, NM), dtype=dtype)
            wfs.kd.comm.receive(a_MM, kpt_rank, 2017)
            return a_MM

    def read(self, filename):
        raise NotImplementedError()
        reader = Reader(filename)
        for arg in ['time', 'omega_w', 'folding', 'width']:
            setattr(self, arg, getattr(reader, arg))
        self.read_u_array(reader, 'rho0_uMM')
        self.read_u_array(reader, 'FReDrho_uwMM')
        self.read_u_array(reader, 'FImDrho_uwMM')
        reader.close()

    def read_u_array(self, reader, name):
        wfs = self.wfs
        for u, kpt in enumerate(wfs.kpt_u):
            getattr(self, name)[u] = reader.proxy(name, kpt.s, kpt.k)[:]
