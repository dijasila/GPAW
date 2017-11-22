import numpy as np

from gpaw.io import Reader
from gpaw.io import Writer

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
        if len(paw.wfs.kpt_u) != 1:
            raise NotImplementedError('K-points not tested!')

        folding = 'Gauss'
        width = 0.08
        frequencies = [1.0]
        self.world = paw.world
        self.wfs = paw.wfs
        kpt_u = self.wfs.kpt_u

        self.set_folding(folding, width)

        self.omega_w = np.array(frequencies, dtype=float) * eV_to_au
        Nw = len(self.omega_w)
        NM = paw.wfs.setups.nao
        Nu = len(kpt_u)

        self.rho0_uMM = [None] * Nu
        self.FReDrho_uwMM = [None] * Nu
        self.FImDrho_uwMM = [None] * Nu
        for u, kpt in enumerate(kpt_u):
            # TODO: does this allocate duplicates on each kpt rank?
            self.rho0_uMM[u] = np.empty((NM, NM), dtype=float)
            self.FReDrho_uwMM[u] = np.empty((Nw, NM, NM), dtype=complex)
            self.FImDrho_uwMM[u] = np.empty((Nw, NM, NM), dtype=complex)

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
        return paw.wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM)

    def _update(self, paw):
        if paw.action == 'init':
            self.time = paw.time
            if paw.niter == 0:
                for u, kpt in enumerate(self.wfs.kpt_u):
                    rho_MM = self._get_density_matrix(paw, kpt)
                    assert np.max(np.absolute(rho_MM.imag)) == 0.0
                    self.rho0_uMM[u][:] = rho_MM.real
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
            print('%2d: %2d/%2d: %s %s' % (u, self.world.rank, self.world.size,
                                           rho_MM.shape, rho_MM.dtype))

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
        self.write_u_array(writer, 'rho0_uMM')
        self.write_u_array(writer, 'FReDrho_uwMM')
        self.write_u_array(writer, 'FImDrho_uwMM')
        writer.close()

    def write_u_array(self, writer, name):
        wfs = self.wfs
        A_uX = getattr(self, name)
        shape = None
        dtype = None
        for A_X in A_uX:
            if dtype is None:
                dtype = A_X.dtype
            assert dtype == A_X.dtype
            if shape is None:
                shape = A_X.shape
            assert shape == A_X.shape
        writer.add_array(name,
                         (wfs.nspins, wfs.kd.nibzkpts) + shape, dtype=dtype)
        for s in range(wfs.nspins):
            for k in range(wfs.kd.nibzkpts):
                A_X = wfs.collect_auxiliary(A_uX, k, s,
                                            shape=shape, dtype=dtype)
                writer.fill(A_X)

    def read(self, filename):
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
