from __future__ import print_function
import numpy as np
from gpaw.xc.libxc import LibXC
from gpaw.xc.functional import XCFunctional
from gpaw.xc.gga import GGA
from gpaw.xc.vdw import VDWFunctional
from gpaw.utilities import compiled_with_libvdwxc
from gpaw.utilities.grid_redistribute import GridRedistributor
from gpaw.utilities.accordion_redistribute import accordion_redistribute
from gpaw.utilities.timing import nulltimer
from gpaw.mpi import SerialCommunicator
import _gpaw

def check_grid_descriptor(gd):
    assert gd.parsize_c[1] == 1 and gd.parsize_c[2] == 1
    nxpts_p = gd.n_cp[0][1:] - gd.n_cp[0][:-1]
    nxpts0 = nxpts_p[0]
    for nxpts in nxpts_p[1:-1]:
        assert nxpts == nxpts0
    assert nxpts_p[-1] <= nxpts0


class LibVDWXC(GGA, object):
    def __init__(self, timer=nulltimer):
        object.__init__(self)
        GGA.__init__(self, LibXC('GGA_X_PBE_R+LDA_C_PW'))
        self._vdw = None
        self._fft_comm = None
        self.timer = timer
        if not compiled_with_libvdwxc():
            raise ImportError('libvdwxc not compiled into GPAW')

    @property
    def name(self):
        if self._fft_comm is None:
            desc = 'serial'
        else:
            if self._fft_comm.size == 1:
                # Invokes MPI libraries and is implementation-wise
                # slightly different from the serial version
                desc = 'in "parallel" with 1 core'
            else:
                desc = ('in parallel with %d cores'
                        % self._fft_comm.size)
        return 'vdW-DF [libvdwxc %s]' % desc

    @name.setter
    def name(self, value):
        # Somewhere in the class hierarchy, someone tries to set the name.
        pass

    def get_setup_name(self):
        return 'revPBE'

    def _vdw_init(self, comm, N_c, cell_cv):
        self.timer.start('lib init')
        manyargs = list(N_c) + list(cell_cv.ravel())
        try:
            comm.get_c_object()
        except NotImplementedError:  # Serial
            self._vdw = _gpaw.libvdwxc_initialize(None, *manyargs)
        else:
            try:
                initfunc = _gpaw.libvdwxc_initialize_mpi
            except AttributeError:
                raise ImportError('parallel libvdwxc not compiled into GPAW')
            self._vdw = initfunc(comm.get_c_object(), *manyargs)
            self._fft_comm = comm
        self.timer.stop('lib init')

    def initialize(self, density, hamiltonian, wfs, occupations):
        self.timer.start('initialize')
        GGA.initialize(self, density, hamiltonian, wfs, occupations)
        bigger_fft_comm = None
        if density.finegd.comm.size == 1 and wfs.world.size > 1:
            bigger_fft_comm = wfs.world
        self._initialize(density.finegd, fft_comm=bigger_fft_comm)

    def _initialize(self, gd, fft_comm=None):
        """This is the real initialize, without any complicated arguments."""
        self.aggressive_distribute = (fft_comm is not None)
        if self.aggressive_distribute:
            gd = gd.new_descriptor(comm=fft_comm,
                                   parsize=(fft_comm.size, 1, 1))
        else:
            fft_comm = gd.comm

        self.dist1 = GridRedistributor(gd, 1, 2)
        self.dist2 = GridRedistributor(self.dist1.gd2, 0, 1)
        self._vdw_init(fft_comm, gd.get_size_of_global_array(), gd.cell_cv)
        self.timer.stop('initialize')

    # This one we write down just to "echo" the original interface
    def calculate(self, gd, n_sg, v_sg, e_g=None):
        if e_g is None:
            e_g = gd.zeros()

        dist_gd = gd
        if self.aggressive_distribute:
            print('distribute aggressively')
            dist_gd = self.dist1.gd
            n1_sg = dist_gd.empty(len(n_sg))
            v1_sg = dist_gd.empty(len(v_sg))
            e1_g = dist_gd.empty() # TODO handle e_g properly
            dist_gd.distribute(n_sg, n1_sg)
            dist_gd.distribute(v_sg, v1_sg)
            dist_gd.distribute(e_g, e1_g)
        else:
            n1_sg = n_sg
            v1_sg = v_sg
            e1_g = e_g
        energy = GGA.calculate(self, dist_gd, n1_sg, v1_sg, e_g=e1_g)

        if self.aggressive_distribute:
            n_sg[:] = self.gd1.collect(n1_sg, broadcast=True)
            v_sg[:] = self.gd1.collect(v1_sg, broadcast=True)
            e_g[:] = self.gd1.collect(e1_g, broadcast=True)
        return gd.integrate(e_g)
    
    def _calculate(self, n_g, sigma_g):
        v_g = np.zeros_like(n_g)
        dedsigma_g = np.zeros_like(sigma_g)
        if self._fft_comm is None:
            energy = _gpaw.libvdwxc_calculate(self._vdw, n_g, sigma_g,
                                              v_g, dedsigma_g) 
        else:
           energy = _gpaw.libvdwxc_calculate_mpi(self._vdw, n_g, sigma_g,
                                                  v_g, dedsigma_g)
        return energy, v_g, dedsigma_g

    def calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        assert self._vdw is not None
        self.timer.start('calculate gga')
        self.n_sg = n_sg
        self.sigma_xg = sigma_xg
        n_sg[:] = np.abs(n_sg)
        sigma_xg[:] = np.abs(sigma_xg)
        assert len(n_sg) == 1
        assert len(sigma_xg) == 1
        GGA.calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        # TODO:  Only call the various redistribute functions when necessary.
        self.timer.start('redist')

        def to_1d_block_distribution(array):
            array = self.dist1.forth(array)
            array = self.dist2.forth(array)
            _gd, array = accordion_redistribute(self.dist2.gd2, array, axis=0)
            return array

        def from_1d_block_distribution(array):
            _gd, array = accordion_redistribute(self.dist2.gd2, array, axis=0,
                                                operation='back')
            array = self.dist2.back(array)
            array = self.dist1.back(array)
            return array

        assert len(sigma_xg) == 1
        assert len(n_sg) == 1
        nblock_g = to_1d_block_distribution(n_sg[0])
        sigmablock_g = to_1d_block_distribution(sigma_xg[0])

        self.timer.stop('redist')
        self.timer.start('libvdwxc')
        Ecnl, vblock_g, dedsigmablock_g = self._calculate(nblock_g,
                                                          sigmablock_g)
        self.timer.stop('libvdwxc')
        self.timer.start('redist')
        v_g = from_1d_block_distribution(vblock_g)
        dedsigma_g = from_1d_block_distribution(dedsigmablock_g)
        self.timer.stop('redist')

        Ecnl = self.gd.comm.sum(Ecnl)
        if self.gd.comm.rank == 0:
            e_g[0, 0, 0] += Ecnl / self.gd.dv # XXXXXXXXXXXXXXXX ugly
        self.Ecnl = Ecnl
        assert len(v_sg) == 1
        v_sg[0, :] += v_g
        assert len(dedsigma_xg) == 1
        dedsigma_xg[0, :] += dedsigma_g
        self.timer.stop('calculate gga')

    def __del__(self):
        if self._vdw is not None:
            _gpaw.libvdwxc_free(self._vdw)
