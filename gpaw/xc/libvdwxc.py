from __future__ import print_function
import numpy as np
from gpaw.xc.libxc import LibXC
from gpaw.xc.functional import XCFunctional
from gpaw.xc.gga import GGA
from gpaw.xc.vdw import VDWFunctional
from gpaw.utilities import compiled_with_libvdwxc
from gpaw.utilities.grid_redistribute import redistribute
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

        nx, ny, nz = gd.parsize_c

        self.gd1 = gd
        self.gd2 = gd.new_descriptor(parsize=(nx, ny * nz, 1))
        self.gd3 = self.gd2.new_descriptor(parsize=(nx * ny * nz, 1, 1))
        self._vdw_init(fft_comm, gd.get_size_of_global_array(), gd.cell_cv)
        self.timer.stop('initialize')

    # This one we write down just to "echo" the original interface
    def calculate(self, gd, n_sg, v_sg, e_g=None):
        if e_g is None:
            e_g = gd.zeros()

        if self.aggressive_distribute:
            print('YARRRRG')
            n1_sg = self.gd1.empty(len(n_sg))
            v1_sg = self.gd1.empty(len(v_sg))
            e1_g = self.gd1.empty() # TODO handle e_g properly
            self.gd1.distribute(n_sg, n1_sg)
            self.gd1.distribute(v_sg, v1_sg)
            self.gd1.distribute(e_g, e1_g)
        else:
            n1_sg = n_sg
            v1_sg = v_sg
            e1_g = e_g
        energy = GGA.calculate(self, self.gd1, n1_sg, v1_sg, e_g=e1_g)
        if self.aggressive_distribute:
            n_sg[:] = self.gd1.collect(n1_sg, broadcast=True)
            v_sg[:] = self.gd1.collect(v1_sg, broadcast=True)
            e_g[:] = self.gd1.collect(e1_g, broadcast=True)
            #energy = self.gd1.comm.sum(energy)
        return energy
    
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
        self.timer.start('calculate gga')
        assert self._vdw is not None
        self.n_sg = n_sg
        self.sigma_xg = sigma_xg
        n_sg[:] = np.abs(n_sg)
        sigma_xg[:] = np.abs(sigma_xg)
        assert len(n_sg) == 1
        assert len(sigma_xg) == 1
        GGA.calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)

        # TODO:  Only call the various redistribute functions when necessary.

        self.timer.start('redist')
        n1_g = n_sg[0]
        n2_g = redistribute(self.gd1, self.gd2, n1_g, 1, 2)
        n3_g = redistribute(self.gd2, self.gd3, n2_g, 0, 1)
        gd2, n4_g = accordion_redistribute(self.gd3, n3_g, axis=0)

        sigma1_g = sigma_xg[0]
        sigma2_g = redistribute(self.gd1, self.gd2, sigma1_g, 1, 2)
        sigma3_g = redistribute(self.gd2, self.gd3, sigma2_g, 0, 1)
        gd4_, sigma4_g = accordion_redistribute(self.gd3, sigma3_g, axis=0)
        
        self.timer.stop('redist')
        self.timer.start('libvdwxc')
        Ecnl, v4_g, dedsigma4_g = self._calculate(n4_g, sigma4_g)
        self.timer.stop('libvdwxc')
        self.timer.start('redist')
        
        _gd, v3_g = accordion_redistribute(self.gd3, v4_g, axis=0,
                                           operation='back')
        _gd, dedsigma3_g = accordion_redistribute(self.gd3, v4_g, axis=0,
                                                  operation='back')

        v2_g = redistribute(self.gd2, self.gd3, v3_g, 0, 1, operation='back')
        v1_g = redistribute(self.gd1, self.gd2, v2_g, 1, 2, operation='back')

        dedsigma2_g = redistribute(self.gd2, self.gd3, dedsigma3_g, 0, 1,
                                   operation='back')
        dedsigma1_g = redistribute(self.gd1, self.gd2, dedsigma2_g, 1, 2,
                                   operation='back')
        self.timer.stop('redist')
        Ecnl = self.gd1.comm.sum(Ecnl)


        self.Ecnl = Ecnl
        self.vnl_g = v1_g
        self.Ecnl = Ecnl
        print('E nonlocal', Ecnl)
        v_sg[0, :] += v1_g
        dedsigma_xg[0, :] += dedsigma1_g
        self.timer.stop('calculate gga')
        return Ecnl # XXX is not actually supposed to return anything
        # Energy should be added to e_g

    def __del__(self):
        if self._vdw is not None:
            _gpaw.libvdwxc_free(self._vdw)
