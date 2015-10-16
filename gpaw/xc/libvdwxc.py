from __future__ import print_function
import numpy as np
from gpaw.xc.libxc import LibXC
from gpaw.xc.gga import GGA
from gpaw.utilities import compiled_with_libvdwxc, compiled_with_fftw_mpi,\
    compiled_with_pfft
from gpaw.utilities.grid_redistribute import GridRedistributor
from gpaw.utilities.accordion_redistribute import accordion_redistribute
from gpaw.utilities.timing import nulltimer
import _gpaw


def check_grid_descriptor(gd):
    assert gd.parsize_c[1] == 1 and gd.parsize_c[2] == 1
    nxpts_p = gd.n_cp[0][1:] - gd.n_cp[0][:-1]
    nxpts0 = nxpts_p[0]
    for nxpts in nxpts_p[1:-1]:
        assert nxpts == nxpts0
    assert nxpts_p[-1] <= nxpts0


_VDW_NUMERICAL_CODES = {'vdW-DF': 1,
                        'vdW-DF2': 2,
                        'vdW-DF-CX': 3}


def get_auto_pfft_grid(size):
    nproc1 = size
    nproc2 = 1
    while nproc1 > nproc2 and nproc1 % 2 == 0:
        nproc1 /= 2
        nproc2 *= 2
    return nproc1, nproc2


class LibVDWXC(GGA, object):
    def __init__(self, gga_kernel, name, timer=nulltimer, parallel='auto',
                 pfft_grid=None):
        """Initialize LibVDWXC object (further initialization required).

        parallel can be 'auto', '', 'serial', 'mpi', or 'pfft'.

         * 'serial' uses FFTW and only works with serial decompositions.

         * 'mpi' uses FFTW-MPI with communicator of the grid
           descriptor, parallelizing along the x axis.

         * 'pfft' uses PFFT and works with any decomposition,
           parallelizing along two directions for best scalability.

         * 'auto' uses PFFT if available, else FFTW-MPI if available,
           else adhoc if applicable, else serial.

         pfft_grid is the 2D CPU grid used by PFFT and can be a tuple
         (nproc1, nproc2) that multiplies to total communicator size,
         or None.  It is an error to specify pfft_grid unless using
         PFFT.  If left unspecified, a hopefully reasonable automatic
         choice will be made.
         """
        object.__init__(self)
        GGA.__init__(self, gga_kernel)
        self._vdw = None
        self.fft_comm = None
        self.timer = timer
        self.vdwcoef = 1.
        self.vdw_functional_name = name

        if parallel == 'auto':
            if compiled_with_pfft():
                parallel = 'pfft'
            elif compiled_with_fftw_mpi():
                parallel = 'mpi'
            else:
                parallel = 'serial'
        self.parallel = parallel
        if parallel != 'pfft' and pfft_grid is not None:
            raise ValueError('pfft_grid specified but pfft not available')
        self.pfft_grid = pfft_grid

        if not compiled_with_libvdwxc():
            raise ImportError('libvdwxc not compiled into GPAW')

    @property
    def name(self):
        if self.parallel == 'serial':
            pardesc = self.parallel
        elif self.parallel == 'mpi':
            pardesc = 'FFTW-MPI with %d cores' % self.fft_comm.size
        else:
            assert self.parallel == 'pfft'
            if self.pfft_grid is None:
                pardesc = 'PFFT with unknown parallelization'
            else:
                pardesc = 'PFFT with %d x %d cores' % tuple(self.pfft_grid)
        return '%s [libvdwxc/%s]' % (self.vdw_functional_name, pardesc)

    @name.setter
    def name(self, value):
        # Somewhere in the class hierarchy, someone tries to set the name.
        pass

    def get_setup_name(self):
        return 'revPBE'

    def _libvdwxc_c_init(self, comm, N_c, cell_cv):
        """Initialize libvdwxc things in C."""
        self.timer.start('libvdwxc init')
        code = _VDW_NUMERICAL_CODES[self.vdw_functional_name]
        self._vdw = _gpaw.libvdwxc_create(code, tuple(N_c),
                                          tuple(cell_cv.ravel()))

        if self.parallel == 'pfft':
            if self.pfft_grid is None:
                self.pfft_grid = get_auto_pfft_grid(comm.size)
            nproc1, nproc2 = self.pfft_grid
            assert nproc1 * nproc2 == comm.size
            _gpaw.libvdwxc_init_pfft(self._vdw, comm.get_c_object(),
                                     nproc1, nproc2)
        elif self.parallel == 'mpi':
            _gpaw.libvdwxc_init_mpi(self._vdw, comm.get_c_object())
        else:
            assert comm.size == 1
            _gpaw.libvdwxc_init_serial(self._vdw)
        self.fft_comm = comm
        self.timer.stop('libvdwxc init')

    def initialize(self, density, hamiltonian, wfs, occupations):
        self.timer.start('initialize')
        GGA.initialize(self, density, hamiltonian, wfs, occupations)
        bigger_fft_comm = None
        if density.finegd.comm.size == 1 and wfs.world.size > 1:
            bigger_fft_comm = wfs.world
        self._initialize(density.finegd, fft_comm=bigger_fft_comm)
        self.timer.stop('initialize')

    def _initialize(self, gd, fft_comm=None):
        """This is the real initialize, without any complicated arguments.

        This will initialize parallelization things in Python and then
        the actual C backend libvdwxc.

        If gd is not domain-decomposed, then fft_comm may passed
        and arrays will be distributed on that for parallel FFTs."""
        # We want a periodic descriptor because it tends to have nicely
        # primey numbers of grid points whereas non-periodic grid descriptors
        # do not.  We don't truly pad anything though.
        pad_gd = gd.new_descriptor(pbc_c=(1, 1, 1))
        if fft_comm is not None:
            assert pad_gd.comm.size == 1
        self.aggressive_distribute = (fft_comm is not None)
        if self.aggressive_distribute:
            pad_gd = pad_gd.new_descriptor(comm=fft_comm,
                                           parsize=(fft_comm.size, 1, 1))
        else:
            fft_comm = pad_gd.comm

        self.pad_gd = pad_gd
        self.dist1 = GridRedistributor(pad_gd, 1, 2)
        self.dist2 = GridRedistributor(self.dist1.gd2, 0, 1)
        self._libvdwxc_c_init(fft_comm, pad_gd.get_size_of_global_array(),
                              pad_gd.cell_cv)

    def calculate(self, gd, n_sg, v_sg, e_g=None):
        """Calculate energy and potential.

        gd may be non-periodic.  To be distinguished from self.gd
        which is always periodic due to priminess of FFT dimensions.
        (To do: proper padded FFTs.)"""
        nspins = len(n_sg)
        if e_g is not None:
            # TODO: handle e_g properly
            raise NotImplementedError('Proper energy density e_g')

        npad_sg = gd.zero_pad(n_sg, global_array=False)

        if self.aggressive_distribute:
            dist_gd = self.dist1.gd
            ndist_sg = dist_gd.empty(nspins)
            dist_gd.distribute(npad_sg, ndist_sg)
        else:
            dist_gd = self.pad_gd
            ndist_sg = npad_sg
        npad_sg = None  # Possible garbage collection

        vdist_sg = dist_gd.zeros(nspins)
        edist_g = dist_gd.empty()

        # We have the minor issue that dist_gd now goes across the cell
        # when evaluating derivatives, so in non-periodic systems we will
        # get (very) small numerical errors until we have proper padding.
        GGA.calculate(self, dist_gd, ndist_sg, vdist_sg, e_g=edist_g)
        ndist_sg = None
        energy = dist_gd.integrate(edist_g)
        edist_g = None

        if self.aggressive_distribute:
            vpad_sg = dist_gd.collect(vdist_sg, broadcast=True)
        else:
            vpad_sg = vdist_sg

        # Now we have to unpad.  Of course the density is zero in the padded
        # region so the energy contribution is zero.  However we added all the
        # vdw energy in the corner of e_g because of our rebellious nature,
        # and now we have to account for our sins
        pad_c = (1 - gd.pbc_c) * (gd.get_processor_position_from_rank() == 0)
        v_sg += vpad_sg[:, pad_c[0]:, pad_c[1]:, pad_c[2]:]
        return energy

    def calculate_nonlocal(self, n_g, sigma_g):
        v_g = np.zeros_like(n_g)
        dedsigma_g = np.zeros_like(sigma_g)
        energy = _gpaw.libvdwxc_calculate(self._vdw, n_g, sigma_g,
                                          v_g, dedsigma_g)
        return energy, v_g, dedsigma_g

    def calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        assert self._vdw is not None
        self.timer.start('calculate gga')
        n_sg[:] = np.abs(n_sg)  # XXXX What to do about this?
        sigma_xg[:] = np.abs(sigma_xg)
        assert len(n_sg) == 1
        assert len(sigma_xg) == 1
        GGA.calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        # TODO:  Only call the various redistribute functions when necessary.

        def to_1d_block_distribution(array):
            array = self.dist1.forth(array)
            array = self.dist2.forth(array)
            array = accordion_redistribute(self.dist2.gd2, array, axis=0)
            return array

        def from_1d_block_distribution(array):
            array = accordion_redistribute(self.dist2.gd2, array, axis=0,
                                           operation='back')
            array = self.dist2.back(array)
            array = self.dist1.back(array)
            return array

        assert len(sigma_xg) == 1
        assert len(n_sg) == 1

        self.timer.start('redist')
        nblock_g = to_1d_block_distribution(n_sg[0])
        sigmablock_g = to_1d_block_distribution(sigma_xg[0])
        self.timer.stop('redist')

        self.timer.start('libvdwxc')
        Ecnl, vblock_g, dedsigmablock_g = self.calculate_nonlocal(nblock_g,
                                                                  sigmablock_g)
        self.timer.stop('libvdwxc')

        self.timer.start('redist')
        v_g = from_1d_block_distribution(vblock_g)
        dedsigma_g = from_1d_block_distribution(dedsigmablock_g)
        #dedsigma_g *= 0.
        self.timer.stop('redist')

        Ecnl = self.gd.comm.sum(Ecnl)
        if self.gd.comm.rank == 0:
            # XXXXXXXXXXXXXXXX ugly
            e_g[0, 0, 0] += Ecnl / self.gd.dv * self.vdwcoef
        self.Ecnl = Ecnl * self.gd.dv
        assert len(v_sg) == 1
        v_sg[0, :] += v_g * self.vdwcoef
        assert len(dedsigma_xg) == 1
        dedsigma_xg[0, :] += dedsigma_g * self.vdwcoef
        self.timer.stop('calculate gga')

    def estimate_memory(self, mem):
        size = self.dist2.gd2.bytecount()
        mem.subnode('thetas', 20 * size)
        mem.subnode('other', 5 * size)

    def __del__(self):
        if self._vdw is not None:
            _gpaw.libvdwxc_free(self._vdw)


class VDWDF(LibVDWXC):
    def __init__(self, timer=nulltimer):
        kernel = LibXC('GGA_X_PBE_R+LDA_C_PW')
        LibVDWXC.__init__(self, gga_kernel=kernel, name='vdW-DF',
                          timer=timer)


class VDWDF2(LibVDWXC):
    def __init__(self, timer=nulltimer):
        kernel = LibXC('GGA_X_RPW86+LDA_C_PW')
        LibVDWXC.__init__(self, gga_kernel=kernel, name='vdW-DF2',
                          timer=timer)


class VDWDFCX(LibVDWXC):
    def __init__(self, timer=nulltimer):
        kernel = CXGGAKernel()
        LibVDWXC.__init__(self, gga_kernel=kernel, name='vdW-DF-CX',
                          timer=timer)


class CXGGAKernel:
    def __init__(self, just_kidding=False):
        self.just_kidding = just_kidding
        self.type = 'GGA'
        self.lda_c = LibXC('LDA_C_PW')
        if self.just_kidding:
            self.name = 'rPW86_with_%s' % self.lda_c.name
        else:
            self.name = 'CX'

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        e_g[:] = 0.0
        dedsigma_xg[:] = 0.0

        self.lda_c.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)

        for arr in [n_sg, v_sg, sigma_xg, dedsigma_xg]:
            assert len(arr) == 1
        self._exchange(n_sg[0], sigma_xg[0], e_g, v_sg[0], dedsigma_xg[0])

    def _exchange(self, rho, grho, sx, v1x, v2x):
        """Calculate cx local exchange.

        Note that this *adds* to the energy density sx so that it can
        be called after LDA correlation part without ruining anything.
        Also it adds to v1x and v2x as is normal in GPAW."""
        tol = 1e-20
        rho[rho < tol] = tol
        grho[grho < tol] = tol
        alp = 0.021789
        beta = 1.15
        a = 1.851
        b = 17.33
        c = 0.163
        mu_LM = 0.09434
        s_prefactor = 6.18733545256027
        Ax = -0.738558766382022 # = -3./4. * (3./pi)**(1./3)
        four_thirds = 4. / 3.

        grad_rho = np.sqrt(grho)

        # eventually we need s to power 12.  Clip to avoid overflow
        # (We have individual tolerances on both rho and grho, but
        # they are not sufficient to guarantee this)
        s_1 = (grad_rho / (s_prefactor * rho**four_thirds)).clip(0.0, 1e20)
        s_2 = s_1 * s_1
        s_3 = s_2 * s_1
        s_4 = s_3 * s_1
        s_5 = s_4 * s_1
        s_6 = s_5 * s_1

        fs_rPW86 = (1.0 + a * s_2 + b * s_4 + c * s_6)**(1./15.)

        if self.just_kidding:
            fs = fs_rPW86
        else:
            fs = (1.0 + mu_LM * s_2) / (1.0 + alp * s_6) \
                + alp * s_6 / (beta + alp * s_6) * fs_rPW86

        # the energy density for the exchange.
        sx[:] += Ax * rho**four_thirds * fs

        df_rPW86_ds = (1. / (15. * fs_rPW86**14.0)) * \
            (2 * a * s_1 + 4 * b * s_3 + 6 * c * s_5)

        if self.just_kidding:
            df_ds = df_rPW86_ds # XXXXXXXXXXXXXXXXXXXX
        else:
            df_ds = 1. / (1. + alp * s_6)**2 \
                * (2.0 * mu_LM * s_1 * (1. + alp * s_6)
                   - 6.0 * alp * s_5 * (1. + mu_LM * s_2)) \
                + alp * s_6 / (beta + alp * s_6) * df_rPW86_ds \
                + 6.0 * alp * s_5 * fs_rPW86 / (beta + alp * s_6) \
                * (1. - alp * s_6 / (beta + alp * s_6))

        # de/dn.  This is the partial derivative of sx wrt. n, for s constant
        v1x[:] += Ax * four_thirds * (rho**(1. / 3.) * fs
                                      - grad_rho / (s_prefactor * rho) * df_ds)
        # de/d(nabla n).  The other partial derivative
        v2x[:] += 0.5 * Ax * df_ds / (s_prefactor * grad_rho)
        # (We may or may not understand what that grad_rho is doing here.)


def test_derivatives():
    gen = np.random.RandomState(1)
    shape = (1, 20, 20, 20)
    ngpts = np.product(shape)
    n_sg = gen.rand(*shape)
    sigma_xg = np.zeros(shape)
    sigma_xg[:] = gen.rand(*shape)

    qe_kernel = CXGGAKernel(just_kidding=True)
    libxc_kernel = LibXC('GGA_X_RPW86+LDA_C_PW')

    cx_kernel = CXGGAKernel(just_kidding=False)

    def check(kernel, n_sg, sigma_xg):
        e_g = np.zeros(shape[1:])
        dedn_sg = np.zeros(shape)
        dedsigma_xg = np.zeros(shape)
        kernel.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)
        return e_g, dedn_sg, dedsigma_xg

    def check_and_write(kernel):
        n1_sg = n_sg.copy()
        e_g, dedn_sg, dedsigma_xg = check(kernel, n_sg, sigma_xg)
        dedn = dedn_sg[0, 0, 0, 0]
        dedsigma = dedsigma_xg[0, 0, 0, 0]

        dn = 1e-6
        n1_sg = n_sg.copy()
        n1_sg[0, 0, 0, 0] -= dn / 2.
        e1_g, _, _ = check(kernel, n1_sg, sigma_xg)

        n1_sg[0, 0, 0, 0] += dn
        e2_g, _, _ = check(kernel, n1_sg, sigma_xg)

        dedn_fd = (e2_g[0, 0, 0] - e1_g[0, 0, 0]) / dn
        dedn_err = abs(dedn - dedn_fd)

        print('e', e_g.sum() / ngpts)
        print('dedn', dedn, 'fd', dedn_fd, 'err %e' % dedn_err)

        sigma1_xg = sigma_xg.copy()
        sigma1_xg[0, 0, 0, 0] -= dn / 2.
        e1s_g, _, _ = check(kernel, n_sg, sigma1_xg)

        sigma1_xg[0, 0, 0, 0] += dn
        e2s_g, _, _ = check(kernel, n_sg, sigma1_xg)

        dedsigma_fd = (e2s_g[0, 0, 0] - e1s_g[0, 0, 0]) / dn
        dedsigma_err = dedsigma - dedsigma_fd

        print('dedsigma', dedsigma, 'fd', dedsigma_fd, 'err %e' % dedsigma_err)
        return e_g, dedn_sg, dedsigma_xg

    print('pw86r libxc')
    e_lxc_g, dedn_lxc_g, dedsigma_lxc_g = check_and_write(libxc_kernel)
    print()
    print('pw86r ours')
    e_qe_g, dedn_qe_g, dedsigma_qe_g = check_and_write(qe_kernel)
    print()
    print('cx')
    check_and_write(cx_kernel)

    print()
    print('lxc vs qe discrepancies')
    print('=======================')
    e_err = np.abs(e_lxc_g - e_qe_g).max()
    print('e', e_err)
    dedn_err = np.abs(dedn_qe_g - dedn_lxc_g).max()
    dedsigma_err = np.abs(dedsigma_lxc_g - dedsigma_qe_g).max()
    print('dedn', dedn_err)
    print('dedsigma', dedsigma_err)


def test_selfconsistent():
    from gpaw import GPAW
    from ase.structure import molecule
    from gpaw.xc.gga import GGA

    system = molecule('H2O')
    system.center(vacuum=3.)

    def test(xc):
        calc = GPAW(mode='lcao',
                    xc=xc,
                    setups='sg15',
                    txt='gpaw.%s.txt' % str(xc)#.kernel.name
                    )
        system.set_calculator(calc)
        return system.get_potential_energy()

    libxc_results = {}

    for name in ['GGA_X_PBE_R+LDA_C_PW', 'GGA_X_RPW86+LDA_C_PW']:
        xc = GGA(LibXC(name))
        e = test(xc)
        libxc_results[name] = e


    cx_gga_results = {}
    cx_gga_results['rpw86'] = test(GGA(CXGGAKernel(just_kidding=True)))
    cx_gga_results['lv_rpw86'] = test(GGA(CXGGAKernel(just_kidding=False)))
    
    vdw_results = {}
    vdw_coef0_results = {}

    for vdw in [VDWDF(), VDWDF2(), VDWDFCX()]:
        vdw.vdwcoef = 0.0
        vdw_coef0_results[vdw.__class__.__name__] = test(vdw)
        vdw.vdwcoef = 1.0 # Leave nicest text file by running real calc last
        vdw_results[vdw.__class__.__name__] = test(vdw)
    
    from gpaw.mpi import world
    # These tests basically verify that the LDA/GGA parts of vdwdf
    # work correctly.
    if world.rank == 0:
        print('Now comparing...')
        err1 = cx_gga_results['rpw86'] - libxc_results['GGA_X_RPW86+LDA_C_PW']
        print('Our rpw86 must be identical to that of libxc. Err=%e' % err1)
        print('RPW86 interpolated with Langreth-Vosko stuff differs by %f'
              % (cx_gga_results['lv_rpw86'] - cx_gga_results['rpw86']))
        print('Each vdwdf with vdwcoef zero must yield same result as gga'
              'kernel')
        err_df1 = vdw_coef0_results['VDWDF'] - libxc_results['GGA_X_PBE_R+'
                                                             'LDA_C_PW']
        print('  df1 err=%e' % err_df1)
        err_df2 = vdw_coef0_results['VDWDF2'] - libxc_results['GGA_X_RPW86+'
                                                              'LDA_C_PW']
        print('  df2 err=%e' % err_df2)
        err_cx = vdw_coef0_results['VDWDFCX'] - cx_gga_results['lv_rpw86']
        print('   cx err=%e' % err_cx)


if __name__ == '__main__':
    test_derivatives()
    #test_selfconsistent()
