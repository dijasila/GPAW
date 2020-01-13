from gpaw.xc.functional import XCFunctional
from gpaw import mpi
import numpy as np


def timer(f):
    from functools import wraps
    from time import time
    @wraps(f)
    def wrapped(*args, **kwargs):
        t1 = time()
        res = f(*args, **kwargs)
        t2 = time()
        if mpi.rank == 0:
            print("Running {} took {} seconds".format(f.__name__, t2 - t1))
        return res
    return wrapped


class WDA(XCFunctional):
    def __init__(self):
        XCFunctional.__init__(self, 'WDA', 'LDA')
        self.num_nbar = 100
        # Params:
        # - num nbar
        # - min of nbar grid
        # - ??

    def initialize(self, density, hamiltonia, wfs, occupations):
        self.wfs = wfs
        self.density = density
    
    @timer
    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        from gpaw.xc.WDAUtils import build_splines, get_K_K

        wn_sg = gd.collect(n_sg, broadcast=True)
        wn_sg[wn_sg < 1e-20] = 1e-20

        self.gd = gd.new_descriptor(comm=mpi.serial_comm)
        nb_i = self.get_nbars(wn_sg, self.num_nbar)
        my_i = self.distribute_nbars(self.num_nbar, mpi.rank, mpi.size)

        Gs_i, Grs_i = build_splines(nb_i[my_i], self.gd)

        K_K = get_K_K(self.gd)
        Gs_ik, Grs_ik = self.get_G_ks(Gs_i, Grs_i, K_K)

        Z_isg = self.calc_Z_isg(wn_sg, Gs_ik)

        alpha_isg = self.get_alphas(Z_isg)

        e1_g = self.wda_energy(wn_sg, alpha_isg, Grs_ik)

        v1_sg = self.V(wn_sg, alpha_isg, Z_isg, Grs_ik, Gs_ik)

        mpi.world.sum(e1_g)
        mpi.world.sum(v1_sg)

        gd.distribute(e1_g, e_g)
        gd.distribute(v1_sg, v_sg)

    @timer
    def get_nbars(self, n_g, npts=100):
        min_dens = np.min(n_g)
        max_dens = np.max(n_g)
        min_dens = max(min_dens, 1e-6)
        nb_i = np.linspace(0.9 * min_dens, 1.1 * max_dens, npts)
        return nb_i

    def distribute_nbars(self, num_nbar, rank, size):
        n = num_nbar // size
        n0 = n + (num_nbar - n * size)
        assert (n * (size - 1) + n0 == num_nbar)

        if rank == 0:
            start = 0
            end = n0
        else:
            start = n0 + (rank - 1) * n
            end = start + n

        return range(start, end)

    @timer
    def get_G_ks(self, Gs_i, Grs_i, k_k):
        Gs_ik = np.array([Gs(k_k) for Gs in Gs_i])
        Grs_ik = np.array([Gs(k_k) for Gs in Gs_i])

        return Gs_ik, Grs_ik

    @timer
    def calc_Z_isg(self, n_sg, Gs_iK):
        assert len(n_sg.shape) == 4
        n_sK = self.fftn(n_sg)
        na = np.newaxis
        Gs_iK = Gs_iK[:, na, :, :, :]
        n_sK = n_sK[na, :, :, :, :]

        Z_isg = self.ifftn(self.npp_conv(Gs_iK, n_sK))
    
        return Z_isg
        
    def npp_conv(self, np_xK, p_xK):
        b = np_xK.shape[-3:] == p_xK.shape[-3:]
        assert b, '{} - {} - {}'.format(np_xK.shape, p_xK.shape, self.gd.n_c)
        return np_xK * p_xK

    def pp_conv(self, p1_xK, p2_xK, volume):
        assert p1_xK.shape[-3:] == p2_xK.shape[-3:]
    
        return volume * p1_xK * p2_xK

    def fftn(self, n_g):
        N = np.prod(n_g.shape[-3:])

        return np.fft.fftn(n_g, axes=(-3, -2, -1)) / N

    def ifftn(self, n_g):
        N = np.prod(n_g.shape[-3:])

        res = np.fft.ifftn(n_g, axes=(-3, -2, -1)) * N
        assert np.allclose(res, res.real)

        return res.real

    @timer
    def get_alphas(self, Z_isg):
        # Assumptions: Z_isg is monotonically increasing
        # at all s, g
        # Z_isg crosses -1 at all s, g

        fg_g = np.argmax(Z_isg > -1, axis=0)
        ll_g = fg_g - 1

        lengths = Z_isg.shape[-4:]
        arrs = [range(x) for x in lengths]

        S, X, Y, Z = np.meshgrid(*arrs, indexing='ij')

        alpha_isg = np.zeros_like(Z_isg)
        alpha_isg[ll_g, S, X, Y, Z] = ((Z_isg[fg_g, S, X, Y, Z] - (-1))
                                       / (Z_isg[fg_g, S, X, Y, Z]
                                          - Z_isg[ll_g, S, X, Y, Z]))
        alpha_isg[fg_g, S, X, Y, Z] = (((-1) - Z_isg[ll_g, S, X, Y, Z])
                                       / (Z_isg[fg_g, S, X, Y, Z]
                                          - Z_isg[ll_g, S, X, Y, Z]))

        return alpha_isg

    @timer
    def wda_energy(self, n_g, alpha_ig, Grs_ik):
        # Convolve n with G/r
        # Multiply result by alpha * n
        # Integrate
        n_k = self.fftn(n_g)
        na = np.newaxis
        res_ig = self.ifftn(self.npp_conv(Grs_ik[:, na, :, :, :], n_k))
        b = res_ig.shape == alpha_ig.shape
        assert b, "{}-{}-{}-{}".format(res_ig.shape,
                                       alpha_ig.shape, n_g.shape, Grs_ik.shape)

        result_g = np.sum(res_ig * n_g[np.newaxis, :] * alpha_ig, axis=0)

        return result_g

    @timer
    def V1(self, n_g, alpha_ig, Grs_ik):
        n_k = self.fftn(n_g)
    
        res_ig = self.ifftn(self.npp_conv(Grs_ik, n_k))
        b = res_ig.shape == alpha_ig.shape
        assert b, "{} - {}".format(res_ig.shape, alpha_ig.shape)

        result_g = np.sum(res_ig * alpha_ig, axis=0)
        return result_g

    @timer
    def V1p(self, n_g, alpha_ig, Grs_ik):
        f_k = self.fftn(n_g[np.newaxis, :] * alpha_ig)
        res_ig = self.ifftn(self.npp_conv(Grs_ik, f_k))
        assert res_ig.shape == alpha_ig.shape

        return res_ig.sum(axis=0)

    @timer
    def V2(self, n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik):
        dZ_ig = np.roll(Z_ig, -1, axis=0) - Z_ig
        n_k = self.fftn(n_g)
        Gnconv_ig = self.fftn(self.npp_conv(Grs_ik, n_k))
        res = self.ifftn(
            self.npp_conv(
                np.roll(Gs_ik, -1, axis=0),
                self.fftn(Gnconv_ig * n_g[np.newaxis, :] / dZ_ig)))
        
        res2 = self.ifftn(
            self.npp_conv(
                np.roll(Gs_ik, -1, axis=0) - Gs_ik,
                self.fftn(Gnconv_ig * n_g[np.newaxis, :] * alpha_ig / dZ_ig)))
        
        result = ((res - res2) * (alpha_ig != 0)).sum(axis=0)
    
        return result

    @timer
    def V(self, n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik):
        Gs_ik = Gs_ik[:, np.newaxis, :, :, :]
        Grs_ik = Grs_ik[:, np.newaxis, :, :, :]
        return (self.V1(n_g, alpha_ig, Grs_ik)
                + self.V1p(n_g, alpha_ig, Grs_ik)
                + self.V2(n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik))

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        return 0
