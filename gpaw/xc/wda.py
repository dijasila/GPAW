from gpaw.xc.functional import XCFunctional
from gpaw import mpi
import numpy as np


class WDA(XCFunctional):
    def __init__(self):
        XCFunctional.__init__(self, 'WDA', 'LDA')
        self.num_nbar = 100

    def initialize(self, density, hamiltonia, wfs, occupations):
        self.wfs = wfs
        self.density = density
    
    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        from gpaw.xc.wda.splines import build_splines

        self.gd = gd.new_descriptor(comm=mpi.serial_comm)
        nb_i = self.get_nbars(n_sg, self.num_nbar)
        my_i = self.distribute_nbars(self.num_nbar, mpi.rank, mpi.size)

        Gs_i, Grs_i = build_splines(nb_i[my_i], self.gd)

        K_K = self.get_K_K(self.gd)
        Gs_ik, Grs_ik = self.get_G_ks(Gs_i, Grs_i, K_K)

        Z_isg = self.calc_Z_isg(n_sg, Gs_ik)

        alpha_isg = self.get_alphas(Z_isg)

        e1_g = self.wda_energy(n_sg, alpha_isg, Grs_ik)

        v1_sg = self.V(n_sg, alpha_isg, Z_isg, Grs_ik, Gs_ik)

        mpi.world.sum(e1_g)
        mpi.world.sum(v1_sg)

        gd.distribute(e1_g, e_g)
        gd.distribute(v1_sg, v_sg)

    def get_nbars(self, n_g, npts=100):
        min_dens = np.min(n_g)
        max_dens = np.max(n_g)
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

    def get_K_K(self, gd):
        from gpaw.utilities.tools import construct_reciprocal
        from ase.units import Bohr
        K2_K, _ = construct_reciprocal(gd)
        K2_K[0, 0, 0] = 0
        return K2_K**(1 / 2)

    def get_G_ks(self, Gs_i, Grs_i, k_k):
        Gs_ik = np.array([Gs(k_k) for Gs in Gs_i])
        Grs_ik = np.array([Gs(k_k) for Gs in Gs_i])

        return Gs_ik, Grs_ik

    def calc_Z_isg(self, n_sg, Gs_iK):
        from convolutions import npp_conv
        
        n_sK = self.fftn(n_sg)
        na = np.newaxis
        Gs_iK = Gs_iK[:, na, :, :, :]
        n_sK = n_sK[na, :, :, :, :]

        Z_isg = self.ifftn(self.npp_conv(Gs_iK, n_sK))
    
        return Z_isg    
        
    def npp_conv(self, np_xK, p_xK):
        assert np_xK.shape[-3:] == p_xK.shape[-3:]
        return np_xK * p_xK


    def pp_conv(self, p1_xK, p2_xK, volume):
        assert p1_xK.shape[-3:] == p2_xK.shape[-3:]
    
        return volume * p1_xK * p2_xK

    def fftn(self, n_g):
        N = np.prod(n_g.shape[-3:])

        return np.fft.fftn(n_g, axes=(-3, -2, -1))  / N

    def ifftn(self, n_g):
        N = np.prod(n_g.shape[-3:])

        res = np.fft.ifftn(n_g, axes=(-3, -2, -1)) * N
        assert np.allclose(res, res.real)

        return res.real

    def get_alphas(self, Z_ig):
        Z_ri = Z_ig.reshape(len(Z_ig), -1).T

        alpha_ri = np.zeros_like(Z_ri)
        for ir, Z_i in enumerate(Z_ri):
            for ii, Z in enumerate(Z_i[:-1]):
                did_cross = (Z_i[ii] <= -1 and Z_i[ii + 1] > -1) \
                            or (Z_i[ii] > -1 and Z_i[ii + 1] <= -1)
                if did_cross:
                    alpha_ri[ir, ii] = (Z_i[ii + 1] - (-1)) / (Z_i[ii + 1] - Z_i[ii])
                    alpha_ri[ir, ii + 1] = ((-1) - Z_i[ii]) / (Z_i[ii + 1] - Z_i[ii])
                    break


        res = alpha_ri.T.reshape(Z_ig.shape)    
        assert np.allclose(res.sum(axis=0), 1)
        return res

    def wda_energy(self, n_g, alpha_ig, Grs_ik):
        # Convolve n with G/r
        # Multiply result by alpha * n
        # Integrate
        from convolutions import npp_conv
        from ffts import fftn, ifftn
    
        n_k = fftn(n_g)
        res_ig = ifftn(npp_conv(Grs_ik, n_k))
        assert res_ig.shape == alpha_ig.shape

        result_g = np.sum(res_ig * n_g[np.newaxis, :] * alpha_ig, axis=0)

        return result_g

    def V1(self, n_g, alpha_ig, Grs_ik):
        n_k = self.fftn(n_g)
    
        res_ig = self.ifftn(self.npp_conv(Grs_ik, n_k))
        assert res_ig.shape == alpha_ig.shape

        result_g = np.sum(res_ig * alpha_ig, axis=0)
        return result_g


    def V1p(self, n_g, alpha_ig, Grs_ik):
        f_k = self.fftn(n_g[np.newaxis, :] * alpha_ig)
        res_ig = self.ifftn(self.npp_conv(Grs_ik, f_k))
        assert res_ig.shape == alpha_ig.shape

        return res_ig.sum(axis=0)


    def V2(self, n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik):
        dZ_ig = np.roll(Z_ig, -1, axis=0) - Z_ig
        n_k = self.fftn(n_g)
        Gnconv_ig = self.fftn(self.npp_conv(Grs_ik, n_k))
        res = self.ifftn(self.npp_conv(np.roll(Gs_ik, -1, axis=0), self.fftn(Gnconv_ig * n_g[np.newaxis, :] / dZ_ig)))
        
        res2 = self.ifftn(self.npp_conv(np.roll(Gs_ik, -1, axis=0) - Gs_ik, self.fftn(Gnconv_ig * n_g[np.newaxis, :] * alpha_ig / dZ_ig)))
        
        result = ((res - res2) * (alpha_ig != 0)).sum(axis=0)
    
        return result


    def V(self, n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik):
        return (V1(n_g, alpha_ig, Grs_ik) 
                + V1p(n_g, alpha_ig, Grs_ik)
                + V2(n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik))
