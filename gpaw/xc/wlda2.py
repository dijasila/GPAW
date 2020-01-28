from gpaw.xc.functional import XCFunctional
import numpy as np
from gpaw.utilities.tools import construct_reciprocal
import gpaw.mpi as mpi


class WLDA(XCFunctional):
    def __init__(self, kernel=None, mode="", filter_kernel=""):
        XCFunctional.__init__(self, 'WLDA', 'LDA')

        self.nindicators = int(5 * 1e2)

        if kernel is not None:
            self.get_weight_function = kernel
        
    def initialize(self, density, hamiltonian, wfs, occupations):
        self.density = density
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.occupations = occupations

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        # 0. Collect density,
        # and get grid_descriptor appropriate for collected density
        wn_sg = gd.collect(n_sg, broadcast=True)
        gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        self.gd = gd1

        alphas = self.setup_indicator_grid(self.nindicators, wn_sg)
        self.alphas = alphas
        self.setup_indicators(alphas)
        my_alpha_indices = self.distribute_alphas(self.nindicators,
                                                  mpi.rank, mpi.size)
        
        # 1. Correct density
        # This or correct via self.get_ae_density(gd, n_sg)
        wn_sg = wn_sg
        wn_sg[wn_sg <  1e-20] = 1e-20
        # 2. calculate weighted density
        # This contains contributions for the alphas at this
        # rank, i.e. we need a world.sum to get all contributions
        nstar_sg = self.alt_weight(wn_sg, my_alpha_indices, gd1)
        mpi.world.sum(nstar_sg)

        # 3. Calculate LDA energy
        e1_g, v1_sg = self.calculate_wlda(wn_sg, nstar_sg, my_alpha_indices)

        gd.distribute(e1_g, e_g)
        gd.distribute(v1_sg, v_sg)
        
        # Done
        
    def setup_indicator_grid(self, nindicators, n_sg):
        md = np.min(n_sg)
        md = max(md, 1e-6)
        mad = np.max(n_sg)
        mad = max(mad, 1e-6)
        # return np.exp(np.linspace(np.log(md * 0.9), np.log(mad * 1.1), nindicators))
        return np.linspace(md * 0.9, mad * 1.1, nindicators)
        

    def setup_indicators(self, alphas):
        
        def get_ind_alpha(ia):
            # Returns a function that is 1 at alphas[ia]
            # and goes smoothly to zero at adjacent points
            if ia > 0 and ia < len(alphas) - 1:
                def ind(x):
                    copy_x = np.zeros_like(x)
                    ind1 = np.logical_and(x < alphas[ia], x >= alphas[ia - 1])
                    copy_x[ind1] = ((x[ind1] - alphas[ia - 1])
                                    / (alphas[ia] - alphas[ia - 1]))

                    ind2 = np.logical_and(x >= alphas[ia], x < alphas[ia + 1])
                    copy_x[ind2] = ((alphas[ia + 1] - x[ind2])
                                    / (alphas[ia + 1] - alphas[ia]))

                    return copy_x
                    
            elif ia == 0:
                def ind(x):
                    copy_x = np.zeros_like(x)
                    ind1 = (x <= alphas[ia])
                    copy_x[ind1] = 1
                        
                    ind2 = np.logical_and((x <= alphas[ia + 1]),
                                          np.logical_not(ind1))
                    copy_x[ind2] = ((alphas[ia + 1] - x[ind2])
                                    / (alphas[ia + 1] - alphas[ia]))

                    return copy_x
                    
            elif ia == len(alphas) - 1:
                def ind(x):
                    copy_x = np.zeros_like(x)
                    ind1 = (x >= alphas[ia])
                    copy_x[ind1] = 1
                        
                    ind2 = np.logical_and((x >= alphas[ia - 1]),
                                          np.logical_not(ind1))
                    copy_x[ind2] = ((x[ind2] - alphas[ia - 1])
                                    / (alphas[ia] - alphas[ia - 1]))

                    return copy_x

            else:
                raise ValueError("Asked for index: {} in grid of length: {}"
                                 .format(ia, len(alphas)))
            return ind
        self.get_indicator_alpha = get_ind_alpha

        def get_ind_sg(wn_sg, ia):
            ind_a = self.get_indicator_alpha(ia)
            ind_sg = ind_a(wn_sg).astype(wn_sg.dtype)

            return ind_sg

        self.get_indicator_sg = get_ind_sg
        self.get_indicator_g = get_ind_sg

        def get_dind_alpha(ia):
            if ia == 0:
                def dind(x):
                    if x <= alphas[ia]:
                        return 0
                    elif x <= alphas[ia + 1]:
                        return -1
                    else:
                        return 0
            elif ia == len(alphas) - 1:
                def dind(x):
                    if x >= alphas[ia]:
                        return 0
                    elif x >= alphas[ia - 1]:
                        return 1
                    else:
                        return 0
            else:
                def dind(x):
                    if x >= alphas[ia - 1] and x <= alphas[ia]:
                        return 1
                    elif x >= alphas[ia] and x <= alphas[ia + 1]:
                        return -1
                    else:
                        return 0

            return dind
        self.get_dindicator_alpha = get_dind_alpha

        def get_dind_g(wn_sg, ia):
            dind_a = self.get_dindicator_alpha(ia)
            dind_g = np.array([dind_a(v)
                               for v
                               in wn_sg.reshape(-1)]).reshape(wn_sg.shape)

            return dind_g
        self.get_dindicator_sg = get_dind_g
        self.get_dindicator_g = get_dind_g

    def distribute_alphas(self, nindicators, rank, size):
        nalphas = nindicators // size
        nalphas0 = nalphas + (nindicators - nalphas * size)
        assert (nalphas * (size - 1) + nalphas0 == nindicators)

        if rank == 0:
            start = 0
            end = nalphas0
        else:
            start = nalphas0 + (rank - 1) * nalphas
            end = start + nalphas

        return range(start, end)

    def alt_weight(self, wn_sg, my_alpha_indices, gd):
        nstar_sg = np.zeros_like(wn_sg)
        
        for ia in my_alpha_indices:
            nstar_sg += self.apply_kernel(wn_sg, ia, gd)
         
        if not (nstar_sg >= 0).all():
            np.save("nstar_sg", nstar_sg)
        assert (nstar_sg >= 0).all()

        return nstar_sg

    def apply_kernel(self, wn_sg, ia, gd):
        f_sg = self.get_indicator_sg(wn_sg, ia) * wn_sg
        f_sG = self.fftn(f_sg, axes=(1, 2, 3))

        w_sG = self.get_weight_function(ia, gd, self.alphas)
        
        r_sg = self.ifftn(w_sG * f_sG, axes=(1, 2, 3))

        assert np.allclose(r_sg, r_sg.real)

        return r_sg.real

    def fftn(self, arr, axes=None):
        if axes is None:
            sqrtN = np.sqrt(np.array(arr.shape).prod())
        else:
            sqrtN = 1
            for ax in axes:
                sqrtN *= arr.shape[ax]
            sqrtN = np.sqrt(sqrtN)

        return np.fft.fftn(arr, axes=axes, norm="ortho") / sqrtN
    
    def ifftn(self, arr, axes=None):
        if axes is None:
            sqrtN = np.sqrt(np.array(arr.shape).prod())
        else:
            sqrtN = 1
            for ax in axes:
                sqrtN *= arr.shape[ax]
            sqrtN = np.sqrt(sqrtN)

        return np.fft.ifftn(arr, axes=axes, norm="ortho") * sqrtN

    def get_weight_function(self, ia, gd, alphas):
        alpha = alphas[ia]

        kF = (3 * np.pi**2 * alpha)**(1 / 3)

        K_G = self._get_K_G(gd)

        res = (1 / (1 + (K_G / (kF + 0.0001))**2)**2)
        res = (res / res[0, 0, 0]).astype(np.complex128)
        assert not np.isnan(res).any()
        return res

    def _get_K_G(self, gd):
        assert gd.comm.size == 1
        k2_Q, _ = construct_reciprocal(gd)
        k2_Q[0, 0, 0] = 0
        return k2_Q**(1 / 2)

    def calculate_wlda(self, wn_sg, nstar_sg, my_alpha_indices):
        # Calculate the XC energy and potential that corresponds
        # to E_XC = \int dr n(r) e_xc(n*(r))
        assert (wn_sg >= 0).all()

        exc_g = np.zeros_like(wn_sg[0])
        vxc_sg = np.zeros_like(wn_sg)

        exc2_g = np.zeros_like(wn_sg[0])
        vxc2_sg = np.zeros_like(wn_sg)

        exc3_g = np.zeros_like(wn_sg[0])
        vxc3_sg = np.zeros_like(wn_sg)

        if len(wn_sg) == 1:
            # self.lda_x1(0, exc_g, wn_sg[0],
            # nstar_sg[0], vxc_sg[0], my_alpha_indices)
            zeta = 0
            # self.lda_c1(0, exc_g, wn_sg[0],
              #          nstar_sg[0], vxc_sg[0], zeta, my_alpha_indices)

            self.lda_x2(0, exc2_g, wn_sg[0],
                        nstar_sg[0], vxc2_sg[0], my_alpha_indices)
            zeta = 0
            self.lda_c2(0, exc2_g, wn_sg[0],
                        nstar_sg[0], vxc2_sg[0], zeta, my_alpha_indices)

            
            # self.lda_x3(0, exc3_g, wn_sg[0],
               #         nstar_sg[0], vxc3_sg[0], my_alpha_indices)
            zeta = 0
            # self.lda_c3(0, exc3_g, wn_sg[0],
                #        nstar_sg[0], vxc3_sg[0], zeta, my_alpha_indices)

            
        else:
            assert False
            na = 2.0 * nstar_sg[0]
            nb = 2.0 * nstar_sg[1]
            n = 0.5 * (na + nb)
            zeta = 0.5 * (na - nb) / n
            
            self.lda_x(1, exc_g, na, vxc_sg[0], my_alpha_indices)
            self.lda_x(1, exc_g, nb, vxc_sg[1], my_alpha_indices)
            self.lda_c(1, exc_g, n, vxc_sg, zeta, my_alpha_indices)
        
        return exc_g + exc2_g - exc3_g, vxc_sg + vxc2_sg - vxc3_sg

    def lda_x1(self, spin, e, wn_g, nstar_g, v, my_alpha_indices):
        from gpaw.xc.lda import lda_constants
        assert spin in [0, 1]
        C0I, C1, CC1, CC2, IF2 = lda_constants()
        
        rs = (C0I / nstar_g) ** (1 / 3.)
        ex = C1 / rs

        dexdrs = -ex / rs

        if spin == 0:
            e[:] += wn_g * ex
        else:
            e[:] += 0.5 * wn_g * ex
        v += ex
        t1 = rs * dexdrs / 3.
        v += self.fold_with_derivative(-t1 * wn_g / nstar_g,
                                       wn_g, my_alpha_indices)

    def lda_x2(self, spin, e, wn_g, nstar_g, v, my_alpha_indices):
        from gpaw.xc.lda import lda_constants
        assert spin in [0, 1]
        C0I, C1, CC1, CC2, IF2 = lda_constants()
        
        rs = (C0I / wn_g) ** (1 / 3.)
        ex = C1 / rs
        dexdrs = -ex / rs
        if spin == 0:
            e[:] += nstar_g * ex
        else:
            e[:] += 0.5 * nstar_g * ex
        v += self.fold_with_derivative(ex, wn_g, my_alpha_indices)
        t1 = rs * dexdrs / 3 * nstar_g / wn_g
        t2 = wn_g**(-2/3) * C1 / (C0I**(1/3)) * 1 / 3 * nstar_g
        assert np.allclose(-t1, t2)
        v -= rs * dexdrs / 3 * nstar_g / wn_g

    def lda_x3(self, spin, e, wn_g, nstar_g, v, my_alpha_indices):
        from gpaw.xc.lda import lda_constants
        assert spin in [0, 1]
        C0I, C1, CC1, CC2, IF2 = lda_constants()
        
        rs = (C0I / nstar_g) ** (1 / 3.)
        ex = C1 / rs
        dexdrs = -ex / rs
        if spin == 0:
            e[:] += nstar_g * ex
        else:
            e[:] += 0.5 * nstar_g * ex
        v = ex - rs * dexdrs / 3.
        v = self.fold_with_derivative(v, wn_g, my_alpha_indices)

    def lda_c1(self, spin, e, wn_g, nstar_g, v, zeta, my_alpha_indices):
        assert spin in [0, 1]
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()
        
        rs = (C0I / nstar_g) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        
        if spin == 0:
            e[:] += wn_g * ec
            v += ec
            v -= self.fold_with_derivative(rs * decdrs_0 / 3. * wn_g / nstar_g,
                                           wn_g, my_alpha_indices)
        else:
            e1, decdrs_1 = G(rs ** 0.5,
                             0.015545, 0.20548, 14.1189,
                             6.1977, 3.3662, 0.62517)
            alpha, dalphadrs = G(rs ** 0.5,
                                 0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                                 0.49671)
            alpha *= -1.
            dalphadrs *= -1.
            zp = 1.0 + zeta
            zm = 1.0 - zeta
            xp = zp ** (1 / 3.)
            xm = zm ** (1 / 3.)
            f = CC1 * (zp * xp + zm * xm - 2.0)
            f1 = CC2 * (xp - xm)
            zeta3 = zeta * zeta * zeta
            zeta4 = zeta * zeta * zeta * zeta
            x = 1.0 - zeta4
            decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                      decdrs_1 * f * zeta4 +
                      dalphadrs * f * x * IF2)
            decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                        f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
            ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4
            e[:] += wn_g * ec

            v[0] += ec
            v[0] -= self.fold_with_derivative((rs * decdrs / 3.0
                                               - (zeta - 1.0)
                                               * decdzeta * wn_g / nstar_g),
                                              wn_g, my_alpha_indices)
            
            v[1] += ec
            v[1] -= self.fold_with_derivative((rs * decdrs / 3.0
                                               - (zeta + 1.0)
                                               * decdzeta * wn_g / nstar_g),
                                              wn_g, my_alpha_indices)

    def lda_c2(self, spin, e, wn_g, nstar_g, v, zeta, my_alpha_indices):
        assert spin in [0, 1]
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()
        
        rs = (C0I / wn_g) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        
        if spin == 0:
            e[:] += nstar_g * ec
            v += self.fold_with_derivative(ec, wn_g, my_alpha_indices)
            v -= rs * decdrs_0 / 3. * nstar_g / wn_g
            # v += ec
            # v -= self.fold_with_derivative(rs * decdrs_0 / 3. * wn_g / nstar_g,
                                 #          wn_g, my_alpha_indices)
        else:
            e1, decdrs_1 = G(rs ** 0.5,
                             0.015545, 0.20548, 14.1189,
                             6.1977, 3.3662, 0.62517)
            alpha, dalphadrs = G(rs ** 0.5,
                                 0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                                 0.49671)
            alpha *= -1.
            dalphadrs *= -1.
            zp = 1.0 + zeta
            zm = 1.0 - zeta
            xp = zp ** (1 / 3.)
            xm = zm ** (1 / 3.)
            f = CC1 * (zp * xp + zm * xm - 2.0)
            f1 = CC2 * (xp - xm)
            zeta3 = zeta * zeta * zeta
            zeta4 = zeta * zeta * zeta * zeta
            x = 1.0 - zeta4
            decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                      decdrs_1 * f * zeta4 +
                      dalphadrs * f * x * IF2)
            decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                        f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
            ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4
            e[:] += wn_g * ec

            v[0] += self.fold_with_derivative(ec, wn_g, my_alpha_indices)
            v[0] -= (rs * decdrs / 3.0
                     - (zeta - 1.0)
                     * decdzeta * nstar_g / wn_g) 

            v[1] += self.fold_with_derivative(ec, wn_g, my_alpha_indices)
            v[1] -= (rs * decdrs / 3.0
                     - (zeta + 1.0) * decdzeta) * nstar_g / wn_g

    def lda_c3(self, spin, e, wn_g, nstar_g, v, zeta, my_alpha_indices):
        assert spin in [0, 1]
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()
        
        rs = (C0I / wn_g) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        
        if spin == 0:
            e[:] += nstar_g * ec
            v += ec - rs * decdrs_0 / 3.
            v[:] = self.fold_with_derivative(v,
                                           wn_g, my_alpha_indices)
        else:
            e1, decdrs_1 = G(rs ** 0.5,
                             0.015545, 0.20548, 14.1189,
                             6.1977, 3.3662, 0.62517)
            alpha, dalphadrs = G(rs ** 0.5,
                                 0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                                 0.49671)
            alpha *= -1.
            dalphadrs *= -1.
            zp = 1.0 + zeta
            zm = 1.0 - zeta
            xp = zp ** (1 / 3.)
            xm = zm ** (1 / 3.)
            f = CC1 * (zp * xp + zm * xm - 2.0)
            f1 = CC2 * (xp - xm)
            zeta3 = zeta * zeta * zeta
            zeta4 = zeta * zeta * zeta * zeta
            x = 1.0 - zeta4
            decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                      decdrs_1 * f * zeta4 +
                      dalphadrs * f * x * IF2)
            decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                        f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
            ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4
            e[:] += wn_g * ec

            v[0] += ec - (rs * decdrs / 3.0 - (zeta - 1.0) * decdzeta)
            v[0] -= self.fold_with_derivative(v[0],
                                              wn_g, my_alpha_indices)
            
            v[1] += ec - (rs * decdrs / 3.0 - (zeta + 1.0) * decdzeta)
            v[1] = self.fold_with_derivative(v[1],
                                              wn_g, my_alpha_indices)
            
    def fold_with_derivative(self, f_g, n_g, my_alpha_indices):
        assert np.allclose(f_g, f_g.real)
        assert np.allclose(n_g, n_g.real)

        res_g = np.zeros_like(f_g)

        for ia in my_alpha_indices:
            ind_g = self.get_indicator_g(n_g, ia)
            dind_g = self.get_dindicator_g(n_g, ia)
            
            fac_g = ind_g + dind_g * n_g
            int_G = self.fftn(f_g)
            w_G = self.get_weight_function(ia, self.gd, self.alphas)
            w_g = self.ifftn(w_G)
            assert np.allclose(w_g, w_g.real)
            r_g = self.ifftn(w_G * int_G)
            assert np.allclose(r_g, r_g.real)
            res_g += r_g.real * fac_g
            assert np.allclose(res_g, res_g.real)

        mpi.world.sum(res_g)
        assert res_g.shape == f_g.shape
        assert res_g.shape == n_g.shape
        return res_g

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        return 0
