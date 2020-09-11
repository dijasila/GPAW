from gpaw.xc.functional import XCFunctional
import numpy as np
from gpaw.utilities.tools import construct_reciprocal
import gpaw.mpi as mpi
import datetime
from functools import wraps
from time import time
from ase.parallel import parprint

def timer(f):
    
    @wraps(f)
    def wrapped(*args):
        t1 = time()
        res = f(*args)
        t2 = time()
        parprint(f"{f.__name__} took {t2-t1} seconds")
        return res

    return wrapped


class WLDA(XCFunctional):
    def __init__(self, rc=None, kernel_type=None,
                 kernel_param=None, wlda_type=None,
                 nindicators=None):
        XCFunctional.__init__(self, 'WLDA', 'LDA')

        self.nindicators = nindicators or int(5 * 1e2)
        self.rcut_factor = rc
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.wlda_type = wlda_type
        if wlda_type not in ['1', '2', '3', 'c']:
            raise ValueError('WLDA type {} not recognized'.format(wlda_type))
        self.energies = []
        self._K_G = None
        
    def initialize(self, density, hamiltonian, wfs, occupations=None):
        self.density = density
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.occupations = occupations

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        """Interface for GPAW."""
        wn_sg = self.get_working_density(n_sg, gd)

        # Construct arrays for un-distributed energy and potential
        exc_g = np.zeros_like(wn_sg[0])
        vxc_sg = np.zeros_like(wn_sg)

        # If spin-paired we can reuse the weighted density
        nstar_sg, alpha_indices = self.wlda_x(wn_sg, exc_g, vxc_sg)

        nstar_sg = self.wlda_c(wn_sg, exc_g, vxc_sg,
                               nstar_sg, alpha_indices)

        self.hartree_correction(self.gd, exc_g, wn_sg, nstar_sg)

        gd.distribute(exc_g, e_g)
        gd.distribute(vxc_sg, v_sg)

    @timer
    def get_working_density(self, n_sg, gd):
        """Construct the "working" density.

        The working density is an approximation to the AE density
        (or sometimes not depending on the selected mode).

        The construction consists of the following steps:

        1. Correct negative density values.
        2. Collect density, so all ranks have full density.
        3. Construct new grid_descriptor for collected density.
        4. Apply approximation to AE density (or not).

        The approximation scheme involves truncating the AE density
        at points closer than some radius of the atoms and replacing full
        density with a smooth polynomial. This procedure is implemented
        in the setup class (gpaw/setup.py).
        """
        n_sg[n_sg < 1e-20] = 1e-40
        wn_sg = gd.collect(n_sg, broadcast=True)
        gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        self.gd = gd1
        wn_sg = self.density_correction(self.gd, wn_sg)
        wn_sg[wn_sg < 1e-20] = 1e-20

        return wn_sg

    @timer
    def wlda_x(self, wn_sg, exc_g, vxc_sg):
        """
        Calculate WLDA exchange energy.
        
        If the system is spin-polarized, we use the formula
            E_x[n_up, n_down] = (E[2n_up] + E[2n_down])/2

        If the system is spin-paired, we can reuse the
        weighted density because we don't calculate the
        exchange functional with 2 times the density.

        Thus

        If spin-paired return density and alpha_indices

        If spin-polarized return nothing (None)
        """
        if len(wn_sg) == 2:
            # If spin-polarized calculate exchange energy according to
            # E[n_up, n_down] = (E[2n_up] + E[2n_down]) / 2
            wn_sg *= 2
            
        nstar_sg, alpha_indices = self.get_weighted_density(wn_sg)

        e1_g = np.zeros_like(wn_sg[0])
        e2_g = np.zeros_like(wn_sg[0])
        e3_g = np.zeros_like(wn_sg[0])
        v1_sg = np.zeros_like(wn_sg)
        v2_sg = np.zeros_like(wn_sg)
        v3_sg = np.zeros_like(wn_sg)

        if len(wn_sg) == 1:
            spin = 0
            self.lda_x1(spin, e1_g, wn_sg[0], nstar_sg[0], v1_sg[0],
                        alpha_indices)

            self.lda_x2(spin, e2_g, wn_sg[0], nstar_sg[0], v2_sg[0],
                        alpha_indices)

            self.lda_x3(spin, e3_g, wn_sg[0], nstar_sg[0], v3_sg[0],
                        alpha_indices)
            exc_g[:] = e1_g + e2_g - e3_g
            vxc_sg[:] = v1_sg + v2_sg - v3_sg
            return nstar_sg, alpha_indices
        else:
            spin = 1
            self.lda_x1(spin, e1_g, wn_sg[0], nstar_sg[0], v1_sg[0],
                        alpha_indices)
            self.lda_x1(spin, e1_g, wn_sg[1], nstar_sg[1], v1_sg[1],
                        alpha_indices)

            self.lda_x2(spin, e2_g, wn_sg[0], nstar_sg[0], v2_sg[0],
                        alpha_indices)
            self.lda_x2(spin, e2_g, wn_sg[1], nstar_sg[1], v2_sg[1],
                        alpha_indices)

            self.lda_x3(spin, e3_g, wn_sg[0], nstar_sg[0], v3_sg[0],
                        alpha_indices)
            self.lda_x3(spin, e3_g, wn_sg[1], nstar_sg[1], v3_sg[1],
                        alpha_indices)

            exc_g[:] = e1_g + e2_g - e3_g
            vxc_sg[:] = v1_sg + v2_sg - v3_sg
            return None, None

    @timer
    def wlda_c(self, wn_sg, exc_g, vxc_sg, nstar_sg, alpha_indices):
        """Calculate the WLDA correlation energy.
        
        If the system is spin-paired we calculate the correlation
        energy as
            E_c[n] = int e_ldac(n*)(n-n*) + e_ldac(n)n*

        If the system is spin-polarized we calculate the correlation
        energy as
            E_c[n_up, n_down] =
                   int e_ldac(n*, zeta)(n-n*) + e_ldac(n, zeta)n*
        """
        if nstar_sg is None or alpha_indices is None:
            assert len(wn_sg) == 2
            n = np.array([wn_sg[0] + wn_sg[1]])
            nstar_sg, alpha_indices = self.get_weighted_density(n)

        e1_g = np.zeros_like(wn_sg[0])
        e2_g = np.zeros_like(wn_sg[0])
        e3_g = np.zeros_like(wn_sg[0])
        v1_sg = np.zeros_like(wn_sg)
        v2_sg = np.zeros_like(wn_sg)
        v3_sg = np.zeros_like(wn_sg)
            
        if len(wn_sg) == 1:
            spin = zeta = 0
            self.lda_c1(0, e1_g, wn_sg[0], nstar_sg[0],
                        v1_sg[0], zeta, alpha_indices)
            self.lda_c2(0, e2_g, wn_sg[0], nstar_sg[0],
                        v2_sg[0], zeta, alpha_indices)
            self.lda_c3(0, e3_g, wn_sg[0], nstar_sg[0],
                        v3_sg[0], zeta, alpha_indices)
        else:
            spin = 1
            zeta_g = (wn_sg[0] - wn_sg[1]) / (wn_sg[0] + wn_sg[1])
            zeta_g[np.isclose(wn_sg[0] + wn_sg[1], 0.0)] = 0.0
            self.lda_c1(spin, e1_g, wn_sg, nstar_sg, v1_sg, zeta_g,
                        alpha_indices)
            self.lda_c2(spin, e2_g, wn_sg, nstar_sg, v2_sg, zeta_g,
                        alpha_indices)
            self.lda_c3(spin, e3_g, wn_sg, nstar_sg, v3_sg, zeta_g,
                        alpha_indices)

        exc_g[:] += e1_g + e2_g - e3_g
        vxc_sg[:] += v1_sg + v2_sg - v3_sg
        return nstar_sg

    @timer
    def get_weighted_density(self, wn_sg):
        """Set up the weighted density.

        This is done by applying the mapping

            n |-> n*(r) = int phi(r-r', n(r')) n(r')dr'
        This is done via the convolution theorem by replacing

            phi(r-r', n(r')) -> sum_a phi(r-r', a)f_a(r')n(r')

        f_a(r') is close to one if n(r') is close to a.

        Each rank has some subset of a's.

        First construct and distribute the as across the ranks.
        
        Then apply the weighting functional.

        Sum across ranks so all ranks have
        the full weighted density.

        Finally correct potentially negative values from
        numerically inaccuracy to avoid complex energies.
        """
        alpha_indices = self.construct_alphas(wn_sg)

        nstar_sg = self.alt_weight(wn_sg, alpha_indices, self.gd)

        mpi.world.sum(nstar_sg)

        nstar_sg[nstar_sg < 1e-20] = 1e-40

        return nstar_sg, alpha_indices

    def construct_alphas(self, wn_sg):
        """Construct indicator data.
        
        Construct alphas that are used to approximately
        calculate the weighted density.

        The alphas should cover all the values of the
        density. A denser grid should be more accurate,
        but will be more computationally expensive.

        After setting up the full grid of alphas, each rank
        determines which alphas it should calculate.
        """
        all_alphas = self.setup_indicator_grid(self.nindicators,
                                               wn_sg)

        self.alphas = all_alphas
        
        self.setup_indicators(all_alphas)

        alpha_indices = self.distribute_alphas(self.nindicators,
                                               mpi.rank, mpi.size)

        return alpha_indices

    def setup_indicator_grid(self, nindicators, n_sg):
        """Set up indicator values.
        
        These values are used to calculate
            int phi(r-r', n(r'))n(r') dr'

        approximately via the convolution theorem.
        We approximate it with
            sum_a int phi(r-r', a) f_a(n(r'))n(r') dr'

        where the a values are the ones chosen here.
        They should span all the values of the density.
        A denser grid should be more accurate but will be
        more computationally heavy.
        """
        md = np.min(n_sg)
        md = max(md, 1e-6)
        mad = np.max(n_sg)
        mad = max(mad, 1e-6)
        if np.allclose(md, mad):
            mad = 2 * mad
        # This is an alternate form of the alpha grid
        # return np.exp(np.linspace(np.log(md * 0.9),
        # np.log(mad * 1.1), nindicators))
        return np.linspace(md * 0.9, mad * 1.1, nindicators)
        
    def setup_indicators(self, alphas):
        """Set up indicator functions and derivatives.

        TODO Refactor
        """
        
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

        self.get_indicator_sg = timer(get_ind_sg)
        self.get_indicator_g = timer(get_ind_sg)

        def get_dind_alpha(ia):
            if ia == 0:
                def dind(x):
                    if x <= alphas[ia]:
                        return 0.0
                    elif x <= alphas[ia + 1]:
                        return -1.0
                    else:
                        return 0.0
            elif ia == len(alphas) - 1:
                def dind(x):
                    if x >= alphas[ia]:
                        return 0.0
                    elif x >= alphas[ia - 1]:
                        return 1.0
                    else:
                        return 0.0
            else:
                def dind(x):
                    if x >= alphas[ia - 1] and x <= alphas[ia]:
                        return 1.0
                    elif x >= alphas[ia] and x <= alphas[ia + 1]:
                        return -1.0
                    else:
                        return 0.0

            return dind
        self.get_dindicator_alpha = get_dind_alpha

        def get_dind_g(wn_sg, ia):
            dind_a = self.get_dindicator_alpha(ia)
            dind_g = np.array([dind_a(v)
                               for v
                               in wn_sg.reshape(-1)]).reshape(wn_sg.shape)

            return dind_g
        self.get_dindicator_sg = timer(get_dind_g)
        self.get_dindicator_g = timer(get_dind_g)

    def distribute_alphas(self, nindicators, rank, size):
        """Distribute alphas across mpi ranks."""
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
        """Calculate the weighted density.

        Calculate
        
            n*(r) = int phi(r-r', a)f_a(r')n('r)

        via the convolution theorem

            n*(r) = phi * (f_a n)
        """
        nstar_sg = np.zeros_like(wn_sg)
        
        for ia in my_alpha_indices:
            nstar_sg += self.apply_kernel(wn_sg, ia, gd)
            
        return nstar_sg

    def apply_kernel(self, wn_sg, ia, gd):
        """Apply the WLDA kernel at given alpha.
        
        Applies the kernel via the convolution theorem.
        """
        f_sg = self.get_indicator_sg(wn_sg, ia) * wn_sg
        f_sG = self.fftn(f_sg, axes=(1, 2, 3))

        w_sG = self.get_weight_function(ia, gd, self.alphas)
        
        r_sg = self.ifftn(w_sG * f_sG, axes=(1, 2, 3))

        return r_sg.real

    def fftn(self, arr, axes=None):
        """Interface for fftn."""
        if axes is None:
            sqrtN = np.sqrt(np.array(arr.shape).prod())
        else:
            sqrtN = 1
            for ax in axes:
                sqrtN *= arr.shape[ax]
            sqrtN = np.sqrt(sqrtN)

        return np.fft.fftn(arr, axes=axes)  # , norm="ortho") / sqrtN
    
    def ifftn(self, arr, axes=None):
        """Interface for ifftn."""
        if axes is None:
            sqrtN = np.sqrt(np.array(arr.shape).prod())
        else:
            sqrtN = 1
            for ax in axes:
                sqrtN *= arr.shape[ax]
            sqrtN = np.sqrt(sqrtN)

        return np.fft.ifftn(arr, axes=axes)  # , norm="ortho") * sqrtN

    def get_weight_function(self, ia, gd, alphas):
        """Construct the weight function/kernel in fourier space."""
        alpha = alphas[ia]
        kF = (3 * np.pi**2 * alpha)**(1 / 3)
        K_G = self._get_K_G(gd)

        kernel_fn = self.get_kernel_fn(self.kernel_type, self.kernel_param)
        res = kernel_fn(kF, K_G)

        res = (res / res[0, 0, 0]).astype(np.complex128)
        assert not np.isnan(res).any()
        return res

    def get_kernel_fn(self, kernel_type, kernel_param):
        """Select kernel type.

        Select the functional form for the weight kernel
        based on settings.
        """
        p = self.kernel_param or 1
        if kernel_type == "lorentz5":
            def kernel(kF, K_G):
                return 1 / (1 + 0.005 * p * (K_G / (kF + 0.0001)**5)**2)
            return kernel
        else:
            def kernel(kF, K_G):
                return 1 / (1 + 0.005 * p * (K_G / (kF + 0.0001)**6)**2)
            return kernel
        return kernel

    def _get_K_G(self, gd):
        """Get the norms of the reciprocal lattice vectors."""
        if self._K_G is not None:
            return self._K_G
        assert gd.comm.size == 1
        k2_Q, _ = construct_reciprocal(gd)
        k2_Q[0, 0, 0] = 0
        self._K_G = k2_Q**(1 / 2)
        return self._K_G

    def lda_x1(self, spin, e, wn_g, nstar_g, v, my_alpha_indices):
        """Apply the WLDA-1 exchange functional.

        Calculate e[n*]n
        """
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
        ratio = wn_g / nstar_g
        ratio[np.isclose(nstar_g, 0.0)] = 0.0
        v += self.fold_with_derivative(-t1 * ratio,
                                       wn_g, my_alpha_indices)

    def lda_x2(self, spin, e, wn_g, nstar_g, v, my_alpha_indices):
        """Apply the WLDA-2 exchange functional.

        Calculate e[n]n*
        """
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
        ratio = nstar_g / wn_g
        ratio[np.isclose(wn_g, 0.0)] = 0.0
        v -= rs * dexdrs / 3 * ratio

    def lda_x3(self, spin, e, wn_g, nstar_g, v, my_alpha_indices):
        """Apply the WLDA-3 exchange functional.

        Calculate e[n*]n*
        """
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
        v[:] = ex - rs * dexdrs / 3.
        v[:] = self.fold_with_derivative(v, wn_g, my_alpha_indices)

    def lda_c1(self, spin, e, wn_g, nstar_g, v, zeta, my_alpha_indices):
        """Apply the WLDA-1 correlation functional.

        Calculate
            e^lda_c[n*]n
        and
            (de^lda_c[n*] / dn*) * (dn* / dn) + e^lda_c[n*]
        """
        assert spin in [0, 1]
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()
        
        rs = (C0I / nstar_g) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        
        if spin == 0:
            e[:] += wn_g * ec
            v += ec
            ratio = wn_g / nstar_g
            ratio[np.isclose(nstar_g, 0.0)] = 0.0
            v -= self.fold_with_derivative(rs * decdrs_0 / 3. * ratio,
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
            ratio = wn_g / nstar_g
            ratio[np.isclose(nstar_g, 0.0)] = 0.0
            v[0] -= self.fold_with_derivative((rs * decdrs / 3.0
                                               - (zeta - 1.0)
                                               * decdzeta * ratio[0]),
                                              wn_g, my_alpha_indices)
            
            v[1] += ec
            v[1] -= self.fold_with_derivative((rs * decdrs / 3.0
                                               - (zeta + 1.0)
                                               * decdzeta * ratio[1]),
                                              wn_g, my_alpha_indices)

    def lda_c2(self, spin, e, wn_g, nstar_g, v, zeta, my_alpha_indices):
        """Apply the WLDA-2 correlation functional.

        Calculate
            e^lda_c[n] n*
        and
            (e^lda_c[n]) * (dn* / dn) + de^lda_c/dn n*
        """
        assert spin in [0, 1]
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()
        
        rs = (C0I / wn_g) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        
        if spin == 0:
            e[:] += nstar_g * ec
            v += self.fold_with_derivative(ec, wn_g, my_alpha_indices)
            ratio = nstar_g / wn_g
            ratio[np.isclose(wn_g, 0.0)] = 0.0
            v -= rs * decdrs_0 / 3. * ratio
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
            ratio = nstar_g / wn_g
            ratio[np.isclose(wn_g, 0.0)] = 0.0
            v[0] -= (rs * decdrs / 3.0
                     - (zeta - 1.0)
                     * decdzeta * ratio[0])

            v[1] += self.fold_with_derivative(ec, wn_g, my_alpha_indices)
            v[1] -= (rs * decdrs / 3.0
                     - (zeta + 1.0) * decdzeta) * ratio[1]

    def lda_c3(self, spin, e, wn_g, nstar_g, v, zeta, my_alpha_indices):
        """Apply the WLDA-3 correlation functional to n*.

        Calculate
            e^lda_c[n*]n* (used in the energy)
        and
            (v^lda_c[n*]) * (dn* / dn) (the potential from the correlation)
        """
        assert spin in [0, 1]
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()
        
        rs = (C0I / nstar_g) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        
        if spin == 0:
            e[:] += nstar_g * ec
            v_place = np.zeros_like(v)
            v_place += ec - rs * decdrs_0 / 3.
            v += self.fold_with_derivative(v_place,
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
            v[0] = self.fold_with_derivative(v[0],
                                             wn_g, my_alpha_indices)
            
            v[1] += ec - (rs * decdrs / 3.0 - (zeta + 1.0) * decdzeta)
            v[1] = self.fold_with_derivative(v[1],
                                             wn_g, my_alpha_indices)

    @timer            
    def fold_with_derivative(self, f_g, n_g, my_alpha_indices):
        """Fold function f_g with the derivative of the weighted density.

        Calculate
            (f) * (dn*/dn) = int dr' f(r') dn*(r') / dn(r)
        """
        assert np.allclose(f_g, f_g.real)
        assert np.allclose(n_g, n_g.real)

        res_g = np.zeros_like(f_g)

        for ia in my_alpha_indices:
            ind_g = self.get_indicator_g(n_g, ia)
            dind_g = self.get_dindicator_g(n_g, ia)
            
            fac_g = ind_g + dind_g * n_g
            int_G = self.fftn(f_g)
            w_G = self.get_weight_function(ia, self.gd, self.alphas)
            r_g = self.ifftn(w_G * int_G)
            res_g += r_g.real * fac_g
            assert np.allclose(res_g, res_g.real)

        mpi.world.sum(res_g)
        assert res_g.shape == f_g.shape
        assert res_g.shape == n_g.shape
        return res_g

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        """Return trivial paw correction."""
        return 0

    def density_correction(self, gd, n_sg):
        """Apply density correction.

        This approximates the AE density around the atoms.
        The true density is replaced by a smooth function
        close to the atoms. The distance within which the
        AE density is replaced is determined by self.rcut_factor.

        If self.rcut_factor is None, the pseudo density is returned.
        """
        if self.rcut_factor:
            from gpaw.xc.WDAUtils import correct_density
            return correct_density(n_sg, gd, self.wfs.setups,
                                   self.wfs.spos_ac, self.rcut_factor)
        else:
            return n_sg

    @timer
    def hartree_correction(self, gd, e_g, wn_sg, nstar_sg):
        """Apply the correction to the Hartree energy.
        
        Calculate
            Delta E = -0.5 int drdr' (n-n*)(r)(n-n*)(r')/|r-r'|

        This corrects the Hartree energy so it is effectively
        calculated with n* instead of n.
        """
        if mpi.rank != 0:
            # We should only calculate the Hartree
            # correction once.
            return
        if nstar_sg is None:
            nstar_sg, _ = self.get_weighted_density(wn_sg)

        K_G = self._get_K_G(gd)
        total_g = (wn_sg - nstar_sg).sum(axis=0)
        total_G = self.fftn(total_g)
        total_G *= -2 * np.pi / K_G**2

        e_g[:] += (total_g * self.ifftn(total_G).real)

    ###### Abandon all hope ye who enter here #######
    def _calculate_impl(self, gd, n_sg, v_sg, e_g):
        """Interface for GPAW."""
        # 0. Collect density,
        # and get grid_descriptor appropriate for collected density
        n_sg[n_sg < 1e-20] = 1e-40
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
        wn_sg = self.density_correction(self.gd, wn_sg)
        wn_sg[wn_sg < 1e-20] = 1e-20
        # 2. calculate weighted density
        # This contains contributions for the alphas at this
        # rank, i.e. we need a world.sum to get all contributions
        nstar_sg = self.alt_weight(wn_sg, my_alpha_indices, gd1)
        mpi.world.sum(nstar_sg)
        nstar_sg[nstar_sg < 1e-20] = 1e-40
        if mpi.rank == 0 and not (nstar_sg >= 0.0).all():
            np.save("nstar_sg.npy", nstar_sg)
        assert (nstar_sg >= 0.0).all()
        # 3. Calculate LDA energy
        e1_g, v1_sg = self.calculate_wlda(wn_sg, nstar_sg, my_alpha_indices)
        
        e_g *= 0.0
        v_sg *= 0.0
        gd.distribute(e1_g, e_g)
        gd.distribute(v1_sg, v_sg)

        if hasattr(self, "energies"):
            niter = getattr(self, "niter", 0)
            ms = datetime.datetime.microsecond
            print("AWDOJAWPDO")
            if len(self.energies) >= 2:
                print("Have and energies")
                err = abs(self.energies[-1] - self.energies[-2])
                if self.niter > 5 and err > 1:
                    if mpi.rank == 0:
                        np.save(f"{niter}_{ms}_wn_sg.npy", wn_sg)
                        np.save(f"{niter}_{ms}_v_sg.npy", v_sg)

        else:
            print("Dont have energies")

    def calculate_wlda(self, wn_sg, nstar_sg, my_alpha_indices):
        """Perform the old wlda implementation."""
        # Calculate the XC energy and potential that corresponds
        # to E_XC = \int dr n(r) e_xc(n*(r))
        assert (wn_sg >= 0).all()

        exc_g = np.zeros_like(wn_sg[0])
        vxc_sg = np.zeros_like(wn_sg)

        exc2_g = np.zeros_like(wn_sg[0])
        vxc2_sg = np.zeros_like(wn_sg)

        exc3_g = np.zeros_like(wn_sg[0])
        vxc3_sg = np.zeros_like(wn_sg)

        t = self.wlda_type
        if len(wn_sg) == 1:
            if t == '1' or t == 'c':
                self.lda_x1(0, exc_g, wn_sg[0],
                            nstar_sg[0], vxc_sg[0], my_alpha_indices)
                zeta = 0
                self.lda_c1(0, exc_g, wn_sg[0],
                            nstar_sg[0], vxc_sg[0], zeta, my_alpha_indices)

            if t == '2' or t == 'c':
                self.lda_x2(0, exc2_g, wn_sg[0],
                            nstar_sg[0], vxc2_sg[0], my_alpha_indices)
                zeta = 0
                self.lda_c2(0, exc2_g, wn_sg[0],
                            nstar_sg[0], vxc2_sg[0], zeta, my_alpha_indices)

            if t == '3' or t == 'c':
                self.lda_x3(0, exc3_g, wn_sg[0],
                            nstar_sg[0], vxc3_sg[0], my_alpha_indices)
                zeta = 0
                self.lda_c3(0, exc3_g, wn_sg[0],
                            nstar_sg[0], vxc3_sg[0], zeta, my_alpha_indices)
        else:
            assert False
            # nstara = 2.0 * nstar_sg[0]
            # nstarb = 2.0 * nstar_sg[1]
            # nstar = 0.5 * (na + nb)
            # zeta = 0.5 * (na - nb) /n
            
            # self.lda_x(1, exc_g, na, vxc_sg[0], my_alpha_indices)
            # self.lda_x(1, exc_g, nb, vxc_sg[1], my_alpha_indices)
            # self.lda_c(1, exc_g, n, vxc_sg, zeta, my_alpha_indices)

            # if t == '1' or t == 'c':
            #     self.lda_x1(0, exc_g, wn_sg[0],
            #                 nstar_sg[0], vxc_sg[0], my_alpha_indices)
            #     self.lda_x1(0, exc_g, wn_sg[0],
            #                 nstar_sg[0], vxc_sg[0], my_alpha_indices)
            #     zeta = 0
            #     self.lda_c1(0, exc_g, wn_sg[0],
            #                 nstar_sg[0], vxc_sg[0], zeta, my_alpha_indices)

            # if t == '2' or t == 'c':
            #     self.lda_x2(0, exc2_g, wn_sg[0],
            #                 nstar_sg[0], vxc2_sg[0], my_alpha_indices)
            #     zeta = 0
            #     self.lda_c2(0, exc2_g, wn_sg[0],
            #                 nstar_sg[0], vxc2_sg[0], zeta, my_alpha_indices)

            # if t == '3' or t == 'c':
            #     self.lda_x3(0, exc3_g, wn_sg[0],
            #                 nstar_sg[0], vxc3_sg[0], my_alpha_indices)
            #     zeta = 0
            #     self.lda_c3(0, exc3_g, wn_sg[0],
            #                 nstar_sg[0], vxc3_sg[0], zeta, my_alpha_indices)

        if t == '1':
            return exc_g, vxc_sg
        elif t == '2':
            return exc2_g, vxc2_sg
        elif t == '3':
            return exc3_g, vxc3_sg
        elif t == 'c':
            return exc_g + exc2_g - exc3_g, vxc_sg + vxc2_sg - vxc3_sg
        else:
            raise ValueError('WLDA type not recognized')
