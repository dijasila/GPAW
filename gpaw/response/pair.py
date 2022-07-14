import numbers
from pathlib import Path

import numpy as np

from ase.utils.timing import timer

import gpaw.mpi as mpi
from gpaw import disable_dry_run
from gpaw import GPAW
from gpaw.fd_operators import Gradient
from gpaw.response.pw_parallelization import block_partition
from gpaw.response.symmetry import KPointFinder
from gpaw.utilities.blas import mmm
from gpaw.response.context import new_context


class KPoint:
    def __init__(self, s, K, n1, n2, blocksize, na, nb,
                 ut_nR, eps_n, f_n, P_ani, shift_c):
        self.s = s    # spin index
        self.K = K    # BZ k-point index
        self.n1 = n1  # first band
        self.n2 = n2  # first band not included
        self.blocksize = blocksize
        self.na = na  # first band of block
        self.nb = nb  # first band of block not included
        self.ut_nR = ut_nR      # periodic part of wave functions in real-space
        self.eps_n = eps_n      # eigenvalues
        self.f_n = f_n          # occupation numbers
        self.P_ani = P_ani      # PAW projections
        self.shift_c = shift_c  # long story - see the
        # PairDensity.construct_symmetry_operators() method


class PairDistribution:
    def __init__(self, pair, mysKn1n2):
        self.pair = pair
        self.mysKn1n2 = mysKn1n2
        self.mykpts = [self.pair.get_k_point(s, K, n1, n2)
                       for s, K, n1, n2 in self.mysKn1n2]

    def kpt_pairs_by_q(self, q_c, m1, m2):
        pair = self.pair
        mykpts = self.mykpts
        for u, kpt1 in enumerate(mykpts):
            progress = u / len(mykpts)
            K2 = pair.kd.find_k_plus_q(q_c, [kpt1.K])[0]
            kpt2 = pair.get_k_point(kpt1.s, K2, m1, m2, block=True)

            yield progress, kpt1, kpt2


class KPointPair:
    """This class defines the kpoint-pair container object.

    Used for calculating pair quantities it contains two kpoints,
    and an associated set of Fourier components."""
    def __init__(self, kpt1, kpt2, Q_G):
        self.kpt1 = kpt1
        self.kpt2 = kpt2
        self.Q_G = Q_G

    def get_k1(self):
        """ Return KPoint object 1."""
        return self.kpt1

    def get_k2(self):
        """ Return KPoint object 2."""
        return self.kpt2

    def get_planewave_indices(self):
        """ Return the planewave indices associated with this pair."""
        return self.Q_G

    def get_transition_energies(self, n_n, m_m):
        """Return the energy difference for specified bands."""
        n_n = np.array(n_n)
        m_m = np.array(m_m)
        kpt1 = self.kpt1
        kpt2 = self.kpt2
        deps_nm = (kpt1.eps_n[n_n - self.kpt1.n1][:, np.newaxis] -
                   kpt2.eps_n[m_m - self.kpt2.n1])
        return deps_nm

    def get_occupation_differences(self, n_n, m_m):
        """Get difference in occupation factor between specified bands."""
        n_n = np.array(n_n)
        m_m = np.array(m_m)
        kpt1 = self.kpt1
        kpt2 = self.kpt2
        df_nm = (kpt1.f_n[n_n - self.kpt1.n1][:, np.newaxis] -
                 kpt2.f_n[m_m - self.kpt2.n1])
        return df_nm


class NoCalculatorPairDensity:
    def __init__(self, gs, *, context, ftol=1e-6,
                 threshold=1, real_space_derivatives=False, nblocks=1):
        self.gs = gs
        self.context = context

        self.fd = context.fd
        self.timer = context.timer
        self.world = context.world

        assert self.gs.kd.symmetry.symmorphic

        self.ftol = ftol
        self.threshold = threshold
        self.real_space_derivatives = real_space_derivatives

        self.blockcomm, self.kncomm = block_partition(context.world, nblocks)
        self.nblocks = nblocks

        self.fermi_level = self.gs.fermi_level
        self.spos_ac = self.gs.spos_ac

        self.nocc1 = None  # number of completely filled bands
        self.nocc2 = None  # number of non-empty bands
        self.count_occupied_bands()

        self.ut_sKnvR = None  # gradient of wave functions for optical limit

        self.vol = self.gs.gd.volume

        self.kd = self.gs.kd
        self.kptfinder = KPointFinder(self.kd.bzk_kc)
        print('Number of blocks:', nblocks, file=self.fd)

    def find_kpoint(self, k_c):
        return self.kptfinder.find(k_c)

    def count_occupied_bands(self):
        self.nocc1 = 9999999
        self.nocc2 = 0
        for kpt in self.gs.kpt_u:
            f_n = kpt.f_n / kpt.weight
            self.nocc1 = min((f_n > 1 - self.ftol).sum(), self.nocc1)
            self.nocc2 = max((f_n > self.ftol).sum(), self.nocc2)
        print('Number of completely filled bands:', self.nocc1, file=self.fd)
        print('Number of non-empty bands:', self.nocc2, file=self.fd)
        print('Total number of bands:', self.gs.bd.nbands,
              file=self.fd)

    def distribute_k_points_and_bands(self, band1, band2, kpts=None):
        """Distribute spins, k-points and bands.

        The attribute self.mysKn1n2 will be set to a list of (s, K, n1, n2)
        tuples that this process handles.
        """

        gs = self.gs

        if kpts is None:
            kpts = np.arange(gs.kd.nbzkpts)

        # nbands is the number of bands for each spin/k-point combination.
        nbands = band2 - band1
        size = self.kncomm.size
        rank = self.kncomm.rank
        ns = gs.nspins
        nk = len(kpts)
        n = (ns * nk * nbands + size - 1) // size
        i1 = min(rank * n, ns * nk * nbands)
        i2 = min(i1 + n, ns * nk * nbands)

        mysKn1n2 = []
        i = 0
        for s in range(ns):
            for K in kpts:
                n1 = min(max(0, i1 - i), nbands)
                n2 = min(max(0, i2 - i), nbands)
                if n1 != n2:
                    mysKn1n2.append((s, K, n1 + band1, n2 + band1))
                i += nbands

        print('BZ k-points:', gs.kd, file=self.fd)
        print('Distributing spins, k-points and bands (%d x %d x %d)' %
              (ns, nk, nbands),
              'over %d process%s' %
              (self.kncomm.size, ['es', ''][self.kncomm.size == 1]),
              file=self.fd)
        print('Number of blocks:', self.blockcomm.size, file=self.fd)

        return PairDistribution(self, mysKn1n2)

    @timer('Get a k-point')
    def get_k_point(self, s, k_c, n1, n2, load_wfs=True, block=False):
        """Return wave functions for a specific k-point and spin.

        s: int
            Spin index (0 or 1).
        K: int
            BZ k-point index.
        n1, n2: int
            Range of bands to include.
        """

        assert n1 <= n2

        gs = self.gs
        kd = gs.kd

        # Parse kpoint: is k_c an index or a vector
        if not isinstance(k_c, numbers.Integral):
            K = self.kptfinder.find(k_c)
            shift0_c = (kd.bzk_kc[K] - k_c).round().astype(int)
        else:
            # Fall back to index
            K = k_c
            shift0_c = np.array([0, 0, 0])
            k_c = None

        if block:
            nblocks = self.blockcomm.size
            rank = self.blockcomm.rank
        else:
            nblocks = 1
            rank = 0

        blocksize = (n2 - n1 + nblocks - 1) // nblocks
        na = min(n1 + rank * blocksize, n2)
        nb = min(na + blocksize, n2)

        U_cc, T, a_a, U_aii, shift_c, time_reversal = \
            self.construct_symmetry_operators(K, k_c=k_c)

        shift_c += -shift0_c
        ik = kd.bz2ibz_k[K]
        assert kd.comm.size == 1
        kpt = gs.kpt_qs[ik][s]

        assert n2 <= len(kpt.eps_n), \
            'Increase GS-nbands or decrease chi0-nbands!'
        eps_n = kpt.eps_n[n1:n2]
        f_n = kpt.f_n[n1:n2] / kpt.weight

        if not load_wfs:
            return KPoint(s, K, n1, n2, blocksize, na, nb,
                          None, eps_n, f_n, None, shift_c)

        with self.timer('load wfs'):
            psit_nG = kpt.psit_nG
            ut_nR = gs.gd.empty(nb - na, gs.dtype)
            for n in range(na, nb):
                ut_nR[n - na] = T(gs.pd.ifft(psit_nG[n], ik))

        with self.timer('Load projections'):
            P_ani = []
            for b, U_ii in zip(a_a, U_aii):
                P_ni = np.dot(kpt.P_ani[b][na:nb], U_ii)
                if time_reversal:
                    P_ni = P_ni.conj()
                P_ani.append(P_ni)

        return KPoint(s, K, n1, n2, blocksize, na, nb,
                      ut_nR, eps_n, f_n, P_ani, shift_c)

    @timer('Get kpoint pair')
    def get_kpoint_pair(self, pd, s, Kork_c, n1, n2, m1, m2,
                        load_wfs=True, block=False):
        assert m1 <= m2
        assert n1 <= n2

        if isinstance(Kork_c, int):
            # If k_c is an integer then it refers to
            # the index of the kpoint in the BZ
            k_c = self.gs.kd.bzk_kc[Kork_c]
        else:
            k_c = Kork_c

        q_c = pd.kd.bzk_kc[0]
        with self.timer('get k-points'):
            kpt1 = self.get_k_point(s, k_c, n1, n2, load_wfs=load_wfs)
            # K2 = wfs.kd.find_k_plus_q(q_c, [kpt1.K])[0]
            kpt2 = self.get_k_point(s, k_c + q_c, m1, m2,
                                    load_wfs=load_wfs, block=block)

        with self.timer('fft indices'):
            Q_G = self.get_fft_indices(kpt1.K, kpt2.K, q_c, pd,
                                       kpt1.shift_c - kpt2.shift_c)

        return KPointPair(kpt1, kpt2, Q_G)

    @timer('get_pair_density')
    def get_pair_density(self, pd, kptpair, n_n, m_m,
                         optical_limit=False, intraband=False,
                         Q_aGii=None, block=False, direction=2,
                         extend_head=True):
        """Get pair density for a kpoint pair."""
        ol = optical_limit = np.allclose(pd.kd.bzk_kc[0], 0.0)
        eh = extend_head
        cpd = self.calculate_pair_densities  # General pair densities
        opd = self.optical_pair_density  # Interband pair densities / q

        if Q_aGii is None:
            Q_aGii = self.initialize_paw_corrections(pd)

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2
        Q_G = kptpair.Q_G  # Fourier components of kpoint pair
        nG = len(Q_G)

        if extend_head:
            n_nmG = np.zeros((len(n_n), len(m_m), nG + 2 * ol), pd.dtype)
        else:
            n_nmG = np.zeros((len(n_n), len(m_m), nG), pd.dtype)

        for j, n in enumerate(n_n):
            Q_G = kptpair.Q_G
            with self.timer('conj'):
                ut1cc_R = kpt1.ut_nR[n - kpt1.na].conj()
            with self.timer('paw'):
                C1_aGi = [np.dot(Q_Gii, P1_ni[n - kpt1.na].conj())
                          for Q_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
                n_nmG[j, :, 2 * ol * eh:] = cpd(ut1cc_R, C1_aGi, kpt2, pd, Q_G,
                                                block=block)
            if optical_limit:
                if extend_head:
                    n_nmG[j, :, 0:3] = opd(n, m_m, kpt1, kpt2,
                                           block=block)
                else:
                    n_nmG[j, :, 0] = opd(n, m_m, kpt1, kpt2,
                                         block=block)[:, direction]
        return n_nmG

    @timer('get_pair_momentum')
    def get_pair_momentum(self, pd, kptpair, n_n, m_m, Q_avGii=None):
        r"""Calculate matrix elements of the momentum operator.

        Calculates::

          n_{nm\mathrm{k}}\int_{\Omega_{\mathrm{cell}}}\mathrm{d}\mathbf{r}
          \psi_{n\mathrm{k}}^*(\mathbf{r})
          e^{-i\,(\mathrm{q} + \mathrm{G})\cdot\mathbf{r}}
          \nabla\psi_{m\mathrm{k} + \mathrm{q}}(\mathbf{r})

        pd: PlaneWaveDescriptor
            Plane wave descriptor of a single q_c.
        kptpair: KPointPair
            KpointPair object containing the two kpoints.
        n_n: list
            List of left-band indices (n).
        m_m:
            List of right-band indices (m).
        """
        gs = self.gs

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2
        Q_G = kptpair.Q_G  # Fourier components of kpoint pair

        # For the same band we
        kd = gs.kd
        gd = gs.gd
        k_c = kd.bzk_kc[kpt1.K] + kpt1.shift_c
        k_v = 2 * np.pi * np.dot(k_c, np.linalg.inv(gd.cell_cv).T)

        # Calculate k + G
        G_Gv = pd.get_reciprocal_vectors(add_q=True)
        kqG_Gv = k_v[np.newaxis] + G_Gv

        # Pair velocities
        n_nmvG = pd.zeros((len(n_n), len(m_m), 3))

        # Calculate derivatives of left-wavefunction
        # (there will typically be fewer of these)
        ut_nvR = self.make_derivative(kpt1.s, kpt1.K, kpt1.n1, kpt1.n2)

        # PAW-corrections
        if Q_avGii is None:
            Q_avGii = self.initialize_paw_nabla_corrections(pd)

        # Iterate over occupied bands
        for j, n in enumerate(n_n):
            ut1cc_R = kpt1.ut_nR[n].conj()

            n_mG = self.calculate_pair_densities(ut1cc_R,
                                                 [], kpt2,
                                                 pd, Q_G)

            n_nmvG[j] = 1j * kqG_Gv.T[np.newaxis] * n_mG[:, np.newaxis]

            # Treat each cartesian component at a time
            for v in range(3):
                # Minus from integration by parts
                utvcc_R = -ut_nvR[n, v].conj()
                Cv1_aGi = [np.dot(P1_ni[n].conj(), Q_vGii[v])
                           for Q_vGii, P1_ni in zip(Q_avGii, kpt1.P_ani)]

                nv_mG = self.calculate_pair_densities(utvcc_R,
                                                      Cv1_aGi, kpt2,
                                                      pd, Q_G)

                n_nmvG[j, :, v] += nv_mG

        # We want the momentum operator
        n_nmvG *= -1j

        return n_nmvG

    @timer('Calculate pair-densities')
    def calculate_pair_densities(self, ut1cc_R, C1_aGi, kpt2, pd, Q_G,
                                 block=True):
        """Calculate FFT of pair-densities and add PAW corrections.

        ut1cc_R: 3-d complex ndarray
            Complex conjugate of the periodic part of the left hand side
            wave function.
        C1_aGi: list of ndarrays
            PAW corrections for all atoms.
        kpt2: KPoint object
            Right hand side k-point object.
        pd: PWDescriptor
            Plane-wave descriptor for for q=k2-k1.
        Q_G: 1-d int ndarray
            Mapping from flattened 3-d FFT grid to 0.5(G+q)^2<ecut sphere.
        """

        dv = pd.gd.dv
        n_mG = pd.empty(kpt2.blocksize)
        myblocksize = kpt2.nb - kpt2.na

        for ut_R, n_G in zip(kpt2.ut_nR, n_mG):
            n_R = ut1cc_R * ut_R
            with self.timer('fft'):
                n_G[:] = pd.fft(n_R, 0, Q_G) * dv
        # PAW corrections:
        with self.timer('gemm'):
            for C1_Gi, P2_mi in zip(C1_aGi, kpt2.P_ani):
                # gemm(1.0, C1_Gi, P2_mi, 1.0, n_mG[:myblocksize], 't')
                mmm(1.0, P2_mi, 'N', C1_Gi, 'T', 1.0, n_mG[:myblocksize])

        if not block or self.blockcomm.size == 1:
            return n_mG
        else:
            n_MG = pd.empty(kpt2.blocksize * self.blockcomm.size)
            self.blockcomm.all_gather(n_mG, n_MG)
            return n_MG[:kpt2.n2 - kpt2.n1]

    @timer('Optical limit')
    def optical_pair_velocity(self, n, m_m, kpt1, kpt2, block=False):
        if self.ut_sKnvR is None or kpt1.K not in self.ut_sKnvR[kpt1.s]:
            self.ut_sKnvR = self.calculate_derivatives(kpt1)

        kd = self.gs.kd
        gd = self.gs.gd
        k_c = kd.bzk_kc[kpt1.K] + kpt1.shift_c
        k_v = 2 * np.pi * np.dot(k_c, np.linalg.inv(gd.cell_cv).T)

        ut_vR = self.ut_sKnvR[kpt1.s][kpt1.K][n - kpt1.n1]
        atomdata_a = self.gs.setups
        C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[n - kpt1.na])
                 for atomdata, P_ni in zip(atomdata_a, kpt1.P_ani)]

        blockbands = kpt2.nb - kpt2.na
        n0_mv = np.empty((kpt2.blocksize, 3), dtype=complex)
        nt_m = np.empty(kpt2.blocksize, dtype=complex)
        n0_mv[:blockbands] = -self.gs.gd.integrate(ut_vR,
                                                   kpt2.ut_nR).T
        nt_m[:blockbands] = self.gs.gd.integrate(kpt1.ut_nR[n - kpt1.na],
                                                 kpt2.ut_nR)

        n0_mv[:blockbands] += (1j * nt_m[:blockbands, np.newaxis] *
                               k_v[np.newaxis, :])

        for C_vi, P_mi in zip(C_avi, kpt2.P_ani):
            # gemm(1.0, C_vi, P_mi, 1.0, n0_mv[:blockbands], 'c')
            mmm(1.0, P_mi, 'N', C_vi, 'C', 1.0, n0_mv[:blockbands])

        if block and self.blockcomm.size > 1:
            n0_Mv = np.empty((kpt2.blocksize * self.blockcomm.size, 3),
                             dtype=complex)
            self.blockcomm.all_gather(n0_mv, n0_Mv)
            n0_mv = n0_Mv[:kpt2.n2 - kpt2.n1]

        return -1j * n0_mv

    def optical_pair_density(self, n, m_m, kpt1, kpt2,
                             block=False):
        # Relative threshold for perturbation theory
        threshold = self.threshold

        eps1 = kpt1.eps_n[n - kpt1.n1]
        deps_m = (eps1 - kpt2.eps_n)[m_m - kpt2.n1]
        n0_mv = self.optical_pair_velocity(n, m_m, kpt1, kpt2,
                                           block=block)

        deps_m = deps_m.copy()
        deps_m[deps_m == 0.0] = np.inf

        smallness_mv = np.abs(-1e-3 * n0_mv / deps_m[:, np.newaxis])
        inds_mv = (np.logical_and(np.inf > smallness_mv,
                                  smallness_mv > threshold))
        n0_mv *= - 1 / deps_m[:, np.newaxis]
        n0_mv[inds_mv] = 0

        return n0_mv

    @timer('Intraband')
    def intraband_pair_density(self, kpt, n_n=None,
                               only_partially_occupied=False):
        """Calculate intraband matrix elements of nabla"""
        # Bands and check for block parallelization
        na, nb, n1 = kpt.na, kpt.nb, kpt.n1
        vel_nv = np.zeros((nb - na, 3), dtype=complex)
        if n_n is None:
            n_n = np.arange(na, nb)
        assert np.max(n_n) < nb, 'This is too many bands'
        assert np.min(n_n) >= na, 'This is too few bands'

        # Load kpoints
        kd = self.gs.kd
        gd = self.gs.gd
        k_c = kd.bzk_kc[kpt.K] + kpt.shift_c
        k_v = 2 * np.pi * np.dot(k_c, np.linalg.inv(gd.cell_cv).T)
        atomdata_a = self.gs.setups
        f_n = kpt.f_n

        width = self.gs.get_occupations_width()

        if width > 1e-15:
            dfde_n = -1 / width * (f_n - f_n**2.0)  # Analytical derivative
            partocc_n = np.abs(dfde_n) > 1e-5  # Is part. occupied?
        else:
            # Just include all bands to be sure
            partocc_n = np.ones(len(f_n), dtype=bool)

        if only_partially_occupied and not partocc_n.any():
            return None

        if only_partially_occupied:
            # Check for block par. consistency
            assert (partocc_n < nb).all(), \
                print('Include more unoccupied bands ', +
                      'or less block parr.', file=self.fd)

        # Break bands into degenerate chunks
        degchunks_cn = []  # indexing c as chunk number
        for n in n_n:
            inds_n = np.nonzero(np.abs(kpt.eps_n[n - n1] -
                                       kpt.eps_n) < 1e-5)[0] + n1

            # Has this chunk already been computed?
            oldchunk = any([n in chunk for chunk in degchunks_cn])
            if not oldchunk and \
               (partocc_n[n - n1] or not only_partially_occupied):
                assert all([ind in n_n for ind in inds_n]), \
                    print('\nYou are cutting over a degenerate band ' +
                          'using block parallelization.',
                          inds_n, n_n, file=self.fd)
                degchunks_cn.append((inds_n))

        # Calculate matrix elements by diagonalizing each block
        for ind_n in degchunks_cn:
            deg = len(ind_n)
            ut_nvR = self.gs.gd.zeros((deg, 3), complex)
            vel_nnv = np.zeros((deg, deg, 3), dtype=complex)
            # States are included starting from kpt.na
            ut_nR = kpt.ut_nR[ind_n - na]

            # Get derivatives
            for ind, ut_vR in zip(ind_n, ut_nvR):
                ut_vR[:] = self.make_derivative(kpt.s, kpt.K,
                                                ind, ind + 1)[0]

            # Treat the whole degenerate chunk
            for n in range(deg):
                ut_vR = ut_nvR[n]
                C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[ind_n[n] - na])
                         for atomdata, P_ni in zip(atomdata_a, kpt.P_ani)]

                nabla0_nv = -self.gs.gd.integrate(ut_vR, ut_nR).T
                nt_n = self.gs.gd.integrate(ut_nR[n], ut_nR)
                nabla0_nv += 1j * nt_n[:, np.newaxis] * k_v[np.newaxis, :]

                for C_vi, P_ni in zip(C_avi, kpt.P_ani):
                    # gemm(1.0, C_vi, P_ni[ind_n - na], 1.0, nabla0_nv, 'c')
                    mmm(1.0, P_ni[ind_n - na], 'N', C_vi, 'C', 1.0, nabla0_nv)

                vel_nnv[n] = -1j * nabla0_nv

            for iv in range(3):
                vel, _ = np.linalg.eig(vel_nnv[..., iv])
                vel_nv[ind_n - na, iv] = vel  # Use eigenvalues

        return vel_nv[n_n - na]

    def get_fft_indices(self, K1, K2, q_c, pd, shift0_c):
        """Get indices for G-vectors inside cutoff sphere."""
        kd = self.gs.kd
        N_G = pd.Q_qG[0]
        shift_c = (shift0_c +
                   (q_c - kd.bzk_kc[K2] + kd.bzk_kc[K1]).round().astype(int))
        if shift_c.any():
            n_cG = np.unravel_index(N_G, pd.gd.N_c)
            n_cG = [n_G + shift for n_G, shift in zip(n_cG, shift_c)]
            N_G = np.ravel_multi_index(n_cG, pd.gd.N_c, 'wrap')
        return N_G

    def construct_symmetry_operators(self, K, k_c=None):
        from gpaw.response.symmetry_ops import construct_symmetry_operators
        return construct_symmetry_operators(
            self.gs, K, k_c, apply_strange_shift=False)

    @timer('Initialize PAW corrections')
    def initialize_paw_corrections(self, pd, soft=False):
        from gpaw.response.paw import calculate_paw_corrections
        return calculate_paw_corrections(
            setups=self.gs.setups, pd=pd, soft=soft,
            spos_ac=self.spos_ac)

    @timer('Initialize PAW corrections')
    def initialize_paw_nabla_corrections(self, pd, soft=False):
        print('Initializing nabla PAW Corrections', file=self.fd)
        from gpaw.response.paw import calculate_paw_nabla_corrections
        return calculate_paw_nabla_corrections(
            setups=self.gs.setups, pd=pd, soft=soft,
            spos_ac=self.spos_ac)

    def calculate_derivatives(self, kpt):
        ut_sKnvR = [{}, {}]
        ut_nvR = self.make_derivative(kpt.s, kpt.K, kpt.n1, kpt.n2)
        ut_sKnvR[kpt.s][kpt.K] = ut_nvR

        return ut_sKnvR

    @timer('Derivatives')
    def make_derivative(self, s, K, n1, n2):
        gs = self.gs
        if self.real_space_derivatives:
            grad_v = [Gradient(gs.gd, v, 1.0, 4, complex).apply
                      for v in range(3)]

        U_cc, T, a_a, U_aii, shift_c, time_reversal = \
            self.construct_symmetry_operators(K)
        A_cv = gs.gd.cell_cv
        M_vv = np.dot(np.dot(A_cv.T, U_cc.T), np.linalg.inv(A_cv).T)
        ik = gs.kd.bz2ibz_k[K]
        assert gs.kd.comm.size == 1
        kpt = gs.kpt_qs[ik][s]
        psit_nG = kpt.psit_nG
        iG_Gv = 1j * gs.pd.get_reciprocal_vectors(q=ik, add_q=False)
        ut_nvR = gs.gd.zeros((n2 - n1, 3), complex)
        for n in range(n1, n2):
            for v in range(3):
                if self.real_space_derivatives:
                    ut_R = T(gs.pd.ifft(psit_nG[n], ik))
                    grad_v[v](ut_R, ut_nvR[n - n1, v],
                              np.ones((3, 2), complex))
                else:
                    ut_R = T(gs.pd.ifft(iG_Gv[:, v] * psit_nG[n], ik))
                    for v2 in range(3):
                        ut_nvR[n - n1, v2] += ut_R * M_vv[v, v2]

        return ut_nvR

    def __del__(self):
        self.context.close()


def normalize_args(calc, txt, world, timer):
    context = new_context(txt, world, timer)
    with context.timer('Read ground state'):
        if isinstance(calc, (str, Path)):
            print('Reading ground state calculation:\n  %s' % calc,
                  file=context.fd)
            with disable_dry_run():
                calc = GPAW(calc, communicator=mpi.serial_comm)
        else:
            assert calc.wfs.world.size == 1

    return calc, context


class PairDensity(NoCalculatorPairDensity):
    def __init__(self, gs, *,
                 world=mpi.world, txt='-', timer=None,
                 **kwargs):
        """Density matrix elements

        Parameters
        ----------
        ftol : float
            Threshold determining whether a band is completely filled
            (f > 1 - ftol) or completely empty (f < ftol).
        threshold : float
            Numerical threshold for the optical limit k dot p perturbation
            theory expansion.
        real_space_derivatives : bool
            Calculate nabla matrix elements (in the optical limit)
            using a real space finite difference approximation.
        """

        # note: gs is just called gs for historical reasons.
        # It's actually calc-or-filename union.

        self.calc, context = normalize_args(gs, txt, world, timer)

        super().__init__(
            gs=self.calc.gs_adapter(),
            context=context,
            **kwargs)
