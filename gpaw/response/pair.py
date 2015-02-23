from __future__ import print_function
import sys
from math import pi

import numpy as np
from ase.units import Hartree
from ase.utils import devnull
from ase.utils.timing import timer, Timer

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.fd_operators import Gradient
from gpaw.occupations import FermiDirac
from gpaw.response.math_func import two_phi_planewave_integrals
from gpaw.utilities.blas import gemm
from gpaw.utilities.progressbar import ProgressBar
from gpaw.wavefunctions.pw import PWLFC
import gpaw.io.tar as io

import warnings


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

class KPointPair:
    def __init__(self, kpt1, kpt2, Q_G):
        self.kpt1 = kpt1
        self.kpt2 = kpt2
        self.Q_G = Q_G

    def get_k1(self):
        return self.kpt1
    
    def get_k2(self):
        return self.kpt2

    def get_planewave_indices(self):
        return self.Q_G

class PWSymmetryAnalyzer:
    def __init__(self, kd, pd, disable_symmetries=False):
        self.pd = pd
        self.kd = kd
        self.disable_symmetries = disable_symmetries
        op_scc = kd.symmetry.op_scc
        self.nsym = len(op_scc)
        
        # Initialize
        self.initialize()

    def initialize(self):
        self.determine_allowed_symmetries()

    def determine_allowed_symmetries(self):
        pd = self.pd

        # Shortcuts
        B_cv = 2.0 * np.pi * pd.gd.icell_cv
        
        q_c = pd.kd.bzk_kc[0]
        G_Gv = pd.get_reciprocal_vectors()
        G_Gc = np.dot(G_Gv, np.linalg.inv(B_cv))
        kd = self.kd

        op_scc = kd.symmetry.op_scc
        time_reversal = kd.symmetry.time_reversal and \
                        not kd.symmetry.has_inversion

        nsym = len(op_scc)
        nsymtot = nsym * (1 + time_reversal)
        newq_sc = np.dot(op_scc, q_c)
        
        shift_sc = np.zeros((nsymtot, 3), int)
        conserveq_s = np.zeros(nsymtot, bool)

        # Direct
        dshift_sc = (newq_sc - q_c[np.newaxis]).round().astype(int)        
        inds_s = np.argwhere((newq_sc == q_c[np.newaxis] + dshift_sc).all(1))
        conserveq_s[inds_s] = True

        shift_sc[:nsym] = dshift_sc

        # Time reversal and Umklapp
        if time_reversal:
            trshift_sc = (-newq_sc - q_c[np.newaxis]).round().astype(int)
            trinds_s = np.argwhere((-newq_sc == q_c[np.newaxis] + trshift_sc).all(1)) + nsym
            conserveq_s[trinds_s] = True
            shift_sc[nsym:nsymtot] = trshift_sc

        s_s = conserveq_s.nonzero()[0]
        t_sc = [] # For later inclusion of nonsymmorphic symmetries
        
        if self.disable_symmetries:
            for s, op_cc in enumerate(op_scc):
                if (op_cc == np.eye(3)).all():
                    s_s = [s]
                    break

        infostring = 'Found {0} allowed symmetries. '.format(len(s_s))
        if time_reversal: 
            infostring += 'Time reversal included. '
        else:
            infostring += 'Time reversal not included. '

        print(infostring)
            
        self.s_s = s_s
        self.shift_sc = shift_sc
        self.t_sc = t_sc

    def group_kpoints(self, K_k):
        """Group kpoints according to the reduced symmetries"""
        s_s = self.s_s
        bz2bz_ks = self.kd.bz2bz_ks
        nk = len(bz2bz_ks)
        sbz2sbz_ks = bz2bz_ks[K_k][:, s_s]  # Reduced number of symmetries
        # Avoid -1 (see documentation in gpaw.symmetry)
        sbz2sbz_ks[sbz2sbz_ks == -1] = nk
        
        smallestk_k = np.sort(sbz2sbz_ks)[:, 0]
        k2g_g = np.unique(smallestk_k, return_index=True)[1]
        
        K_gs = sbz2sbz_ks[k2g_g]
        K_gk = [np.unique(K_s[K_s != nk]) for K_s in K_gs]

        return K_gk

    def get_kpoint_mapping(self, K1, K2):
        """Get allowed symmetry for mapping between K1 and K2"""
        s_s = self.s_s
        bz2bz_ks = self.kd.bz2bz_ks        
        bzk2rbz_s = bz2bz_ks[K1][:, s_s]
        try:
            s = np.argwhere(bzk2rbz_s == K2)[0][0]
        except IndexError:
            print('K = {0} cannot be mapped into K = {1}'.format(K1, K2))
            raise
        return s_s[s]

    def get_shift(self, K1, K2, U_cc):
        kd = self.kd
        k1_c = kd.bzk_kc[K1]
        k2_c = kd.bzk_kc[K2]
        
        shift_c = np.dot(U_cc, k1_c) - k2_c
        assert np.allclose(shift_c.round(), shift_c)
        shift_c = shift_c.round().astype(int)
        
        return shift_c

    def create_Q_mapping(self):
        Q2Q_sG = []
        for s in s_s:
            UQ_G = self.map_Q_indices(s)
            Q2Q_sG.append(UQ_G)
        
        self.Q2Q_sG = Q2Q_sG

    def map_G(self, K1, K2, a_MG):
        G_G, sign = self.map_G_vectors(K1, K2)
        if sign == -1:
            return a_MG[..., G_G].conj()
        else:
            return a_MG[..., G_G]

    def map_v(self, K1, K2, a_Mv, shift_c=True):
        A_cv = self.pd.gd.cell_cv
        iA_cv = self.pd.gd.icell_cv

        # Get symmetry
        s = self.get_kpoint_mapping(K1, K2)
        U_cc, sign, _ = self.get_symmetry_operator(s)

        # Get potential shift
        shift_c = self.get_shift(K1, K2, U_cc)
        shift_v = 2 * np.pi * np.dot(iA_cv.T, shift_c)

        # Create cartesian operator
        M_vv = sign * np.dot(np.dot(A_cv.T, U_cc.T), iA_cv)
        
        if sign == -1:
            return np.dot(a_Mv.conj(), M_vv) - 1j * shift_v
        else:
            return np.dot(a_Mv, M_vv) - 1j * shift_v

    def timereversal(self, s):
        tr = bool(s // self.nsym)
        assert not tr
        return tr

    def get_symmetry_operator(self, s):
        op_scc = self.kd.symmetry.op_scc
        if self.timereversal(s):
            sign = -1
        else:
            sign = 1

        return op_scc[s % self.nsym], sign, self.shift_sc[s]

    def map_Q_indices(self, K1, K2):
        pd = self.pd
        s = self.get_kpoint_mapping(K1, K2)
        op_cc, sign, shift_c = self.get_symmetry_operator(s)
        B_cv = 2.0 * np.pi * pd.gd.icell_cv
        G_Gv = pd.get_reciprocal_vectors()
        G_Gc = np.dot(G_Gv, np.linalg.inv(B_cv))
        UG_Gc = np.dot(G_Gc - shift_c, 
                       sign * np.linalg.inv(op_cc).T)
        assert np.allclose(UG_Gc.round(), UG_Gc)
        UQ_G = np.ravel_multi_index(UG_Gc.round().astype(int).T,
                                    pd.gd.N_c, 'wrap')

#        print(s, self.nsym, op_cc, sign, shift_c)
#        print(pd.Q_qG[0][0:10])
#        print(UQ_G[0:10])
#        print(G_Gv[0:10])

        return UQ_G, sign

    def map_G_vectors(self, K1, K2):
        pd = self.pd
        UQ_G, sign = self.map_Q_indices(K1, K2)
        Q_G = pd.Q_qG[0]
        G_G = len(Q_G) * [None]
        for G, UQ in enumerate(UQ_G):
            G_G[G] = np.argwhere(Q_G == UQ)[0][0]

#        print(G_G)
        return G_G, sign

    def unfold_ibz_kpoint(self, ik):
        kd = self.kd
        K_k = np.unique(kd.bz2bz_ks[kd.ibz2bz_k[ik]])
        K_k = K_k[K_k != -1]
        return K_k


class PairDensity:
    def __init__(self, calc, ecut=50,
                 ftol=1e-6, threshold=1,
                 real_space_derivatives=False,
                 world=mpi.world, txt=sys.stdout, timer=None, nblocks=1,
                 gate_voltage=None):
        if ecut is not None:
            ecut /= Hartree

        if gate_voltage is not None:
            gate_voltage /= Hartree

        self.ecut = ecut
        self.ftol = ftol
        self.threshold = threshold
        self.real_space_derivatives = real_space_derivatives
        self.world = world
        self.gate_voltage = gate_voltage

        if nblocks == 1:
            self.blockcomm = self.world.new_communicator([world.rank])
            self.kncomm = world
        else:
            assert world.size % nblocks == 0, world.size
            rank1 = world.rank // nblocks * nblocks
            rank2 = rank1 + nblocks
            self.blockcomm = self.world.new_communicator(range(rank1, rank2))
            ranks = range(world.rank % nblocks, world.size, nblocks)
            self.kncomm = self.world.new_communicator(ranks)

        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt

        self.timer = timer or Timer()

        if isinstance(calc, str):
            print('Reading ground state calculation:\n  %s' % calc,
                  file=self.fd)
            if not calc.split('.')[-1] == 'gpw':
                calc = calc + '.gpw'
            self.reader = io.Reader(calc, comm=mpi.serial_comm)
            calc = GPAW(calc, txt=None, communicator=mpi.serial_comm,
                        read_projections=False)
        else:
            self.reader = None
            assert calc.wfs.world.size == 1

        assert calc.wfs.kd.symmetry.symmorphic
        self.calc = calc

        if gate_voltage is not None:
            self.add_gate_voltage(gate_voltage)

        self.spos_ac = calc.atoms.get_scaled_positions()

        self.nocc1 = None  # number of completely filled bands
        self.nocc2 = None  # number of non-empty bands
        self.count_occupied_bands()

        self.vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))

        self.ut_sKnvR = None  # gradient of wave functions for optical limit

        print('Number of blocks:', nblocks, file=self.fd)

    def add_gate_voltage(self, gate_voltage=0):
        """Shifts the Fermi-level by e * Vg. By definition e = 1."""
        assert isinstance(self.calc.occupations, FermiDirac)
        print('Shifting Fermi-level by %.2f eV' % (gate_voltage * Hartree),
              file=self.fd)

        for kpt in self.calc.wfs.kpt_u:
            kpt.f_n = (self.shift_occupations(kpt.eps_n, gate_voltage)
                       * kpt.weight)

    def shift_occupations(self, eps_n, gate_voltage):
        """Shift fermilevel."""
        fermi = self.calc.occupations.get_fermi_level() + gate_voltage
        width = self.calc.occupations.width
        tmp = (eps_n - fermi) / width
        f_n = np.zeros_like(eps_n)
        f_n[tmp <= 100] = 1 / (1 + np.exp(tmp[tmp <= 100]))
        f_n[tmp > 100] = 0.0
        return f_n

    def count_occupied_bands(self):
        self.nocc1 = 9999999
        self.nocc2 = 0
        for kpt in self.calc.wfs.kpt_u:
            f_n = kpt.f_n / kpt.weight
            self.nocc1 = min((f_n > 1 - self.ftol).sum(), self.nocc1)
            self.nocc2 = max((f_n > self.ftol).sum(), self.nocc2)
        print('Number of completely filled bands:', self.nocc1, file=self.fd)
        print('Number of partially filled bands:', self.nocc2, file=self.fd)
        print('Total number of bands:', self.calc.wfs.bd.nbands,
              file=self.fd)

    def distribute_k_points_and_bands(self, band1, band2, kpts=None):
        """Distribute spins, k-points and bands.

        nbands: int
            Number of bands for each spin/k-point combination.

        The attribute self.mysKn1n2 will be set to a list of (s, K, n1, n2)
        tuples that this process handles.
        """

        wfs = self.calc.wfs

        if kpts is None:
            kpts = range(wfs.kd.nbzkpts)

        nbands = band2 - band1
        size = self.kncomm.size
        rank = self.kncomm.rank
        ns = wfs.nspins
        nk = len(kpts)
        n = (ns * nk * nbands + size - 1) // size
        i1 = rank * n
        i2 = min(i1 + n, ns * nk * nbands)

        self.mysKn1n2 = []
        i = 0
        for s in range(ns):
            for K in kpts:
                n1 = min(max(0, i1 - i), nbands)
                n2 = min(max(0, i2 - i), nbands)
                if n1 != n2:
                    self.mysKn1n2.append((s, K, n1 + band1, n2 + band1))
                i += nbands

        print('BZ k-points:', self.calc.wfs.kd.description, file=self.fd)
        print('Distributing spins, k-points and bands (%d x %d x %d)' %
              (ns, nk, nbands),
              'over %d process%s' %
              (self.kncomm.size, ['es', ''][self.kncomm.size == 1]),
              file=self.fd)
        print('Number of blocks:', self.blockcomm.size, file=self.fd)

    @timer('Get a k-point')
    def get_k_point(self, s, K, n1, n2, block=False):
        """Return wave functions for a specific k-point and spin.

        s: int
            Spin index (0 or 1).
        K: int
            BZ k-point index.
        n1, n2: int
            Range of bands to include.
        """

        wfs = self.calc.wfs

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
            self.construct_symmetry_operators(K)
        ik = wfs.kd.bz2ibz_k[K]
        kpt = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]

        eps_n = kpt.eps_n[n1:n2]
        f_n = kpt.f_n[n1:n2] / kpt.weight

        psit_nG = kpt.psit_nG
        ut_nR = wfs.gd.empty(nb - na, wfs.dtype)
        for n in range(na, nb):
            ut_nR[n - na] = T(wfs.pd.ifft(psit_nG[n], ik))

        P_ani = []
        if self.reader is None:
            for b, U_ii in zip(a_a, U_aii):
                P_ni = np.dot(kpt.P_ani[b][na:nb], U_ii)
                if time_reversal:
                    P_ni = P_ni.conj()
                P_ani.append(P_ni)
        else:
            II_a = []
            I1 = 0
            for U_ii in U_aii:
                I2 = I1 + len(U_ii)
                II_a.append((I1, I2))
                I1 = I2

            P_ani = []
            P_nI = self.reader.get('Projections', kpt.s, kpt.k)
            for b, U_ii in zip(a_a, U_aii):
                I1, I2 = II_a[b]
                P_ni = np.dot(P_nI[na:nb, I1:I2], U_ii)
                if time_reversal:
                    P_ni = P_ni.conj()
                P_ani.append(P_ni)
            
        return KPoint(s, K, n1, n2, blocksize, na, nb,
                      ut_nR, eps_n, f_n, P_ani, shift_c)

    def generate_pair_densities(self, pd, m1, m2, intraband=True,
                                disable_symmetries=False, disable_optical_limit=False):
        """Generator for returning pair densities. """

        print('Initializing PAW Corrections', file=self.fd)
        self.Q_aGii = self.initialize_paw_corrections(pd)

        q_c = pd.kd.bzk_kc[0]
        optical_limit = not disable_optical_limit and np.allclose(q_c, 0.0)

        PWSA = PWSymmetryAnalyzer(self.calc.wfs.kd, pd, 
                                  disable_symmetries=disable_symmetries)
        pb = ProgressBar(self.fd)
        for kn, (s, ik, n1, n2) in pb.enumerate(self.mysKn1n2):
            Kstar_k = PWSA.unfold_ibz_kpoint(ik)
            for K_k in PWSA.group_kpoints(Kstar_k):
                K1 = K_k[0]
                kptpair = self.get_kpoint_pair(pd, s, K1, n1, n2, m1, m2)
                kpt1 = kptpair.get_k1()
                kpt2 = kptpair.get_k2()

                # Use kpt2 to compute intraband transitions
                # These conditions are sufficent to make sure
                # that it still works in parallel
                if kpt1.n1 == 0 and self.blockcomm.rank == 0 and \
                   optical_limit and self.intraband:
                    assert self.nocc2 <= kpt2.nb, \
                        print('Error: Too few unoccupied bands')
                    vel0_mv = self.intraband_pair_density(kpt2)
                    if len(vel0_mv):
                        for K2 in K_k:
                            vel_mv = PWSA.map_v(K1, K2, np.array(vel0_mv))
                            yield ([], [], kpt2.f_n, [], [],
                                   [], [], vel_mv)

                for n in range(n2 - n1):
                    # In case omegamax is set by user
                    eps1 = kpt1.eps_n[n]
                    if self.omegamax is not None:
                        m = [m for m, d in enumerate(eps1 - kpt2.eps_n)
                             if abs(d) <= self.omegamax]
                    else:
                        m = range(0, kpt2.n2 - kpt2.n1)

                    if not len(m):
                        continue

                    deps_m = (eps1 - kpt2.eps_n)[m]
                    f1 = kpt1.f_n[n]
                    df_m = (f1 - kpt2.f_n)[m]
                    df_m[df_m <= 1e-20] = 0.0
                    n0_nmG, n0_nmv, _ = self.get_pair_density(pd, kptpair,
                                                              [n], m,
                                                              intraband=False)
                    
                    for K2 in K_k:
                        n_nmG = PWSA.map_G(K1, K2, n0_nmG)
                        n_nmv = PWSA.map_v(K1, K2, n0_nmv)
                        n_nmG[0][:, 0] = n_nmv[0][:, 0]
                        kptpair2 = self.get_kpoint_pair(pd, s, K2, 
                                                        n1, n2, m1, m2)
                        n1_nmG, n1_nmv, _ = self.get_pair_density(pd, kptpair2,
                                                                  [n], m,
                                                                  intraband=False)
                        n1_nmG[0][:, 0] = n1_nmv[0][:, 0]


                        if (abs(n_nmG[0] - n1_nmG[0]) > 1e-10).any():
                            print(K1)
                            print(K2, n1_nmG[0][0, 0])
                            print(K2, n_nmG[0][0, 0])

                        if not optical_limit:
                            yield (n, m, [], df_m, deps_m, n_nmG[0], [], [])
                        else:
                            yield (n, m, [], df_m, deps_m, n_nmG[0], 
                                   n_nmv[0], [])
        pb.finish()

    @timer('Get kpoint pair')
    def get_kpoint_pair(self, pd, s, K, n1, n2, m1, m2):
        wfs = self.calc.wfs
        q_c = pd.kd.bzk_kc[0]
        with self.timer('get k-points'):
            kpt1 = self.get_k_point(s, K, n1, n2)
            K2 = wfs.kd.find_k_plus_q(q_c, [K])[0]
            kpt2 = self.get_k_point(s, K2, m1, m2, block=True)

        with self.timer('fft indices'):
            Q_G = self.get_fft_indices(kpt1.K, kpt2.K, q_c, pd, 
                                       kpt1.shift_c - kpt2.shift_c)

        return KPointPair(kpt1, kpt2, Q_G)

    @timer('get_pair_density')
    def get_pair_density(self, pd, kptpair, n_n, m_m, intraband=False):
        q_c = pd.kd.bzk_kc[0]
        optical_limit = not self.no_optical_limit and np.allclose(q_c, 0.0)

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2
        Q_G = kptpair.Q_G # Fourier components of kpoint pair

        n_nmG = pd.empty((len(n_n), len(m_m)))
        if optical_limit:
            n_nmv = np.empty((len(n_n), len(m_m), 3), pd.dtype)
        else:
            n_nmv = []

        for j, n in enumerate(n_n):
            eps1 = kpt1.eps_n[n]
            deps_m = (eps1 - kpt2.eps_n)[m_m]
            Q_G = kptpair.Q_G
            with self.timer('conj'):
                ut1cc_R = kpt1.ut_nR[n].conj()
            with self.timer('paw'):
                C1_aGi = [np.dot(Q_Gii, P1_ni[n].conj())
                          for Q_Gii, P1_ni in zip(self.Q_aGii, kpt1.P_ani)]
                n_mG = self.calculate_pair_densities(ut1cc_R,
                                                     C1_aGi, kpt2,
                                                     pd, Q_G)[m_m]
            
            if optical_limit:
                n_nmv[j] = self.optical_pair_density(n, m_m, kpt1, kpt2, 
                                                     deps_m)
                
            n_nmG[j] = n_mG

        if intraband:
            vel_mv = self.intraband_pair_density(kpt2)
        else:
            vel_mv = []

        return n_nmG, n_nmv, vel_mv
        
    @timer('Calculate pair-densities')
    def calculate_pair_densities(self, ut1cc_R, C1_aGi, kpt2, pd, Q_G):
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
                gemm(1.0, C1_Gi, P2_mi, 1.0, n_mG[:myblocksize], 't')

        if self.blockcomm.size == 1:
            return n_mG
        else:
            n_MG = pd.empty(kpt2.blocksize * self.blockcomm.size)
            self.blockcomm.all_gather(n_mG, n_MG)
            return n_MG[:kpt2.n2 - kpt2.n1]

    @timer('Optical limit')
    def optical_pair_density(self, n, m, kpt1, kpt2, deps_m):
        if self.ut_sKnvR is None or kpt1.K not in self.ut_sKnvR[kpt1.s]:
            self.ut_sKnvR = self.calculate_derivatives(kpt1)

        # Relative threshold for perturbation theory
        threshold = self.threshold

        kd = self.calc.wfs.kd
        gd = self.calc.wfs.gd
        k_c = kd.bzk_kc[kpt1.K] + kpt1.shift_c
        k_v = 2 * np.pi * np.dot(k_c, np.linalg.inv(gd.cell_cv).T)

        ut_vR = self.ut_sKnvR[kpt1.s][kpt1.K][n]
        atomdata_a = self.calc.wfs.setups
        C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[n])
                 for atomdata, P_ni in zip(atomdata_a, kpt1.P_ani)]

        blockbands = kpt2.nb - kpt2.na
        n0_mv = np.empty((kpt2.blocksize, 3), dtype=complex)
        nt_m = np.empty(kpt2.blocksize, dtype=complex)
        n0_mv[:blockbands] = -self.calc.wfs.gd.integrate(ut_vR, kpt2.ut_nR).T
        nt_m[:blockbands] = self.calc.wfs.gd.integrate(kpt1.ut_nR[n],
                                                       kpt2.ut_nR)
        n0_mv += 1j * nt_m[:, np.newaxis] * k_v[np.newaxis, :]

        for C_vi, P_mi in zip(C_avi, kpt2.P_ani):
            P_mi = P_mi.copy()
            gemm(1.0, C_vi, P_mi, 1.0, n0_mv[:blockbands], 'c')

        if self.blockcomm.size != 1:
            n0_Mv = np.empty((kpt2.blocksize * self.blockcomm.size, 3),
                             dtype=complex)
            self.blockcomm.all_gather(n0_mv, n0_Mv)
            n0_mv = n0_Mv[:kpt2.n2 - kpt2.n1]

        # In case not all unoccupied bands are included
        if n0_mv.shape[0] != len(m):
            n0_mv = n0_mv[m]

        deps_m = deps_m.copy()
        deps_m[deps_m >= 0.0] = np.inf

        smallness_mv = np.abs(-1e-3 * n0_mv / deps_m[:, np.newaxis])
        inds_mv = (np.logical_and(np.inf > smallness_mv,
                                  smallness_mv > threshold))

        if inds_mv.any():
            indent8 = ' ' * 8
            print('\n    WARNING: Optical limit perturbation' +
                  ' theory failed for:', file=self.fd)
            print(indent8 + 'kpt_c = [%1.2f, %1.2f, %1.2f]'
                  % (k_c[0], k_c[1], k_c[2]), file=self.fd)
            inds_m = inds_mv.any(axis=1)
            depsi_m = deps_m[inds_m]
            n0i_mv = np.abs(n0_mv[inds_m])
            smallness_mv = smallness_mv[inds_m]
            for depsi, n0i_v, smallness_v in zip(depsi_m, n0i_mv,
                                                 smallness_mv):
                print(indent8 + 'Energy eigenvalue difference %1.2e ' % -depsi,
                      file=self.fd)
                print(indent8 + 'Matrix element' +
                      ' %1.2e %1.2e %1.2e' % (n0i_v[0], n0i_v[1], n0i_v[2]),
                      file=self.fd)
                print(indent8 + 'Smallness' +
                      ' %1.2e %1.2e %1.2e\n' % (smallness_v[0],
                                                smallness_v[1],
                                                smallness_v[2]),
                      file=self.fd)

        n0_mv *= 1j / deps_m[:, np.newaxis]
        n0_mv[inds_mv] = 0
#        n_mG[:, 0] = n0_mv[:, 0]

        return n0_mv

    @timer('Intraband')
    def intraband_pair_density(self, kpt, n_n=None):
        vel_nv = np.zeros((kpt.nb - kpt.na, 3), dtype=complex)
        if n_n is None:
            n_n = range(kpt.nb - kpt.na)

        kd = self.calc.wfs.kd
        gd = self.calc.wfs.gd
        k_c = kd.bzk_kc[kpt.K] + kpt.shift_c
        k_v = 2 * np.pi * np.dot(k_c, np.linalg.inv(gd.cell_cv).T)
        atomdata_a = self.calc.wfs.setups
        assert np.max(n_n) < kpt.nb
        f_n = kpt.f_n

        width = self.calc.occupations.width
        if width == 0.0:
            return []

        assert isinstance(self.calc.occupations, FermiDirac)
        dfde_n = - 1. / width * (f_n - f_n**2.0)
        partocc_n = np.abs(dfde_n) > 1e-5
        if not partocc_n.any():
            return []

        # Break bands into degenerate chunks
        degchunks_cn = []  # indexing c as chunk number
        for n in range(kpt.nb - kpt.na):
            inds_n = np.nonzero(np.abs(kpt.eps_n[n] - kpt.eps_n) < 1e-5)[0]
            # Has this chunk already been computed?
            oldchunk = any([n in chunk for chunk in degchunks_cn])
            if not oldchunk and partocc_n[n]:
                degchunks_cn.append((inds_n))

        for ind_n in degchunks_cn:
            deg = len(ind_n)
            ut_nvR = self.calc.wfs.gd.zeros((deg, 3), complex)
            vel_nnv = np.zeros((deg, deg, 3), dtype=complex)
            ut_nR = kpt.ut_nR[ind_n]

            # Get derivatives
            for ind, ut_vR in zip(ind_n, ut_nvR):
                ut_vR[:] = self.make_derivative(kpt.s, kpt.K,
                                                kpt.na + ind,
                                                kpt.na + ind + 1)[0]

            # Treat the whole degenerate chunk
            for n in range(deg):
                ut_vR = ut_nvR[n]
                C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[ind_n[n]])
                         for atomdata, P_ni in zip(atomdata_a, kpt.P_ani)]

                nabla0_nv = -self.calc.wfs.gd.integrate(ut_vR, ut_nR).T
                nt_n = self.calc.wfs.gd.integrate(ut_nR[n], ut_nR)
                nabla0_nv += 1j * nt_n[:, np.newaxis] * k_v[np.newaxis, :]

                for C_vi, P_ni in zip(C_avi, kpt.P_ani):
                    gemm(1.0, C_vi, P_ni[ind_n[0:deg]], 1.0, nabla0_nv, 'c')

                vel_nnv[n] = -1j * nabla0_nv
            
            for iv in range(3):
                vel, _ = np.linalg.eig(vel_nnv[..., iv])
                vel_nv[ind_n, iv] = vel
            
        return vel_nv[n_n]
    
    def get_fft_indices(self, K1, K2, q_c, pd, shift0_c):
        """Get indices for G-vectors inside cutoff sphere."""
        kd = self.calc.wfs.kd
        N_G = pd.Q_qG[0]
        shift_c = (shift0_c +
                   (q_c - kd.bzk_kc[K2] + kd.bzk_kc[K1]).round().astype(int))
        if shift_c.any():
            n_cG = np.unravel_index(N_G, pd.gd.N_c)
            n_cG = [n_G + shift for n_G, shift in zip(n_cG, shift_c)]
            N_G = np.ravel_multi_index(n_cG, pd.gd.N_c, 'wrap')
        return N_G

    def construct_symmetry_operators(self, K):
        """Construct symmetry operators for wave function and PAW projections.

        We want to transform a k-point in the irreducible part of the BZ to
        the corresponding k-point with index K.

        Returns U_cc, T, a_a, U_aii, shift_c and time_reversal, where:

        * U_cc is a rotation matrix.
        * T() is a function that transforms the periodic part of the wave
          function.
        * a_a is a list of symmetry related atom indices
        * U_aii is a list of rotation matrices for the PAW projections
        * shift_c is three integers: see code below.
        * time_reversal is a flag - if True, projections should be complex
          conjugated.

        See the get_k_point() method for how to use these tuples.
        """

        wfs = self.calc.wfs
        kd = wfs.kd

        s = kd.sym_k[K]
        U_cc = kd.symmetry.op_scc[s]
        time_reversal = kd.time_reversal_k[K]
        ik = kd.bz2ibz_k[K]
        k_c = kd.bzk_kc[K]
        ik_c = kd.ibzk_kc[ik]

        sign = 1 - 2 * time_reversal
        shift_c = np.dot(U_cc, ik_c) - k_c * sign
        assert np.allclose(shift_c.round(), shift_c)
        shift_c = shift_c.round().astype(int)

        if (U_cc == np.eye(3)).all():
            T = lambda f_R: f_R
        else:
            N_c = self.calc.wfs.gd.N_c
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')
            T = lambda f_R: f_R.ravel()[i].reshape(N_c)

        if time_reversal:
            T0 = T
            T = lambda f_R: T0(f_R).conj()
            shift_c *= -1

        a_a = []
        U_aii = []
        for a, id in enumerate(self.calc.wfs.setups.id_a):
            b = kd.symmetry.a_sa[s, a]
            S_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            x = np.exp(2j * pi * np.dot(ik_c, S_c))
            U_ii = wfs.setups[a].R_sii[s].T * x
            a_a.append(b)
            U_aii.append(U_ii)

        return U_cc, T, a_a, U_aii, shift_c, time_reversal

    @timer('Initialize PAW corrections')
    def initialize_paw_corrections(self, pd, soft=False):
        wfs = self.calc.wfs
        q_v = pd.K_qv[0]
        optical_limit = np.allclose(q_v, 0)

        G_Gv = pd.get_reciprocal_vectors()
        if optical_limit:
            G_Gv[0] = 1

        pos_av = np.dot(self.spos_ac, pd.gd.cell_cv)

        # Collect integrals for all species:
        Q_xGii = {}
        for id, atomdata in wfs.setups.setups.items():
            if soft:
                ghat = PWLFC([atomdata.ghat_l], pd)
                ghat.set_positions(np.zeros((1, 3)))
                Q_LG = ghat.expand()
                Q_Gii = np.dot(atomdata.Delta_iiL, Q_LG).T
            else:
                Q_Gii = two_phi_planewave_integrals(G_Gv, atomdata)
                ni = atomdata.ni
                Q_Gii.shape = (-1, ni, ni)

            Q_xGii[id] = Q_Gii

        Q_aGii = []
        for a, atomdata in enumerate(wfs.setups):
            id = wfs.setups.id_a[a]
            Q_Gii = Q_xGii[id]
            x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
            Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)
            if optical_limit:
                Q_aGii[a][0] = atomdata.dO_ii

        return Q_aGii


    def calculate_derivatives(self, kpt):
        ut_sKnvR = [{}, {}]
        ut_nvR = self.make_derivative(kpt.s, kpt.K, kpt.n1, kpt.n2)
        ut_sKnvR[kpt.s][kpt.K] = ut_nvR

        return ut_sKnvR

    @timer('Derivatives')
    def make_derivative(self, s, K, n1, n2):
        wfs = self.calc.wfs
        if self.real_space_derivatives:
            grad_v = [Gradient(wfs.gd, v, 1.0, 4, complex).apply
                      for v in range(3)]

        U_cc, T, a_a, U_aii, shift_c, time_reversal = \
            self.construct_symmetry_operators(K)
        A_cv = wfs.gd.cell_cv
        M_vv = np.dot(np.dot(A_cv.T, U_cc.T), np.linalg.inv(A_cv).T)
        ik = wfs.kd.bz2ibz_k[K]
        kpt = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]
        psit_nG = kpt.psit_nG
        iG_Gv = 1j * wfs.pd.get_reciprocal_vectors(q=ik, add_q=False)
        ut_nvR = wfs.gd.zeros((n2 - n1, 3), complex)
        for n in range(n1, n2):
            for v in range(3):
                if self.real_space_derivatives:
                    ut_R = T(wfs.pd.ifft(psit_nG[n], ik))
                    grad_v[v](ut_R, ut_nvR[n - n1, v],
                              np.ones((3, 2), complex))
                else:
                    ut_R = T(wfs.pd.ifft(iG_Gv[:, v] * psit_nG[n], ik))
                    for v2 in range(3):
                        ut_nvR[n - n1, v2] += ut_R * M_vv[v, v2]

        return ut_nvR
