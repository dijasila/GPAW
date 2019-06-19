import numpy as np
from scipy.spatial import cKDTree

from ase.utils import convert_string_to_fd
from ase.utils.timing import Timer, timer

from gpaw import GPAW
import gpaw.mpi as mpi
# gpaw.utilities.blas import gemm
from gpaw.response.math_func import two_phi_planewave_integrals


class KohnShamKPoint:
    """Kohn-Sham orbitals participating in transitions for a given k-point."""
    def __init__(self, K, n_t, s_t, mynt, nt, ta, tb,
                 ut_tR, eps_t, f_t, P_ati, shift_c):
        self.K = K      # BZ k-point index
        self.n_t = n_t  # Band index for each transition
        self.s_t = s_t  # Spin index for each transition
        self.mynt = mynt  # Number of transitions in this block
        self.nt = nt    # Total number of transitions between all blocks
        self.ta = ta    # First transition index of this block
        self.tb = tb    # First transition index of this block not included
        self.ut_tR = ut_tR      # periodic part of wave functions in real-space
        self.eps_t = eps_t      # eigenvalues
        self.f_t = f_t          # occupation numbers
        self.P_ati = P_ati      # PAW projections
        self.shift_c = shift_c  # long story - see the
        # PairDensity.construct_symmetry_operators() method


class KohnShamKPointPairs:
    """Object containing all transitions between Kohn-Sham orbitals from a
    specified k-point to another."""
    def __init__(self, kpt1, kpt2):
        self.kpt1 = kpt1
        self.kpt2 = kpt2

    def get_transition_energies(self):
        """Return the energy differences between orbitals."""
        return self.kpt2.eps_t - self.kpt1.eps_t

    def get_occupation_differences(self):
        """Get difference in occupation factor between orbitals."""
        return self.kpt2.f_t - self.kpt1.f_t

    def get_t_distribution(self):
        """Get what transitions are included compared to the total picture."""
        mynt = self.kpt1.mynt
        nt = self.kpt1.nt
        ta = self.kpt1.ta
        tb = self.kpt1.tb
        assert mynt == self.kpt2.mynt
        assert nt == self.kpt2.nt
        assert ta == self.kpt2.ta
        assert tb == self.kpt2.tb

        return mynt, nt, ta, tb


class KohnShamPair:
    """Class for extracting pairs of Kohn-Sham orbitals from a ground
    state calculation."""
    def __init__(self, gs, world=mpi.world, transitionblockscomm=None,
                 txt='-', timer=None):
        """
        Parameters
        ----------
        transitionblockscomm : gpaw.mpi.Communicator
            Communicator for distributing the transitions among processes
        """
        # Output .txt filehandle
        self.fd = convert_string_to_fd(txt, world)
        self.timer = timer or Timer()

        with self.timer('Read ground state'):
            print('Reading ground state calculation:\n  %s' % gs,
                  file=self.fd)
            self.calc = GPAW(gs, txt=None, communicator=mpi.serial_comm)

        self.transitionblockscomm = transitionblockscomm

        # Prepare to find k-point data from vector
        kd = self.calc.wfs.kd
        self.kdtree = cKDTree(np.mod(np.mod(kd.bzk_kc, 1).round(6), 1))

        # Count bands so it is possible to remove null transitions
        self.nocc1 = None  # number of completely filled bands
        self.nocc2 = None  # number of non-empty bands
        self.count_occupied_bands()

    def count_occupied_bands(self):
        """Count number of occupied and unoccupied bands in ground state
        calculation. Can be used to omit null-transitions between two occupied
        bands or between two unoccupied bands."""
        ftol = 1.e-9  # Could be given as input XXX
        self.nocc1 = 9999999
        self.nocc2 = 0
        for kpt in self.calc.wfs.kpt_u:
            f_n = kpt.f_n / kpt.weight
            self.nocc1 = min((f_n > 1 - ftol).sum(), self.nocc1)
            self.nocc2 = max((f_n > ftol).sum(), self.nocc2)
        print('Number of completely filled bands:', self.nocc1, file=self.fd)
        print('Number of partially filled bands:', self.nocc2, file=self.fd)
        print('Total number of bands:', self.calc.wfs.bd.nbands,
              file=self.fd)

    @timer('Get Kohn-Sham pairs')
    def get_kpoint_pairs(self, n1_t, n2_t, k1_c, k2_c, s1_t, s2_t):
        """Get all pairs of Kohn-Sham orbitals for transitions:
        (n1_t, k1, s1_t) -> (n2_t, k2, s2_t)
        Here, t is a composite band and spin transition index."""
        assert len(n1_t) == len(n2_t)
        kpt1 = self.get_kpoint(k1_c, n1_t, s1_t)
        kpt2 = self.get_kpoint(k2_c, n2_t, s2_t)

        return KohnShamKPointPairs(kpt1, kpt2)

    @timer('Get Kohn-Sham orbitals with given k-point')
    def get_kpoint(self, k_c, n_t, s_t):
        """Get KohnShamKPoint"""
        assert len(n_t) == len(s_t)

        # Parse kpoint to index
        K = self.find_kpoint(k_c)

        nt = len(n_t)
        mynt, ta, tb = self.distribute_transitions(nt)
        myn_t = n_t[ta:tb]
        mys_t = s_t[ta:tb]

        (U_cc, T, a_a, U_aii, shift_c,  # U_cc is unused for now XXX
         time_reversal) = self.construct_symmetry_operators(K, k_c=k_c)

        ut_tR, eps_t, f_t, P_ati = self.extract_orbitals(K, myn_t, mys_t,
                                                         T, a_a, U_aii,
                                                         time_reversal)
        
        return KohnShamKPoint(K, myn_t, mys_t, mynt, nt, ta, tb,
                              ut_tR, eps_t, f_t, P_ati, shift_c)

    def find_kpoint(self, k_c):
        return self.kdtree.query(np.mod(np.mod(k_c, 1).round(6), 1))[1]

    def distribute_transitions(self, nt):
        """Distribute transitions between processes in block communicator"""
        if self.transitionblockscomm is None:
            mynt = nt
            ta = 0
            tb = nt
        else:
            nblocks = self.transitionblockscomm.size
            rank = self.transitionblockscomm.rank

            mynt = (nt + nblocks - 1) // nblocks
            ta = min(rank * mynt, nt)
            tb = min(ta + mynt, nt)

        return mynt, ta, tb

    def construct_symmetry_operators(self, K, k_c=None):
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

        See the extract_orbitals() method for how to use these tuples.
        """

        wfs = self.calc.wfs
        kd = wfs.kd

        s = kd.sym_k[K]
        U_cc = kd.symmetry.op_scc[s]
        time_reversal = kd.time_reversal_k[K]
        ik = kd.bz2ibz_k[K]
        if k_c is None:
            k_c = kd.bzk_kc[K]
        ik_c = kd.ibzk_kc[ik]

        sign = 1 - 2 * time_reversal
        shift_c = np.dot(U_cc, ik_c) - k_c * sign

        try:
            assert np.allclose(shift_c.round(), shift_c)
        except AssertionError:
            print('shift_c ' + str(shift_c), file=self.fd)
            print('k_c ' + str(k_c), file=self.fd)
            print('kd.bzk_kc[K] ' + str(kd.bzk_kc[K]), file=self.fd)
            print('ik_c ' + str(ik_c), file=self.fd)
            print('U_cc ' + str(U_cc), file=self.fd)
            print('sign ' + str(sign), file=self.fd)
            raise AssertionError

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
            S_c = np.dot(self.calc.spos_ac[a], U_cc) - self.calc.spos_ac[b]
            x = np.exp(2j * np.pi * np.dot(ik_c, S_c))
            U_ii = wfs.setups[a].R_sii[s].T * x
            a_a.append(b)
            U_aii.append(U_ii)

        shift0_c = (kd.bzk_kc[K] - k_c).round().astype(int)
        shift_c += -shift0_c

        return U_cc, T, a_a, U_aii, shift_c, time_reversal

    @timer('Extract orbitals from ground state')
    def extract_orbitals(self, K, n_t, s_t, T, a_a, U_aii, time_reversal):
        """Get information about Kohn-Sham orbitals."""
        wfs = self.calc.wfs
        ik = wfs.kd.bz2ibz_k[K]

        # Find array shapes
        nt = len(n_t)
        ni_a = [U_ii.shape[0] for U_ii in U_aii]

        t_t = np.arange(0, nt)
        ut_tR = wfs.gd.empty(nt, wfs.dtype)
        eps_t = np.zeros(nt)
        f_t = np.zeros(nt)
        P_ati = [np.zeros((nt, ni), dtype=complex) for ni in ni_a]

        for s in set(s_t):  # In the ground state, kpts are indexes by u=(s, k)
            myt = s_t == s
            kpt = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]

            eps_t[myt] = kpt.eps_n[n_t[myt]]
            f_t[myt] = kpt.f_n[n_t[myt]] / kpt.weight

            with self.timer('load wfs'):
                '''
                # Extracting psit_tG is slow, so extract after n and sort back
                t_myt, n_myt = t_t[myt], n_t[myt]
                myt_myn = np.argsort(n_myt)
                n_myn = n_myt[myt_myn]
                for t, n in zip(t_myt[myt_myn], n_myn):
                    ut_tR[t] = T(wfs.pd.ifft(kpt.psit_nG[n], ik))
                '''
                '''
                psit_mytG = np.array(kpt.psit_nG)[n_t[myt]]
                for t, psit_G in zip(t_t[myt], psit_mytG):
                    with self.timer('doing transform'):
                        ut_tR[t] = T(wfs.pd.ifft(psit_G, ik))
                '''
                psit_nG = kpt.psit_nG
                for t, n in zip(t_t[myt], n_t[myt]):  # Can be vectorized? XXX
                    ut_tR[t] = T(wfs.pd.ifft(psit_nG[n], ik))

            with self.timer('Load projections'):
                for a, U_ii in zip(a_a, U_aii):  # Can be vectorized? XXX
                    P_myti = np.dot(kpt.P_ani[a][n_t[myt]], U_ii)
                    if time_reversal:
                        P_myti = P_myti.conj()
                    P_ati[a][myt, :] = P_myti

        return ut_tR, eps_t, f_t, P_ati


class PairMatrixElement:
    """Class for calculating matrix elements for transitions in Kohn-Sham
    linear response functions."""
    def __init__(self, kspair):
        """
        Parameters
        ----------
        kslrf : KohnShamLinearResponseFunction instance
        """
        self.calc = kspair.calc
        self.fd = kspair.fd
        self.timer = kspair.timer
        self.transitionblockscomm = kspair.transitionblockscomm

    def __call__(self, kskptpairs, *args, **kwargs):
        """Calculate the matrix element for all transitions in kskptpairs."""
        raise NotImplementedError('Define specific matrix element')


class PlaneWavePairDensity(PairMatrixElement):
    """Class for calculating pair densities:

    n_T(q+G) = <s'n'k'| e^(i (q + G) r) |snk>

    in the plane wave mode"""
    def __init__(self, kspair):
        PairMatrixElement.__init__(self, kspair)

        # Save PAW correction for all calls with same q_c
        self.Q_aGii = None
        self.currentq_c = None

    @timer('Calculate pair density')
    def __call__(self, kskptpairs, pd):
        """Calculate the pair densities for all transitions:
        n_t(q+G) = <s'n'k+q| e^(i (q + G) r) |snk>
                 = <snk| e^(-i (q + G) r) |s'n'k+q>
        """
        Q_aGii = self.initialize_paw_corrections(pd)
        Q_G = self.get_fft_indices(kskptpairs, pd)
        mynt, nt, ta, tb = kskptpairs.get_t_distribution()
        
        n_mytG = pd.empty(mynt)

        # Calculate smooth part of the pair densities:
        # Note: maybe code is slower or just not ready for numpy vectorization
        # Check speed with old and new stuff
        with self.timer('Calculate smooth part'):
            ut1cc_tR = kskptpairs.kpt1.ut_tR.conj()
            n_tR = ut1cc_tR * kskptpairs.kpt2.ut_tR
            n_mytG = np.array([pd.fft(n_tR[t], 0, Q_G) * pd.gd.dv
                               for t in range(tb - ta)])  # Vectorized? XXX
        '''
        # Unvectorized, but using gemm
        for t in range(ta, tb):  # Could be vectorized? XXX
            # Multiply periodic parts of Kohn-Sham orbitals in transition
            ut1cc_R = kskptpairs.kpt1.ut_tR[t - ta].conj()
            n_R = ut1cc_R * kskptpairs.kpt2.ut_tR[t - ta]
            n_mytG[t, :] += pd.fft(n_R, 0, Q_G) * pd.gd.dv

            # Calculate PAW corrections
            C1_aGi = [np.dot(Q_Gii, P1_ti[t - ta].conj())
                      for Q_Gii, P1_ti in zip(Q_aGii, kskptpairs.kpt1.P_ati)]
            for C1_Gi, P2_ti in zip(C1_aGi, kskptpairs.kpt2.P_ati):
                gemm(1.0, C1_Gi, P2_ti[t - ta], 1.0, n_mytG[t], 't')
        '''
        # Calculate PAW corrections with numpy
        with self.timer('PAW corrections'):
            for Q_Gii, P1_ti, P2_ti in zip(Q_aGii, kskptpairs.kpt1.P_ati,
                                           kskptpairs.kpt2.P_ati):
                C1_Git = np.tensordot(Q_Gii, P1_ti.conj(), axes=([1, 1]))
                n_mytG += np.sum(C1_Git * P2_ti.T[np.newaxis, :, :], axis=1).T

        if mynt == nt or self.transitionblockscomm is None:
            return n_mytG
        else:
            n_tG = pd.empty(mynt * self.transitionblockscomm.size)
            self.transitionblockscomm.all_gather(n_mytG, n_tG)
            return n_tG[:nt]

    def initialize_paw_corrections(self, pd):
        """Initialize PAW corrections, if not done already, for the given q"""
        q_c = pd.kd.bzk_kc[0]
        if self.Q_aGii is None or not np.allclose(q_c - self.currentq_c, 0.):
            self.Q_aGii = self._initialize_paw_corrections(pd)
            self.currentq_c = q_c
        return self.Q_aGii

    @timer('Initialize PAW corrections')
    def _initialize_paw_corrections(self, pd):
        wfs = self.calc.wfs
        spos_ac = self.calc.spos_ac
        G_Gv = pd.get_reciprocal_vectors()

        pos_av = np.dot(spos_ac, pd.gd.cell_cv)

        # Collect integrals for all species:
        Q_xGii = {}
        for id, atomdata in wfs.setups.setups.items():
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

        return Q_aGii

    @timer('Get G-vector indices')
    def get_fft_indices(self, kskptpairs, pd):
        """Get indices for G-vectors inside cutoff sphere."""
        kpt1 = kskptpairs.kpt1
        kpt2 = kskptpairs.kpt2
        kd = self.calc.wfs.kd
        q_c = pd.kd.bzk_kc[0]

        N_G = pd.Q_qG[0]

        shift_c = kpt1.shift_c - kpt2.shift_c
        shift_c += (q_c - kd.bzk_kc[kpt2.K]
                    + kd.bzk_kc[kpt1.K]).round().astype(int)
        if shift_c.any():
            n_cG = np.unravel_index(N_G, pd.gd.N_c)
            n_cG = [n_G + shift for n_G, shift in zip(n_cG, shift_c)]
            N_G = np.ravel_multi_index(n_cG, pd.gd.N_c, 'wrap')
        return N_G
