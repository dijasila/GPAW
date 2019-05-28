import numpy as np
from scipy.spatial import cKDTree

from ase.utils import convert_string_to_fd
from ase.utils.timing import Timer, timer

from gpaw import GPAW
import gpaw.mpi as mpi


class KohnShamKPoint:
    """Kohn-Sham orbitals participating in transitions for a given k-point."""
    def __init__(self, K, n_t, s_t, blocksize, ta, tb,
                 ut_tR, eps_t, f_t, P_ati, shift_c):
        self.K = K      # BZ k-point index
        self.n_t = n_t  # Band index for each transition
        self.s_t = s_t  # Spin index for each transition
        self.blocksize = blocksize
        self.ta = ta    # first transition index of this block
        self.tb = tb    # first transition index of this block not included
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

    def get_k1(self):  # Which of these methods are actually used? XXX
        """ Return KPoint object 1."""
        return self.kpt1

    def get_k2(self):
        """ Return KPoint object 2."""
        return self.kpt2

    def get_transition_energies(self):
        """Return the energy differences between orbitals."""
        return self.kpt2.eps_t - self.kpt1.eps_t

    def get_occupation_differences(self):
        """Get difference in occupation factor between orbitals."""
        return self.kpt2.f_t - self.kpt1.f_t


class KohnShamPair:
    """Class for extracting pairs of Kohn-Sham orbitals from a ground
    state calculation."""
    def __init__(self, gs, world=mpi.world, nblocks=1, txt='-', timer=None):
        # Output .txt filehandle and timer
        self.fd = convert_string_to_fd(txt, world)
        self.timer = timer or Timer()

        # Communicators
        self.world = world
        self.blockcomm = None
        self.kncomm = None
        self.initialize_communicators(nblocks)

        with self.timer('Read ground state'):
            print('Reading ground state calculation:\n  %s' % gs,
                  file=self.fd)
            self.calc = GPAW(gs, txt=None, communicator=mpi.serial_comm)

        # Prepare to find k-point data from vector
        kd = self.calc.wfs.kd
        self.kdtree = cKDTree(np.mod(np.mod(kd.bzk_kc, 1).round(6), 1))

        # Count bands to remove null-transitions
        self.nocc1 = None  # number of completely filled bands
        self.nocc2 = None  # number of non-empty bands
        self.count_occupied_bands()

    def initialize_communicators(self, nblocks):
        """Set up MPI communicators to avoid each process storing the same
        arrays."""
        if nblocks == 1:
            self.blockcomm = self.world.new_communicator([self.world.rank])
            self.kncomm = self.world
        else:
            assert self.world.size % nblocks == 0, self.world.size
            rank1 = self.world.rank // nblocks * nblocks
            rank2 = rank1 + nblocks
            self.blockcomm = self.world.new_communicator(range(rank1, rank2))
            ranks = range(self.world.rank % nblocks, self.world.size, nblocks)
            self.kncomm = self.world.new_communicator(ranks)
        print('Number of blocks:', nblocks, file=self.fd)

    def count_occupied_bands(self):
        """Count number of occupied and unoccupied bands in ground state
        calculation. Can be used to omit null-transitions between two occupied
        bands or between two unoccupied bands."""
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

    @timer('Get Kohn-Sham pairs')
    def get_kpoint_pairs(self, n1_t, n2_t, k1_c, k2_c, s1_t, s2_t):
        """Get all pairs of Kohn-Sham orbitals for transitions:
        (n1_t, k1, s1_t) -> (n2_t, k2, s2_t)
        Here, t is a composite band and spin transition index."""
        assert len(n1_t) == len(n2_t)
        with self.timer('get k-points'):
            kpt1 = self.get_kpoint(k1_c, n1_t, s1_t)
            kpt2 = self.get_kpoint(k2_c, n2_t, s2_t)
        return KohnShamKPointPairs(kpt1, kpt2)

    @timer('Get Kohn-Sham orbitals with given k-point')
    def get_kpoint(self, k_c, n_t, s_t):
        """Get KohnShamKPoint"""
        assert len(n_t) == len(s_t)

        # Parse kpoint to index
        K = self.find_kpoint(k_c)

        blocksize, ta, tb = self.distribute_transitions(len(n_t))
        myn_t = n_t[ta:tb]
        mys_t = s_t[ta:tb]

        (U_cc, T, a_a, U_aii, shift_c,  # U_cc is unused for now XXX
         time_reversal) = self.construct_symmetry_operators(K, k_c=k_c)

        ut_tR, eps_t, f_t, P_ati = self.extract_orbitals(K, myn_t, mys_t,
                                                         T, a_a, U_aii,
                                                         time_reversal)
        
        return KohnShamKPoint(K, myn_t, mys_t, blocksize, ta, tb,
                              ut_tR, eps_t, f_t, P_ati, shift_c)

    def find_kpoint(self, k_c):
        return self.kdtree.query(np.mod(np.mod(k_c, 1).round(6), 1))[1]

    def distribute_transitions(self, nt):
        """Distribute transitions between processes."""
        nblocks = self.blockcomm.size
        rank = self.blockcomm.rank

        blocksize = (nt + nblocks - 1) // nblocks
        ta = min(rank * blocksize, nt)
        tb = min(ta + blocksize, nt)

        return blocksize, ta, tb

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
            S_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            x = np.exp(2j * np.pi * np.dot(ik_c, S_c))
            U_ii = wfs.setups[a].R_sii[s].T * x
            a_a.append(b)
            U_aii.append(U_ii)

        shift0_c = (kd.bzk_kc[K] - k_c).round().astype(int)
        shift_c += -shift0_c

        return U_cc, T, a_a, U_aii, shift_c, time_reversal

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
