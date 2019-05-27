import numpy as np
from scipy.spatial import cKDTree

from ase.utils import convert_string_to_fd
from ase.utils.timing import Timer

from gpaw import GPAW
import gpaw.mpi as mpi


class KPoint:
    """Kohn-Sham orbitals participating in transitions for a given k-point."""
    def __init__(self, K, n_t, s_t, blocksize, ta, tb,
                 ut_tR, eps_t, f_t, P_ati, shift_c):
        self.K = K      # BZ k-point index
        self.n_t = n_t  # Band index for each transition
        self.s_t = s_t  # Spin index for each transition
        self.blocksize = blocksize
        self.ta = ta    # first transition of block
        self.tb = tb    # first transition of block not included
        self.ut_tR = ut_tR      # periodic part of wave functions in real-space
        self.eps_t = eps_t      # eigenvalues
        self.f_t = f_t          # occupation numbers
        self.P_ati = P_ati      # PAW projections

        self.shift_c = shift_c  # long story - see the  # remove? XXX
        # PairDensity.construct_symmetry_operators() method


class KPointPair:
    """Object containing all transitions between Kohn-Sham orbitals with a pair
    of k-points."""
    def __init__(self, kpt1, kpt2):
        self.kpt1 = kpt1
        self.kpt2 = kpt2

    def get_k1(self):
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
        self.KDTree = cKDTree(np.mod(np.mod(kd.bzk_kc, 1).round(6), 1))

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
