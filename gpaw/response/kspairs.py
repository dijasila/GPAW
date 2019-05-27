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
