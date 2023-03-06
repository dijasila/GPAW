import numpy as np


class PairTransitions:
    """Some documentation here! XXX
    """

    def __init__(self, n1_t, n2_t, s1_t, s2_t):
        """Construct the PairTransitions object.

        Parameters
        ----------
        n1_t : np.array
            Band index of k-point k for each transition t.
        n2_t : np.array
            Band index of k-point k + q for each transition t.
        s1_t : np.array
            Spin index of k-point k for each transition t.
        s2_t : np.array
            Spin index of k-point k + q for each transition t.
        """
        self.n1_t = n1_t
        self.n2_t = n2_t
        self.s1_t = s1_t
        self.s2_t = s2_t

        assert len(n2_t) == len(self)
        assert len(s1_t) == len(self)
        assert len(s2_t) == len(self)

    def __len__(self):
        return len(self.n1_t)

    @staticmethod
    def from_transitions_domain_arguments(bandsummation, nbands, nocc1, nocc2,
                                          nspins, spin_rotation):
        """Some documentation here! XXX

        This excludes transitions between two occupied bands and two unoccupied
        bands respectively.
        """

        n1_M, n2_M = get_band_transitions_domain(bandsummation, nbands,
                                                 nocc1=nocc1,
                                                 nocc2=nocc2)
        s1_S, s2_S = get_spin_transitions_domain(bandsummation,
                                                 spin_rotation, nspins)

        n1_t, n2_t, s1_t, s2_t = transitions_in_composite_index(n1_M, n2_M,
                                                                s1_S, s2_S)

        return PairTransitions(n1_t, n2_t, s1_t, s2_t)


def get_band_transitions_domain(bandsummation, nbands, nocc1=None, nocc2=None):
    """Get all pairs of bands to sum over

    Parameters
    ----------
    bandsummation : str
        Band summation method
    nbands : int
        number of bands
    nocc1 : int
        number of completely filled bands
    nocc2 : int
        number of non-empty bands

    Returns
    -------
    n1_M : ndarray
        band index 1, M = (n1, n2) composite index
    n2_M : ndarray
        band index 2, M = (n1, n2) composite index
    """
    _get_band_transitions_domain =\
        create_get_band_transitions_domain(bandsummation)
    n1_M, n2_M = _get_band_transitions_domain(nbands)

    return remove_null_transitions(n1_M, n2_M, nocc1=nocc1, nocc2=nocc2)


def create_get_band_transitions_domain(bandsummation):
    """Creator component deciding how to carry out band summation."""
    if bandsummation == 'pairwise':
        return get_pairwise_band_transitions_domain
    elif bandsummation == 'double':
        return get_double_band_transitions_domain
    raise ValueError(bandsummation)


def get_double_band_transitions_domain(nbands):
    """Make a simple double sum"""
    n_n = np.arange(0, nbands)
    m_m = np.arange(0, nbands)
    n_nm, m_nm = np.meshgrid(n_n, m_m)
    n_M, m_M = n_nm.flatten(), m_nm.flatten()

    return n_M, m_M


def get_pairwise_band_transitions_domain(nbands):
    """Make a sum over all pairs"""
    n_n = range(0, nbands)
    n_M = []
    m_M = []
    for n in n_n:
        m_m = range(n, nbands)
        n_M += [n] * len(m_m)
        m_M += m_m

    return np.array(n_M), np.array(m_M)


def remove_null_transitions(n1_M, n2_M, nocc1=None, nocc2=None):
    """Remove pairs of bands, between which transitions are impossible"""
    n1_newM = []
    n2_newM = []
    for n1, n2 in zip(n1_M, n2_M):
        if nocc1 is not None and (n1 < nocc1 and n2 < nocc1):
            continue  # both bands are fully occupied
        elif nocc2 is not None and (n1 >= nocc2 and n2 >= nocc2):
            continue  # both bands are completely unoccupied
        n1_newM.append(n1)
        n2_newM.append(n2)

    return np.array(n1_newM), np.array(n2_newM)


def get_spin_transitions_domain(bandsummation, spinrot, nspins):
    """Get structure of the sum over spins

    Parameters
    ----------
    bandsummation : str
        Band summation method
    spinrot : str
        spin rotation
    nspins : int
        number of spin channels in ground state calculation

    Returns
    -------
    s1_s : ndarray
        spin index 1, S = (s1, s2) composite index
    s2_S : ndarray
        spin index 2, S = (s1, s2) composite index
    """
    _get_spin_transitions_domain =\
        create_get_spin_transitions_domain(bandsummation)
    return _get_spin_transitions_domain(spinrot, nspins)


def create_get_spin_transitions_domain(bandsummation):
    """Creator component deciding how to carry out spin summation."""
    if bandsummation == 'pairwise':
        return get_pairwise_spin_transitions_domain
    elif bandsummation == 'double':
        return get_double_spin_transitions_domain
    raise ValueError(bandsummation)


def get_double_spin_transitions_domain(spinrot, nspins):
    """Usual spin rotations forward in time"""
    if nspins == 1:
        if spinrot is None or spinrot == '0':
            s1_S = [0]
            s2_S = [0]
        else:
            raise ValueError(spinrot, nspins)
    else:
        if spinrot is None:
            s1_S = [0, 0, 1, 1]
            s2_S = [0, 1, 0, 1]
        elif spinrot == '0':
            s1_S = [0, 1]
            s2_S = [0, 1]
        elif spinrot == 'u':
            s1_S = [0]
            s2_S = [0]
        elif spinrot == 'd':
            s1_S = [1]
            s2_S = [1]
        elif spinrot == '-':
            s1_S = [0]  # spin up
            s2_S = [1]  # spin down
        elif spinrot == '+':
            s1_S = [1]  # spin down
            s2_S = [0]  # spin up
        else:
            raise ValueError(spinrot)

    return np.array(s1_S), np.array(s2_S)


def get_pairwise_spin_transitions_domain(spinrot, nspins):
    """In a sum over pairs, transitions including a spin rotation may have to
    include terms, propagating backwards in time."""
    if spinrot in ['+', '-']:
        assert nspins == 2
        return np.array([0, 1]), np.array([1, 0])
    else:
        return get_double_spin_transitions_domain(spinrot, nspins)


def transitions_in_composite_index(n1_M, n2_M, s1_S, s2_S):
    """Use a composite index t for transitions (n, s) -> (n', s')."""
    n1_MS, s1_MS = np.meshgrid(n1_M, s1_S)
    n2_MS, s2_MS = np.meshgrid(n2_M, s2_S)
    return n1_MS.flatten(), n2_MS.flatten(), s1_MS.flatten(), s2_MS.flatten()
