from abc import ABC, abstractmethod
import numpy as np


class Integrand:
    @abstractmethod
    def matrix_element(self, k_v, s):
        ...

    @abstractmethod
    def eigenvalues(self, k_v, s):
        ...

    def __iter__(self):
        yield self.matrix_element
        yield self.eigenvalues


class PlasmaFrequencyIntegrand(Integrand):
    def __init__(self, chi0drudecalc, qpd, analyzer):
        self._drude = chi0drudecalc
        self.qpd = qpd
        self.analyzer = analyzer

    def _band_summation(self):
        # Intraband response needs only integrate partially unoccupied bands.
        return self._drude.nocc1, self._drude.nocc2

    def matrix_element(self, k_v, s):
        """NB: In dire need of documentation! XXX."""
        n1, n2 = self._band_summation()
        k_c = np.dot(self.qpd.gd.cell_cv, k_v) / (2 * np.pi)
        kpt1 = self._drude.pair.get_k_point(s, k_c, n1, n2)
        n_n = range(n1, n2)

        vel_nv = self._drude.pair.intraband_pair_density(kpt1, n_n)

        if self._drude.integrationmode is None:
            f_n = kpt1.f_n
            width = self._drude.gs.get_occupations_width()
            if width > 1e-15:
                dfde_n = - 1. / width * (f_n - f_n**2.0)
            else:
                dfde_n = np.zeros_like(f_n)
            vel_nv *= np.sqrt(-dfde_n[:, np.newaxis])
            weight = np.sqrt(self.analyzer.get_kpoint_weight(k_c) /
                             self.analyzer.how_many_symmetries())
            vel_nv *= weight

        return vel_nv

    def eigenvalues(self, k_v, s):
        """A function that can return the intraband eigenvalues.

        A method describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        n1, n2 = self._band_summation()
        gs = self._drude.gs
        kd = gs.kd
        k_c = np.dot(self.qpd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = self._drude.pair.find_kpoint(k_c)
        ik = kd.bz2ibz_k[K1]
        kpt1 = gs.kpt_qs[ik][s]
        assert gs.kd.comm.size == 1

        return kpt1.eps_n[n1:n2]
