"""
Module for the calcuation of the electron-phonon self-energy
"""
import numpy as np

from .observable import Observable


class Selfenergy(Observable):
    """Class for electron-phonon self-energy"""
    def fan_self_energy_sa(self, eta: float = 0.1):
        r"""Fan-Migdal contribution to self-energy in on-the-mass-shell approx

        See F. Giustino, Reviews of Modern Physics 89, 015003 (2017)
        Eqs.157, 158, but with

        \Sigma = 0.5 [ \Sigma(\omega=E_nk) + \Sigma(\omega=E_n'k)]

        Parameters
        ----------
        eta: float
            Broadening for self-energy in eV
        """
        nspin = self.g_sqklnn.shape[0]
        nk = self.g_sqklnn.shape[2]
        nbands = self.g_sqklnn.shape[4]

        fw_ql = self.get_bose_factor(self.w_ql)
        fan_sknn = np.zeros((nspin, nk, nbands, nbands), dtype=complex)

        kd = self.calc.wfs.kd
        k_qc = kd.get_bz_q_points(first=True)
        assert k_qc.ndim == 2

        for s in range(self.calc.wfs.nspins):
            for q, q_c in enumerate(k_qc):
                # Find indices of k+q for the k-points
                kplusq_k = kd.find_k_plus_q(q_c)
                for k in range(kd.nbzkpts):
                    k_c = kd.bzk_kc[k]
                    kplusq_c = k_c + q_c
                    kplusq_c -= kplusq_c.round()  # 1BZ
                    assert np.allclose(kplusq_c, kd.bzk_kc[kplusq_k[k]])
                    # k+q occupation numbers
                    f_n = self.calc.get_occupation_numbers(kplusq_k[k], s) * \
                        kd.nspins / 2 / kd.weight_k[kplusq_k[k]]
                    # print(s, q, k, f_n)
                    assert np.isclose(max(f_n), 1.0, atol=0.1)

                    # according to QE ieta < 0 if E_kn < Ef
                    _f = self.calc.get_occupation_numbers(k, s) * \
                        kd.nspins / 2 / kd.weight_k[k]
                    mask = np.where(_f > 0.5)[0]
                    ieta = 1j * eta * np.ones_like(_f, dtype=complex)
                    ieta[mask] *= -1.

                    # Get eigenvalues returns eV (internal GPAW is Ha)
                    ek_n = self.calc.get_eigenvalues(k, s) + ieta  # k
                    eq_n = self.calc.get_eigenvalues(kplusq_k[k], s)  # k+q
                    deltae_mn = ek_n[None, :] - eq_n[:, None]

                    for l, (fw, w) in enumerate(zip(fw_ql[q], self.w_ql[q])):
                        ff_mn = (fw + f_n[:, None]) / (deltae_mn + w) + \
                                (fw + 1. - f_n[:, None]) / (deltae_mn - w)
                        g_nn = self.g_sqklnn[s, q, k, l]

                        # n^prime = p
                        # omega = e_n' and omega=e_n contribution in OSA
                        fan_nn = np.einsum('mn,mp,mn->np', g_nn.conj(), g_nn,
                                           ff_mn)
                        fan_nn += np.einsum('mn,mp,mp->np', g_nn.conj(), g_nn,
                                            ff_mn)
                        fan_nn /= 2. * w * 2.  # last two to average OSA
                        fan_sknn[s, k] += fan_nn / len(k_qc)
        return fan_sknn
