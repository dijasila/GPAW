import numpy as np

from gpaw.response.chi0 import Chi0


class JDOS(Chi0):
    def __init__(self, calc, *args, **kwargs):
        Chi0.__init__(self, calc, ecut=1e-3, *args, **kwargs)
        
    def calculate(self):
        _, _, _, chi0_wvv = Chi0.calculate(self, [0, 0, 0])

        return -np.imag(chi0_wvv)[:, 0, 0]

    def get_intraband_response(self, k_v, s, n1=None, n2=None,
                               kd=None, symmetry=None, pd=None):
        return np.zeros((n2 - n1, 3), complex)

    def get_matrix_element(self, k_v, s, n1=None, n2=None,
                           m1=None, m2=None,
                           pd=None, kd=None,
                           symmetry=None):

        wfs = self.calc.wfs
        kd = wfs.kd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = kd.where_is_q(k_c, kd.bzk_kc)
        ik = kd.bz2ibz_k[K1]
        kpt1 = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]
        K2 = kd.where_is_q(k_c + pd.kd.bzk_kc[0], kd.bzk_kc)
        ik = kd.bz2ibz_k[K2]
        kpt2 = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]
        df_nm = (kpt1.f_n[n1:n2][:, np.newaxis]
                 - kpt2.f_n[m1:m2][np.newaxis])
        df_nm[df_nm <= 0.0] = 0
        df_M = df_nm.astype(complex).reshape(-1)
        df_MG = np.tile(df_M[:, np.newaxis], (1, 3))


        return df_MG
