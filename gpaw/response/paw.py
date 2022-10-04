import numpy as np
from gpaw.response.math_func import two_phi_planewave_integrals


class PAWCorrections:
    def __init__(self, Q_aGii):
        self.Q_aGii = Q_aGii

def calculate_paw_corrections(setups, pd, spos_ac):
    q_v = pd.K_qv[0]
    optical_limit = np.allclose(q_v, 0)

    G_Gv = pd.get_reciprocal_vectors()
    if optical_limit:
        G_Gv[0] = 1

    pos_av = spos_ac @ pd.gd.cell_cv

    # Collect integrals for all species:
    Q_xGii = {}
    for id, atomdata in setups.setups.items():
        ni = atomdata.ni
        Q_Gii = two_phi_planewave_integrals(G_Gv, atomdata)
        Q_xGii[id] = Q_Gii.reshape(-1, ni, ni)

    Q_aGii = []
    for a, atomdata in enumerate(setups):
        id = setups.id_a[a]
        Q_Gii = Q_xGii[id]
        x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
        Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)
        if optical_limit:
            Q_aGii[a][0] = atomdata.dO_ii

    return Q_aGii
