import numpy as np
from gpaw.response.math_func import (two_phi_planewave_integrals,
                                     two_phi_nabla_planewave_integrals)


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


def calculate_paw_nabla_corrections(setups, spos_ac, pd, soft):
    G_Gv = pd.get_reciprocal_vectors()
    pos_av = np.dot(spos_ac, pd.gd.cell_cv)

    # Collect integrals for all species:
    Q_xvGii = {}
    for id, atomdata in setups.setups.items():
        if soft:
            raise NotImplementedError
        else:
            Q_vGii = two_phi_nabla_planewave_integrals(G_Gv, atomdata)
            ni = atomdata.ni
            Q_vGii.shape = (3, -1, ni, ni)

        Q_xvGii[id] = Q_vGii

    Q_avGii = []
    for a, atomdata in enumerate(setups):
        id = setups.id_a[a]
        Q_vGii = Q_xvGii[id]
        x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
        Q_avGii.append(x_G[np.newaxis, :, np.newaxis, np.newaxis] * Q_vGii)

    return Q_avGii
