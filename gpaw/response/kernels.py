# -*- coding: utf-8
# In an attempt to appease epydoc and still have readable docstrings,
# the vertical bar | is represented by u'\u2758' in this module.
"""This module defines Coulomb and XC kernels for the response model.
"""

import numpy as np

def get_coulomb_kernel(pd, N_c, truncation=None, q_inf=None):
    """Factory function that calls returns the specified flavour 
    of the Coulomb interaction"""

    if truncation is None:
        if pd.kd.gamma:
            v_G = np.zeros(len(pd.G2_qG[0]))
            v_G[0] = 4 * np.pi
            v_G[1:] = 4 * np.pi / pd.G2_qG[0][1:]
        else:
            v_G = 4 * np.pi / pd.G2_qG[0]
        if q_inf is not None:
             v_G[0] = 4 * np.pi / q_inf**2

    elif truncation == '2D':
        v_G = calculate_2D_truncated_coulomb(pd, q_inf=q_inf)
        if pd.kd.gamma and q_inf is None:
            v_G[0] = 0.0

    elif truncation == 'wigner-seitz':
        from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
        wstc = WignerSeitzTruncatedCoulomb(pd.gd.cell_cv, N_c)
        v_G = wstc.get_potential(pd)
        if pd.kd.gamma:
            v_G[0] = 0.0
    else:
        raise ValueError('Truncation scheme %s not implemented' % truncation)
  
    return v_G

def calculate_2D_truncated_coulomb(pd, q_inf=None):
    """ Simple truncation of Coulomb kernel along z
    Rozzi, C., Varsano, D., Marini, A., Gross, E., & Rubio, A. (2006).
    Exact Coulomb cutoff technique for supercell calculations.
    Physical Review B, 73(20), 205119. doi:10.1103/PhysRevB.73.205119
    """

    qG_Gv = pd.get_reciprocal_vectors(add_q=True)
    if pd.kd.gamma:  # Set small finite q to handle divergence
        if q_inf is not None:
            qG_Gv += [q_inf, 0, 0]
        else:
            qG_Gv[0] = 1.
    nG = len(qG_Gv)
    L = pd.gd.cell_cv[2, 2]
    R = L / 2.  # Truncation length is half of unit cell
    qGpar_G = ((qG_Gv[:, 0])**2 + (qG_Gv[:, 1]**2))**0.5
    qGz_G = qG_Gv[:, 2]
    
    v_G = 4 * np.pi / (qG_Gv**2).sum(axis=1)
    # K_G *= 1. + np.exp(-qG_par * R) * (qG_z / qG_par * np.sin(qG_z * R)\
    #                                     - np.cos(qG_z * R))

    v_G *= 1.0 - np.exp(-qG_par * R) * np.cos(qG_z * R)
    # sin(qG_z * R) = 0 when R = L/2

    return v_G
