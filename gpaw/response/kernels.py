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
        v_G = calculate_2D_truncated_coulomb(pd, q_inf=q_inf, N_c=N_c)
        if pd.kd.gamma and q_inf is None:
            v_G[0] = 0.0

    elif truncation == '1D':
        v_G = calculate_1D_truncated_coulomb(pd, q_inf=q_inf, N_c=N_c)
        if pd.kd.gamma and q_inf is None:
            v_G[0] = 0.0

    elif truncation == '0D':
        v_G = calculate_0D_truncated_coulomb(pd, q_inf=q_inf)
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

def calculate_2D_truncated_coulomb(pd, q_inf=None, N_c=None):
    """ Simple 2D truncation of Coulomb kernel PRB 73, 205119.
        The non-periodic direction is determined from k-point grid.
    """

    qG_Gv = pd.get_reciprocal_vectors(add_q=True)
    if pd.kd.gamma:  # Set small finite q to handle divergence
        if q_inf is not None:
            qG_Gv += [q_inf, 0, 0]
        else: # Only to avoid warning. Later set to zero in factory function 
            qG_Gv[0] = [1., 1., 1.]
    nG = len(qG_Gv)

    # The non-periodic direction is determined from k-point grid
    Nn_c = np.where(N_c == 1)[0]
    Np_c = np.where(N_c != 1)[0]
    if len(Nn_c) != 1:
        # The k-point grid does not fit with boundary conditions
        Nn_c = [2]    # Choose reduced cell vectors 0, 1
        Np_c = [0, 1] # Choose reduced cell vector 2
    R = pd.gd.cell_cv[Nn_c[0], Nn_c[0]] / 2. 
    # Truncation length is half of cell vector in non-periodic direction

    qGpar_G = ((qG_Gv[:, Np_c[0]])**2 + (qG_Gv[:, Np_c[1]]**2))**0.5
    qGz_G = qG_Gv[:, Nn_c[0]]
    
    v_G = 4 * np.pi / (qG_Gv**2).sum(axis=1)
    # K_G *= 1. + np.exp(-qGpar_G * R) * (qG_z / qGpar_G * np.sin(qGz_G * R)\
    #                                     - np.cos(qGz_G * R))

    v_G *= 1.0 - np.exp(-qGpar_G * R) * np.cos(qGz_G * R)
    # sin(qG_z * R) = 0 when R = L/2

    return v_G

def calculate_1D_truncated_coulomb(pd, q_inf=None, N_c=None):
    """ Simple 1D truncation of Coulomb kernel PRB 73, 205119. 
    The periodic direction is determined from k-point grid.
    """

    from scipy.special import j1, k0, j0, k1

    qG_Gv = pd.get_reciprocal_vectors(add_q=True)
    if q_inf is None:
        raise ValueError('Presently, calculations only work with a small q in the normal direction')
    qG_Gv += [q_inf, q_inf, 0]
    nG = len(qG_Gv)

    # The periodic direction is determined from k-point grid
    Nn_c = np.where(N_c == 1)[0]
    Np_c = np.where(N_c != 1)[0]
    if len(Nn_c) != 2:
        # The k-point grid does not fit with boundary conditions
        Nn_c = [0, 1]    # Choose reduced cell vectors 0, 1
        Np_c = [2]       # Choose reduced cell vector 2
    Acell_cv = pd.gd.cell_cv[Nn_c, :][:, Nn_c]
    R = (np.linalg.det(Acell_cv) / np.pi)**0.5
    # The radius is determined from area of non-periodic part of cell

    qGnR_G = (qG_Gv[:, Nn_c[0]]**2 + qG_Gv[:, Nn_c[1]]**2)**0.5 * R
    qGpR_G = abs(qG_Gv[:, Np_c[0]]) * R
    v_G = 4 * np.pi / (qG_Gv**2).sum(axis=1)
    v_G *= (1. + qGnR_G * j1(qGnR_G) * k0(qGpR_G)
            - qGpR_G * j0(qGnR_G) * k1(qGpR_G))

    return v_G

def calculate_0D_truncated_coulomb(pd, q_inf=None):
    """ Simple spherical truncation of the Coulomb interaction
    PRB 73, 205119
    """

    qG_Gv = pd.get_reciprocal_vectors(add_q=True)
    if pd.kd.gamma:  # Set small finite q to handle divergence
        if q_inf is not None:
            qG_Gv += [0, 0, q_inf]
        else: # Only to avoid warning. Later set to zero in factory function 
            qG_Gv[0] = [1., 1., 1.]
    nG = len(qG_Gv)
    
    R = (3 * np.linalg.det(pd.gd.cell_cv) / (4 * np.pi))**(1. / 3.)
    # The radius is determined from volume of cell

    qG2_G = (qG_Gv**2).sum(axis=1)
    
    v_G = 4 * np.pi / qG2_G
    v_G *= 1.0 - np.cos(qG2_G**0.5 * R)

    return v_G
