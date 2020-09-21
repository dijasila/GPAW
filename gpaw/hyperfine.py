"""Hyperfine parameters.

See:

    First-principles calculations of defects in oxygen-deficient
    silica exposed to hydrogen

    Peter E. Blöchl

    Phys. Rev. B 62, 6158 – Published 1 September 2000

    https://doi.org/10.1103/PhysRevB.62.6158

"""
from typing import Any, Tuple, List
from math import pi

import numpy as np

from gpaw import GPAW
from gpaw.setup import Setup
from gpaw.grid_descriptor import GridDescriptor
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.utilities import unpack2
from gpaw.gaunt import gaunt

Array1D = Any
Array2D = Any
Array3D = Any


def hyperfine_parameters(calc: GPAW) -> Tuple[Array1D, Array3D]:
    dens = calc.density
    nt_sR = dens.nt_sG
    W1_a, W2_avv = smooth_part(
        nt_sR[0] - nt_sR[1],
        dens.gd,
        calc.atoms.get_scaled_positions())

    D_asp = calc.density.D_asp
    for a, D_sp in D_asp.items():
        W1, W2_vv = paw_correction(unpack2(D_sp[0] - D_sp[1]),
                                   calc.wfs.setups[a])
        W1_a[a] += W1
        W2_avv[a] += W2_vv

    return W1_a, W2_avv


def smooth_part(spin_density_R: Array3D,
                gd: GridDescriptor,
                spos_ac: Array2D,
                ecut: float = None) -> Tuple[Array1D, Array3D]:
    pd = PWDescriptor(ecut, gd)
    spin_density_G = pd.fft(spin_density_R)
    G_Gv = pd.get_reciprocal_vectors()
    eiGR_aG = np.exp(-1j * spos_ac.dot(gd.cell_cv).dot(G_Gv.T))

    W1_a = pd.integrate(spin_density_G, eiGR_aG) / gd.dv * (2 / 3)

    spin_density_G[0] = 0.0
    G2_G = pd.G2_qG[0].copy()
    G2_G[0] = 1.0
    spin_density_G /= G2_G

    W2_vva = np.empty((3, 3, len(spos_ac)))
    for v1 in range(3):
        for v2 in range(3):
            W_a = pd.integrate(G_Gv[:, v1] * G_Gv[:, v2] * spin_density_G,
                               eiGR_aG)
            W2_vva[v1, v2] = -W_a / gd.dv

    W2_a = np.trace(W2_vva) / 3
    for v in range(3):
        W2_vva[v, v] -= W2_a

    return W1_a, W2_vva.transpose((2, 0, 1))


Y2_m = (np.array([15 / 4, 15 / 4, 5 / 16, 15 / 4, 15 / 16]) / pi)**0.5
Y2_mvv = np.array([[[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]],
                   [[-2, 0, 0],
                    [0, -2, 0],
                    [0, 0, 4]],
                   [[0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0]],
                   [[2, 0, 0],
                    [0, -1, 0],
                    [0, 0, 0]]])


def paw_correction(spin_density_ii: Array2D,
                   setup: Setup) -> Tuple[float, Array2D]:
    D0_jj = expand(spin_density_ii, setup.l_j, 0)[0]

    phit_jg = np.array(setup.data.phit_jg)
    phi_jg = np.array(setup.data.phi_jg)

    rgd = setup.rgd

    nt0 = phit_jg[:, 0].dot(D0_jj).dot(phit_jg[:, 0]) / (4 * pi)**0.5
    n0 = phit_jg[:, 0].dot(D0_jj).dot(phi_jg[:, 0]) / (4 * pi)**0.5
    W1 = (n0 - nt0) * 2 / 3

    D2_mjj = expand(spin_density_ii, setup.l_j, 2)
    dn2_mg = np.einsum('mab, ag, bg -> mg', D2_mjj, phi_jg, phi_jg)
    dn2_mg -= np.einsum('mab, ag, bg -> mg', D2_mjj, phit_jg, phit_jg)
    A_m = dn2_mg[:, 1:].dot(rgd.dr_g[1:] / rgd.r_g[1:]) * (4 * pi)
    A_m *= Y2_m
    W2_vv = Y2_mvv.T.dot(A_m)
    W2 = np.trace(W2_vv) / 3
    for v in range(3):
        W2_vv[v, v] -= W2

    return W1, W2_vv


def expand(D_ii: Array2D,
           l_j: List[int],
           l: int) -> Array3D:
    G_LLm = gaunt(lmax=2)[:, :, l**2:(l + 1)**2]
    D_mjj = np.empty((2 * l + 1, len(l_j), len(l_j)))
    i1a = 0
    for j1, l1 in enumerate(l_j):
        i1b = i1a + 2 * l1 + 1
        i2a = 0
        for j2, l2 in enumerate(l_j):
            i2b = i2a + 2 * l2 + 1
            D_mjj[:, j1, j2] = np.einsum('ab, abm -> m',
                                         D_ii[i1a:i1b, i2a:i2b],
                                         G_LLm[l1**2:(l1 + 1)**2,
                                               l2**2:(l2 + 1)**2])
            i2a = i2b
        i1a = i1b
    return D_mjj
