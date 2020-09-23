"""Hyperfine parameters.

See:

    First-principles calculations of defects in oxygen-deficient
    silica exposed to hydrogen

    Peter E. Blöchl

    Phys. Rev. B 62, 6158 – Published 1 September 2000

    https://doi.org/10.1103/PhysRevB.62.6158

"""
from typing import Any, List
from math import pi

import numpy as np
from scipy.integrate import simps
import ase.units as units

from gpaw import GPAW
from gpaw.atom.radialgd import RadialGridDescriptor
from gpaw.setup import Setup
from gpaw.grid_descriptor import GridDescriptor
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.utilities import unpack2
from gpaw.gaunt import gaunt

Array1D = Any
Array2D = Any
Array3D = Any


def hyperfine_parameters(calc: GPAW) -> Array3D:
    """Calculate isotropic and anisotropic hyperfine coupling paramters.

    One tensor per atom is returned.  The isotropic part is a=trace(A)/3
    and the anisotropic part is A-a.
    """
    dens = calc.density
    nt_sR = dens.nt_sG
    A_avv = smooth_part(
        nt_sR[0] - nt_sR[1],
        dens.gd,
        calc.atoms.get_scaled_positions())

    D_asp = calc.density.D_asp
    for a, D_sp in D_asp.items():
        A_vv = paw_correction(unpack2(D_sp[0] - D_sp[1]),
                              calc.wfs.setups[a])
        A_avv[a] += A_vv

    return A_avv


def smooth_part(spin_density_R: Array3D,
                gd: GridDescriptor,
                spos_ac: Array2D,
                ecut: float = None) -> Array3D:
    """Contribution from pseudo spin-density."""
    pd = PWDescriptor(ecut, gd)
    spin_density_G = pd.fft(spin_density_R)
    G_Gv = pd.get_reciprocal_vectors()
    eiGR_aG = np.exp(-1j * spos_ac.dot(gd.cell_cv).dot(G_Gv.T))

    W1_a = pd.integrate(spin_density_G, eiGR_aG) / gd.dv * (2 / 3)

    spin_density_G[0] = 0.0
    G2_G = pd.G2_qG[0].copy()
    G2_G[0] = 1.0
    spin_density_G /= G2_G

    W_vva = np.empty((3, 3, len(spos_ac)))
    for v1 in range(3):
        for v2 in range(3):
            W_a = pd.integrate(G_Gv[:, v1] * G_Gv[:, v2] * spin_density_G,
                               eiGR_aG)
            W_vva[v1, v2] = -W_a / gd.dv

    W_a = np.trace(W_vva) / 3
    for v in range(3):
        W_vva[v, v] -= W_a
        W_vva[v, v] += W1_a

    return W_vva.transpose((2, 0, 1))


# Normalization constants for xy, yz, 3z^2-r^2, xz, x^2-y^2:
Y2_m = (np.array([15 / 4,
                  15 / 4,
                  5 / 16,
                  15 / 4,
                  15 / 16]) / pi)**0.5
# Second derivatives:
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
                    [0, -2, 0],
                    [0, 0, 0]]])


def paw_correction(spin_density_ii: Array2D,
                   setup: Setup) -> Array2D:
    """Corrections from 1-center expansions of spin-density."""
    # Spherical part:
    D0_jj = expand(spin_density_ii, setup.l_j, 0)[0]

    phit_jg = np.array(setup.data.phit_jg)
    phi_jg = np.array(setup.data.phi_jg)

    rgd = setup.rgd

    # Spin-density from pseudo density:
    nt0 = phit_jg[:, 0].dot(D0_jj).dot(phit_jg[:, 0]) / (4 * pi)**0.5

    # All-electron contribution diveges as r^-beta and must be integrated
    # over a small region of size rT:
    n0_g = np.einsum('ab, ag, bg -> g', D0_jj, phi_jg, phi_jg) / (4 * pi)**0.5
    # Velocity of light in atomic units:
    c = 2 * units._hplanck / (units._mu0 * units._c * units._e**2)
    beta = 2 * (1 - (1 - (setup.Z / c)**2)**0.5)
    rT = setup.Z / c**2
    n0 = integrate(n0_g, rgd, rT, beta)

    W1 = (n0 - nt0) * 2 / 3  # isotropic result

    # Now the anisotropic part from the l=2 part of the density:
    D2_mjj = expand(spin_density_ii, setup.l_j, 2)
    dn2_mg = np.einsum('mab, ag, bg -> mg', D2_mjj, phi_jg, phi_jg)
    dn2_mg -= np.einsum('mab, ag, bg -> mg', D2_mjj, phit_jg, phit_jg)
    A_m = dn2_mg[:, 1:].dot(rgd.dr_g[1:] / rgd.r_g[1:]) / 5
    A_m *= Y2_m
    W_vv = Y2_mvv.T.dot(A_m)
    W = np.trace(W_vv) / 3
    for v in range(3):
        W_vv[v, v] -= W
        W_vv[v, v] += W1

    return W_vv


def expand(D_ii: Array2D,
           l_j: List[int],
           l: int) -> Array3D:
    """Get expansion coefficients."""
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


def delta(r: float, rT: float) -> float:
    """Extended delta function."""
    return 2 / rT / (1 + 2 * r / rT)**2


def integrate(n0_g: Array1D,
              rgd: RadialGridDescriptor,
              rT: float,
              beta: float) -> float:
    """Take care of r^-beta divergence."""
    r_g = rgd.r_g
    a_i = np.polyfit(r_g[1:5], n0_g[1:5] * r_g[1:5]**beta, 3)
    r4 = r_g[4]
    dr = rT / 50
    n = max(int(r4 / dr / 2) * 2 + 1, 3)
    r_j = np.linspace(0, r4, n)

    b_i = np.arange(3, -1, -1) + 1 - beta
    d0 = delta(0, rT)
    d1 = -8 / rT**2
    n0 = a_i.dot(d0 * r4**b_i / b_i + d1 * r4**(b_i + 1) / (b_i + 1))

    d_j = delta(r_j, rT) - (d0 + d1 * r_j)

    head_j = d_j * np.polyval(a_i, r_j)
    head_j[1:] *= r_j[1:]**-beta
    n0 += simps(head_j, r_j)

    tail_g = n0_g[4:] * delta(r_g[4:], rT)
    n0 += simps(tail_g, r_g[4:], even='first')

    return n0


# from https://en.wikipedia.org/wiki/Gyromagnetic_ratio
gyromagnetic_ratios = {'H': 42.6,
                       'O': -5.8}


def main(argv: List[str] = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.hyperfine',
        description='Calculate hyperfine parameters.')
    add = parser.add_argument
    add('file', metavar='input-file',
        help='GPW-file (with or without wave functions).')
    add('-g', '--gyromagnetic-ratios',
        help='Gyromagnetic ratios (in MHz/T).')
    if hasattr(parser, 'parse_intermixed_args'):
        args = parser.parse_intermixed_args(argv)
    else:
        args = parser.parse_args(argv)

    calc = GPAW(args.file)
    atoms = calc.get_atoms()

    ratios = gyromagnetic_ratios.copy()
    if args.gyromagnetic_ratios:
        for symbol, value in (part.split(':')
                              for part in args.gyromagnetic_ratios.split(',')):
            ratios[symbol] = float(value)

    A_avv = hyperfine_parameters(calc)

    print('Isotropic and anisotropic hyperfine coupling paramters in MHz/T:\n')
    print('  atom  magmom      ',
          '       '.join(['iso', 'xx', 'yy', 'zz', 'yz', 'xz', 'xy']))
    symbols = atoms.symbols
    magmoms = atoms.get_magnetic_moments()
    used = {}
    for a, A_vv in enumerate(A_avv):
        symbol = symbols[a]
        magmom = magmoms[a]
        ratio = ratios.get(symbol, 1.0)
        used[symbol] = ratio
        A_vv *= ratio * 1450 / 8.538  # ?????????????????????????????
        A = np.trace(A_vv) / 3
        print(f'{a:3} {symbol:>2}  {magmom:6.3f}',
              ''.join(f'{x:9.2f}' for x in
                      [A,
                       A_vv[0, 0] - A, A_vv[1, 1] - A, A_vv[2, 2] - A,
                       A_vv[1, 2], A_vv[0, 2], A_vv[0, 1]]))

    magmom = atoms.get_magnetic_moment()
    print(f'\nTotal magnetic moment: {magmom:.3f}')
    print('\nGyromagnetic ratios used:')
    for symbo, ratio in used.items():
        print(f'{symbol:2} {ratio:10.3f} MHz/T')


if __name__ == '__main__':
    # import ase.units as u
    # print(u._mu0 * u._hbar**2 * u._e**2 * 2.000 * 5.5 /
    #      (6 * np.pi * (u.Bohr * 1e-10)**3 * u._me * u._mp * u._e))
    main()
