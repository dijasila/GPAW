"""This module calculates XC kernels for response function calculations.
"""

import numpy as np

from ase.utils.timing import Timer
from ase.units import Ha

from gpaw.io.tar import Reader
from gpaw.response.kxc import AdiabaticSusceptibilityFXC
from gpaw.response.tms import get_goldstone_scaling


def get_xc_kernel(pd, chi0, functional='ALDA', kernel='density',
                  rshelmax=-1, rshewmin=None,
                  chi0_wGG=None,
                  fxc_scaling=None,
                  density_cut=None,
                  spinpol_cut=None):
    """
    Factory function that calls the relevant functions below
    """

    if kernel == 'density':
        return get_density_xc_kernel(pd, chi0, functional=functional,
                                     rshelmax=rshelmax, rshewmin=rshewmin,
                                     chi0_wGG=chi0_wGG,
                                     density_cut=density_cut)
    elif kernel in ['+-', '-+']:
        # Currently only collinear adiabatic xc kernels are implemented
        # for which the +- and -+ kernels are the same
        return get_transverse_xc_kernel(pd, chi0, functional=functional,
                                        rshelmax=rshelmax, rshewmin=rshewmin,
                                        chi0_wGG=chi0_wGG,
                                        fxc_scaling=fxc_scaling,
                                        density_cut=density_cut,
                                        spinpol_cut=spinpol_cut)
    else:
        raise ValueError('%s kernels not implemented' % kernel)


def get_density_xc_kernel(pd, chi0, functional='ALDA',
                          rshelmax=-1, rshewmin=None,
                          chi0_wGG=None,
                          density_cut=None):
    """
    Density-density xc kernels
    Factory function that calls the relevant functions below
    """

    calc = chi0.calc
    fd = chi0.fd
    nspins = len(calc.density.nt_sG)
    assert nspins == 1

    if functional[0] == 'A':
        # Standard adiabatic kernel
        print('Calculating %s kernel' % functional, file=fd)
        Kcalc = AdiabaticSusceptibilityFXC(calc, functional,
                                           world=chi0.world, txt=fd,
                                           timer=chi0.timer,
                                           rshelmax=rshelmax,
                                           rshewmin=rshewmin,
                                           density_cut=density_cut)
        Kxc_GG = Kcalc('00', pd)
        if pd.kd.gamma:
            Kxc_GG[0, :] = 0.0
            Kxc_GG[:, 0] = 0.0
        Kxc_sGG = np.array([Kxc_GG])
    elif functional == 'DM':
        print('Calculating DM kernel', file=fd)
        Kxc_sGG = calculate_dm_kernel(pd, calc)
    elif functional == 'Bootstrap':
        print('Calculating Bootstrap kernel', file=fd)
        Kxc_sGG = get_bootstrap_kernel(pd, chi0, chi0_wGG, fd)
    else:
        raise ValueError('Invalid functional for the density-density '
                         'fxc kernel:', functional)

    return Kxc_sGG[0]


def get_transverse_xc_kernel(pd, chi0, functional='ALDA_x',
                             rshelmax=-1, rshewmin=None,
                             chi0_wGG=None,
                             fxc_scaling=None,
                             density_cut=None,
                             spinpol_cut=None):
    """ +-/-+ xc kernels
    Currently only collinear ALDA kernels are implemented
    Factory function that calls the relevant functions below
    """

    calc = chi0.calc
    fd = chi0.fd
    nspins = len(calc.density.nt_sG)
    assert nspins == 2

    if functional in ['ALDA_x', 'ALDA_X', 'ALDA']:
        # Adiabatic kernel
        print("Calculating transverse %s kernel" % functional, file=fd)
        Kcalc = AdiabaticSusceptibilityFXC(calc, functional,
                                           world=chi0.world, txt=fd,
                                           timer=chi0.timer,
                                           rshelmax=rshelmax,
                                           rshewmin=rshewmin,
                                           density_cut=density_cut,
                                           spinpol_cut=spinpol_cut)
    else:
        raise ValueError("%s spin kernel not implemented" % functional)

    Kxc_GG = Kcalc('+-', pd)

    if fxc_scaling is not None:
        assert isinstance(fxc_scaling[0], bool)
        if fxc_scaling[0]:
            if fxc_scaling[1] is None:
                assert pd.kd.gamma
                print('Finding rescaling of kernel to fulfill the '
                      'Goldstone theorem', file=chi0.fd)
                mode = fxc_scaling[2]
                assert mode in ['fm', 'afm']
                fxc_scaling[1] = get_goldstone_scaling(mode,
                                                       chi0.wd.omega_w,
                                                       -chi0_wGG, Kxc_GG,
                                                       world=chi0.world)

            assert isinstance(fxc_scaling[1], float)
            Kxc_GG *= fxc_scaling[1]

    return Kxc_GG


def calculate_dm_kernel(pd, calc):
    """Density matrix kernel"""

    assert pd.kd.gamma

    nv = calc.wfs.setups.nvalence
    psit_nG = np.array([calc.wfs.kpt_u[0].psit_nG[n]
                        for n in range(4 * nv)])
    vol = np.linalg.det(calc.wfs.gd.cell_cv)
    Ng = np.prod(calc.wfs.gd.N_c)
    rho_GG = np.dot(psit_nG.conj().T, psit_nG) * vol / Ng**2

    maxG2 = np.max(pd.G2_qG[0])
    cut_G = np.arange(calc.wfs.pd.ngmax)[calc.wfs.pd.G2_qG[0] <= maxG2]

    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0

    Kxc_GG = np.diagflat(4 * np.pi / G_G**2)
    Kxc_GG = np.dot(Kxc_GG, rho_GG.take(cut_G, 0).take(cut_G, 1))
    Kxc_GG -= 4 * np.pi * np.diagflat(1.0 / G_G**2)

    return np.array([Kxc_GG])


def get_bootstrap_kernel(pd, chi0, chi0_wGG, fd):
    """ Bootstrap kernel (see below) """

    if chi0.world.rank == 0:
        chi0_GG = chi0_wGG[0]
        if chi0.world.size > 1:
            # If size == 1, chi0_GG is not contiguous, and broadcast()
            # will fail in debug mode.  So we skip it until someone
            # takes a closer look.
            chi0.world.broadcast(chi0_GG, 0)
    else:
        nG = pd.ngmax
        chi0_GG = np.zeros((nG, nG), complex)
        chi0.world.broadcast(chi0_GG, 0)

    return calculate_bootstrap_kernel(pd, chi0_GG, fd)


def calculate_bootstrap_kernel(pd, chi0_GG, fd):
    """Bootstrap kernel PRL 107, 186401"""

    if pd.kd.gamma:
        v_G = np.zeros(len(pd.G2_qG[0]))
        v_G[0] = 4 * np.pi
        v_G[1:] = 4 * np.pi / pd.G2_qG[0][1:]
    else:
        v_G = 4 * np.pi / pd.G2_qG[0]

    nG = len(v_G)
    K_GG = np.diag(v_G)

    fxc_GG = np.zeros((nG, nG), dtype=complex)
    dminv_GG = np.zeros((nG, nG), dtype=complex)

    for iscf in range(120):
        dminvold_GG = dminv_GG.copy()
        Kxc_GG = K_GG + fxc_GG

        chi_GG = np.dot(np.linalg.inv(np.eye(nG, nG)
                                      - np.dot(chi0_GG, Kxc_GG)), chi0_GG)
        dminv_GG = np.eye(nG, nG) + np.dot(K_GG, chi_GG)

        alpha = dminv_GG[0, 0] / (K_GG[0, 0] * chi0_GG[0, 0])
        fxc_GG = alpha * K_GG
        print(iscf, 'alpha =', alpha, file=fd)
        error = np.abs(dminvold_GG - dminv_GG).sum()
        if np.sum(error) < 0.1:
            print('Self consistent fxc finished in %d iterations !' % iscf,
                  file=fd)
            break
        if iscf > 100:
            print('Too many fxc scf steps !', file=fd)

    return np.array([fxc_GG])
