"""XC density kernels for response function calculations."""

import numpy as np

from gpaw.response.fxc_kernels import AdiabaticSusceptibilityFXC


def get_density_xc_kernel(pd, chi0, functional='ALDA',
                          rshelmax=-1, rshewmin=None,
                          chi0_wGG=None):
    """Density-density xc kernels.
    Factory function that calls the relevant functions below."""

    calc = chi0.calc
    fd = chi0.fd
    nspins = len(chi0.gs.nt_sG)
    assert nspins == 1

    if functional[0] == 'A':
        # Standard adiabatic kernel
        print('Calculating %s kernel' % functional, file=fd)
        Kcalc = AdiabaticSusceptibilityFXC(calc, functional,
                                           world=chi0.world, txt=fd,
                                           timer=chi0.timer,
                                           rshelmax=rshelmax,
                                           rshewmin=rshewmin)
        Kxc_GG = Kcalc('00', pd)
        if pd.kd.gamma:
            Kxc_GG[0, :] = 0.0
            Kxc_GG[:, 0] = 0.0
        Kxc_sGG = np.array([Kxc_GG])
    elif functional[:2] == 'LR':
        print('Calculating LR kernel with alpha = %s' % functional[2:],
              file=fd)
        Kxc_sGG = calculate_lr_kernel(pd, calc, alpha=float(functional[2:]))
    elif functional == 'Bootstrap':
        print('Calculating Bootstrap kernel', file=fd)
        Kxc_sGG = get_bootstrap_kernel(pd, chi0, chi0_wGG, fd)
    else:
        raise ValueError('Invalid functional for the density-density '
                         'xc kernel:', functional)

    return Kxc_sGG[0]


def calculate_lr_kernel(pd, calc, alpha=0.2):
    """Long range kernel: fxc = \alpha / |q+G|^2"""

    assert pd.kd.gamma

    f_G = np.zeros(len(pd.G2_qG[0]))
    f_G[0] = -alpha
    f_G[1:] = -alpha / pd.G2_qG[0][1:]

    return np.array([np.diag(f_G)])


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

    Kxc_GG = np.zeros((nG, nG), dtype=complex)
    dminv_GG = np.zeros((nG, nG), dtype=complex)

    for iscf in range(120):
        dminvold_GG = dminv_GG.copy()
        Kxc_GG = K_GG + Kxc_GG

        chi_GG = np.dot(np.linalg.inv(np.eye(nG, nG)
                                      - np.dot(chi0_GG, Kxc_GG)), chi0_GG)
        dminv_GG = np.eye(nG, nG) + np.dot(K_GG, chi_GG)

        alpha = dminv_GG[0, 0] / (K_GG[0, 0] * chi0_GG[0, 0])
        Kxc_GG = alpha * K_GG
        print(iscf, 'alpha =', alpha, file=fd)
        error = np.abs(dminvold_GG - dminv_GG).sum()
        if np.sum(error) < 0.1:
            print('Self consistent fxc finished in %d iterations !' % iscf,
                  file=fd)
            break
        if iscf > 100:
            print('Too many fxc scf steps !', file=fd)

    return np.array([Kxc_GG])
