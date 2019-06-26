import numpy as np
from ase.units import Bohr, Hartree

from gpaw.utilities import h2gpts
from gpaw.fftw import get_efficient_fft_size
from gpaw.wavefunctions.fd import FD


def get_number_of_grid_points(cell_cv, h=None, mode=None, realspace=None,
                              symmetry=None):
    if mode is None:
        mode = FD()

    if realspace is None:
        realspace = mode.name != 'pw'

    if h is None:
        if mode.name == 'pw':
            h = np.pi / (4 * mode.ecut)**0.5
        elif mode.name == 'lcao' and not realspace:
            h = np.pi / (4 * 340 / Hartree)**0.5
        else:
            h = 0.2 / Bohr

    if realspace or mode.name == 'fd':
        N_c = h2gpts(h, cell_cv, 4)
    else:
        N_c = h2gpts(h, cell_cv, 1)
        if symmetry is None:
            N_c = np.array([get_efficient_fft_size(N) for N in N_c])
        else:
            N_c = np.array([get_efficient_fft_size(N, n)
                            for N, n in zip(N_c, symmetry.gcd_c)])

    if symmetry is not None:
        ok = symmetry.check_grid(N_c)
        if not ok:
            # Choose more symmetric number of grid points:
            S_cc = symmetry.op_scc.any(axis=0)
            S_cc = S_cc + S_cc.T
            if S_cc[0, 1] and S_cc[1, 2]:
                assert S_cc[1, 2]
                gcd = symmetry.gcd_c.max()
                N = get_efficient_fft_size(N_c.max(), gcd)
                N_c = np.array([N, N, N])
            elif S_cc[0, 1]:
                N = N_c[:2].max()
                gcd = symmetry.gcd_c[:2].max()
                N_c[:2] = get_efficient_fft_size(N, gcd)
                assert not S_cc[0, 2]
                assert not S_cc[1, 2]
            elif S_cc[1, 2]:
                N = N_c[1:].max()
                gcd = symmetry.gcd_c[1:].max()
                N_c[1:] = get_efficient_fft_size(N, gcd)
                assert not S_cc[0, 1]
                assert not S_cc[0, 2]
            elif S_cc[0, 2]:
                N = N_c[::2].max()
                gcd = symmetry.gcd_c[::2].max()
                N_c[::2] = get_efficient_fft_size(N, gcd)
                assert not S_cc[0, 1]
                assert not S_cc[1, 2]
            else:
                1 / 0
            ok = symmetry.check_grid(N_c)
            assert ok, N_c

    return N_c
