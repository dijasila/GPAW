from math import pi, nan

# import numpy as np
# from ase.utils import seterr
# from scipy.special import erf

# from gpaw.typing import Array1D
from gpaw.core import PlaneWaves


class PoissonSolver:
    def solve(self,
              vHt,
              rhot) -> float:
        raise NotImplementedError


class PoissonSolverWrapper(PoissonSolver):
    def __init__(self, solver):
        self.description = solver.get_description()
        self.solver = solver

    def solve(self,
              vHt,
              rhot) -> float:
        self.solver.solve(vHt.data, rhot.data)
        return nan


class ReciprocalSpacePoissonSolver(PoissonSolver):
    def __init__(self,
                 pw: PlaneWaves,
                 charge: float = 0.0):
        self.pw = pw
        self.charge = charge

        self.ekin_g = pw.ekin_G.copy()
        if pw.comm.rank == 0:
            # Avoid division by zero:
            self.ekin_g[0] = 1.0

        self.description = (
            f'Uniform background charge: {self.charge:.3f} electrons')

    def solve(self,
              vHt_g,
              rhot_g) -> float:
        """Solve Poisson equeation.

        Places result in vHt_q ndarray.
        """
        epot = self._solve(vHt_g, rhot_g)
        return epot

    def _solve(self,
               vHt_g,
               rhot_g) -> float:
        vHt_g.data[:] = 2 * pi * rhot_g.data
        if self.pw.comm.rank == 0:
            # Use uniform backgroud charge in case we have a charged system:
            vHt_g.data[0] = 0.0
        vHt_g.data /= self.ekin_g
        epot = 0.5 * vHt_g.integrate(rhot_g)
        return epot


'''
class ChargedReciprocalSpacePoissonSolver(ReciprocalSpacePoissonSolver):
    def __init__(self,
                 pw: PlaneWaves,
                 charge: float,
                 alpha: float = None,
                 eps: float = 1e-5):
        """Reciprocal-space Poisson solver for charged molecules.

        * Add a compensating Guassian-shaped charge to the density
          in order to make the total charge neutral (placed in the
          middle of the unit cell

        * Solve Poisson equation.

        * Correct potential so that it has the correct 1/r
          asymptotic behavior

        * Correct energy to remove the artificial interaction with
          the compensation charge
        """
        ReciprocalSpacePoissonSolver.__init__(self, pd, charge)
        self.charge = charge

        if alpha is None:
            # Shortest distance from center to edge of cell:
            rcut = 0.5 / (pd.gd.icell_cv**2).sum(axis=1).max()**0.5

            # Make sure e^(-alpha*rcut^2)=eps:
            alpha = -rcut**-2 * np.log(eps)

        self.alpha = alpha

        center_v = pd.gd.cell_cv.sum(axis=0) / 2
        G2_q = pd.G2_qG[0]
        G_qv = pd.get_reciprocal_vectors()
        self.charge_q = np.exp(-1 / (4 * alpha) * G2_q +
                               1j * (G_qv @ center_v))
        self.charge_q *= charge / pd.gd.dv

        R_Rv = pd.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        d_R = ((R_Rv - center_v)**2).sum(axis=3)**0.5

        with seterr(invalid='ignore'):
            potential_R = erf(alpha**0.5 * d_R) / d_R
        if ((pd.gd.N_c % 2) == 0).all():
            R_c = pd.gd.N_c // 2
            if pd.gd.is_my_grid_point(R_c):
                potential_R[tuple(R_c - pd.gd.beg_c)] = (4 * alpha / pi)**0.5
        self.potential_q = charge * pd.fft(potential_R)

    def __str__(self):
        return ('Using Gaussian-shaped compensation charge: e^(-ar^2) '
                f'with a={self.alpha:.3f} bohr^-2')

    def _solve(self,
               vHt_q: Array1D,
               rhot_q: Array1D) -> float:
        neutral_q = rhot_q + self.charge_q
        if self.pd.gd.comm.rank == 0:
            error = neutral_q[0] * self.pd.gd.dv
            assert error.imag == 0.0, error
            assert abs(error.real) < 0.01, error
            neutral_q[0] = 0.0

        vHt_q[:] = 4 * pi * neutral_q
        vHt_q /= self.G2_q
        epot = 0.5 * self.pd.integrate(vHt_q, neutral_q)
        epot -= self.pd.integrate(self.potential_q, rhot_q)
        epot -= self.charge**2 * (self.alpha / 2 / pi)**0.5
        vHt_q -= self.potential_q
        return epot
'''
