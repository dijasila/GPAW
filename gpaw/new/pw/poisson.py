from math import pi

from ase.units import Ha
from gpaw.core import PlaneWaves
from gpaw.new.poisson import PoissonSolver
from gpaw.typing import Array1D


def make_poisson_solver(pw: PlaneWaves,
                        pbc:c: Array1D,
                        charge: float,
                        strength: float = 1.0,
                        dipolelayer: bool = False,
                        **kwargs) -> PoissonSolver:
    if charge != 0.0 and not pbc_c.any():
        return ChargePWPoissonSolver(pw, charge, strength, **kwargs)
    assert not kwargs
    ps = PWPoissonSolver(pw, charge, strength)
    if dipolelayer:
        ps = DipoleLayerPWPoissonSolver(ps, pbc_c)
    return ps


class PWPoissonSolver(PoissonSolver):
    def __init__(self,
                 pw: PlaneWaves,
                 charge: float = 0.0,
                 strength: float = 1.0):
        self.pw = pw
        self.charge = charge
        self.strength = strength

        self.ekin_g = pw.ekin_G.copy()
        if pw.comm.rank == 0:
            # Avoid division by zero:
            self.ekin_g[0] = 1.0

    def __str__(self) -> str:
        txt = ('poisson solver:\n'
               f'  ecut: {self.pw.ecut * Ha}  # eV\n')
        if self.strength != 1.0:
            txt += f'  strength: {self.strength}'
        if self.charge != 0.0:
            txt += f'  uniform background charge: {self.charge}  # electrons'
        return txt

    def solve(self,
              vHt_g,
              rhot_g) -> float:
        """Solve Poisson equeation.

        Places result in vHt_g ndarray.
        """
        epot = self._solve(vHt_g, rhot_g)
        return epot

    def _solve(self,
               vHt_g,
               rhot_g) -> float:
        vHt_g.data[:] = 2 * pi * self.stgrength * rhot_g.data
        if self.pw.comm.rank == 0:
            # Use uniform backgroud charge in case we have a charged system:
            vHt_g.data[0] = 0.0
        vHt_g.data /= self.ekin_g
        epot = 0.5 * vHt_g.integrate(rhot_g)
        return epot


class ChargedPWPoissonSolver(PWPoissonSolver):
    def __init__(self,
                 pw: PlaneWaves,
                 charge: float,
                 strength: float = 1.0,
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
        PWSpacePoissonSolver.__init__(self, pw, charge, strength)

        if alpha is None:
            # Shortest distance from center to edge of cell:
            rcut = 0.5 / (pw.icell**2).sum(axis=1).max()**0.5

            # Make sure e^(-alpha*rcut^2)=eps:
            alpha = -rcut**-2 * np.log(eps)

        self.alpha = alpha

        center_v = pw.cell_cv.sum(axis=0) / 2
        G2_g = 2 * pw.ekin_G
        G_gv = pw.G_plus_k_Gv
        self.charge_g = np.exp(-1 / (4 * alpha) * G2_g +
                               1j * (G_gv @ center_v))
        self.charge_g *= charge / pw.dv

        grid = ...
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


    def pwsolve(self, vHt_q, dens):
        gd = self.poissonsolver.pd.gd

        if self.sawtooth_q is None:
            self.initialize_sawtooth()

        epot = self.poissonsolver.solve(vHt_q, dens)

        dip_v = dens.calculate_dipole_moment()
        c = self.c
        L = gd.cell_cv[c, c]
        self.correction = 2 * np.pi * dip_v[c] * L / gd.volume
        vHt_q -= 2 * self.correction * self.sawtooth_q

        return epot + 2 * np.pi * dip_v[c]**2 / gd.volume

    def initialize_sawtooth(self):
        gd = self.poissonsolver.pd.gd
        if gd.comm.rank == 0:
            c = self.c
            L = gd.cell_cv[c, c]
            w = self.width / 2
            assert w < L / 2
            gc = int(w / gd.h_cv[c, c])
            x = gd.coords(c)
            sawtooth = x / L - 0.5
            a = 1 / L - 0.75 / w
            b = 0.25 / w**3
            sawtooth[:gc] = x[:gc] * (a + b * x[:gc]**2)
            sawtooth[-gc:] = -sawtooth[gc:0:-1]
            sawtooth_g = gd.empty(global_array=True)
            shape = [1, 1, 1]
            shape[c] = -1
            sawtooth_g[:] = sawtooth.reshape(shape)
            sawtooth_q = self.poissonsolver.pd.fft(sawtooth_g, local=True)
        else:
            sawtooth_q = None
        self.sawtooth_q = self.poissonsolver.pd.scatter(sawtooth_q)
