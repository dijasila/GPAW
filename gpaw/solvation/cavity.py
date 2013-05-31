from gpaw.utilities.tools import coordinates
from ase.units import Bohr
import numpy as np


class BaseCavityDensity:
    name = 'unnamed'

    def __init__(self):
        self.gd = None

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def update_atoms(self, atoms):
        pass

    def get_rho_drho(self, nt_g):
        """Return density and derivative in 1 / Bohr ** 3."""
        raise NotImplementedError

    def get_atomic_position_derivative(self, index):
        raise NotImplementedError

    def print_parameters(self, text):
        """Print parameters using text function."""
        pass


class ElCavityDensity(BaseCavityDensity):
    name = 'Pseudo Electron Density'

    def __init__(self):
        BaseCavityDensity.__init__(self)

    def get_rho_drho(self, nt_g):
        return (nt_g, 1.)

    def get_atomic_position_derivative(self, index):
        return np.zeros(3)


class ExponentElCavityDensity(BaseCavityDensity):
    name = 'Pseudo Electron Density with Exponent'

    def __init__(self, beta):
        self.beta = beta
        BaseCavityDensity.__init__(self)

    def get_rho_drho(self, nt_g):
        n0 = 0.001
        nt_g = nt_g / n0
        if self.beta == .5:
            return (nt_g, 1. / n0)
        else:
            return (
                nt_g ** (2. * self.beta),
                2. * self.beta * nt_g ** (2. * self.beta - 1.) / n0
                )

    def get_atomic_position_derivative(self, index):
        return np.zeros(3)

    def print_parameters(self, text):
        text('beta: %11.6f' % (self.beta, ))


class SSS09CavityDensity(BaseCavityDensity):
    """Fake cavity density from van der Waals radii.

    Following Sanchez et al J. Chem. Phys. 131 (2009) 174108.

    """
    name = 'SSS09'

    def __init__(self, vdw_radii_map):
        """Initialize the van der Waals radii.

        Keyword arguments:
        vdw_radii_map -- maps atomic numbers to van der Waals radii in Angstrom

        """
        BaseCavityDensity.__init__(self)
        self.vdw_radii_map = vdw_radii_map
        self._rho = None
        self._positions = None
        self._vdw_radii = None

    def set_grid_descriptor(self, gd):
        BaseCavityDensity.set_grid_descriptor(self, gd)
        self._rho = gd.empty()

    def update_atoms(self, atoms):
        self._positions = atoms.positions / Bohr
        self._vdw_radii = [self.vdw_radii_map[n] for n in atoms.numbers]
        self._vdw_radii = np.array(self._vdw_radii) / Bohr
        self._rho.fill(.0)
        for p, r_vdw in zip(self._positions, self._vdw_radii):
            r = np.sqrt(coordinates(self.gd, origin=p)[1])
            self._rho += np.exp(r_vdw - r)

    def get_rho_drho(self, nt_g):
        return (self._rho, .0)

    def get_atomic_position_derivative(self, index):
        rvec, r = coordinates(
            self.gd,
            origin=self._positions[index]
            )
        r = np.sqrt(r)
        return np.exp(self._vdw_radii[index] - r) * rvec / r

    def print_parameters(self, text):
        if self._vdw_radii is not None:
            text('Van der Waals radii: %s' % (list(self._vdw_radii * Bohr), ))
        else:
            text('Van der Waals radii: %s' % ('not initialized', ))


class RepulsiveVdWCavityDensity(BaseCavityDensity):
    """Fake cavity density from van der Waals radii.

    Corresponds to an exponential repulsive potential
    taking the value 1.0 at the van der Waals radius

    """
    name = 'Repulsive van der Waals'

    def __init__(self, vdw_radii_map, r0):
        """Initialize the van der Waals radii.

        Keyword arguments:
        vdw_radii_map -- maps atomic numbers to van der Waals radii in Angstrom
        r0 -- scale in Ang

        """
        BaseCavityDensity.__init__(self)
        self.vdw_radii_map = vdw_radii_map
        self.r0 = r0
        self._rho = None
        self._positions = None
        self._vdw_radii = None

    def set_grid_descriptor(self, gd):
        BaseCavityDensity.set_grid_descriptor(self, gd)
        self._rho = gd.empty()

    def update_atoms(self, atoms):
        r0 = self.r0 / Bohr
        self._positions = atoms.positions / Bohr
        self._vdw_radii = [self.vdw_radii_map[n] for n in atoms.numbers]
        self._vdw_radii = np.array(self._vdw_radii) / Bohr
        self._rho.fill(.0)
        for p, r_vdw in zip(self._positions, self._vdw_radii):
            r = np.sqrt(coordinates(self.gd, origin=p)[1])
            self._rho += np.exp((r_vdw - r) / r0)

    def get_rho_drho(self, nt_g):
        return (self._rho, .0)

    def get_atomic_position_derivative(self, index):
        r0 = self.r0 / Bohr
        rvec, r = coordinates(
            self.gd,
            origin=self._positions[index]
            )
        r = np.sqrt(r)
        return np.exp((self._vdw_radii[index] - r) / r0) * rvec / r / r0

    def print_parameters(self, text):
        if self._vdw_radii is not None:
            text('Van der Waals radii: %s' % (list(self._vdw_radii * Bohr), ))
        else:
            text('Van der Waals radii: %s' % ('not initialized', ))
        text('r0                 : %11.6f' % (self.r0, ))


class BaseSmoothedStep:
    name = 'unnamed'

    def __init__(self):
        self.gd = None

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def get_theta_dtheta(self, rho):
        """Return smoothed step function and derivative from cavity density."""
        raise NotImplementedError

    def print_paramters(self, text):
        """Print parameters using text function."""


class FG02SmoothedStep(BaseSmoothedStep):
    """Smoothed step function.

    Following J. Fattebert, and F. Gygi, J Comput Chem 23: 662-666, 2002.

    """
    name = 'FG02'

    def __init__(self, rho0, beta, rhomin=0.0001 / Bohr ** 3):
        """Initialize parameters.

        Keyword arguments:
        rho0 -- cutoff density value in 1 / Ang ** 3
        beta -- exponent
        rhomin -- theta(rho < rhomin) == 0

        """
        BaseSmoothedStep.__init__(self)
        self.rho0 = rho0
        self.beta = beta
        self.rhomin = rhomin

    def get_theta_dtheta(self, rho):
        inside = rho >= self.rhomin * Bohr ** 3
        rho0 = self.rho0 * Bohr ** 3
        x = (rho[inside] / rho0) ** (2. * self.beta)
        theta = np.zeros_like(rho)
        dtheta = np.zeros_like(rho)
        theta[inside] = x / (1. + x)
        dtheta[inside] = 2. * self.beta * x / (rho[inside] * (1. + x) ** 2)
        return (theta, dtheta)

    def print_parameters(self, text):
        text('rho0  : %11.6f' % (self.rho0, ))
        text('beta  : %11.6f' % (self.beta, ))
        text('rhomin: %11.6f' % (self.rhomin, ))


class ADM12SmoothedStep(BaseSmoothedStep):
    """Smoothed step function.

    Following O. Andreussi, I. Dabo, and N. Marzari,
    J. Chem. Phys. 136, 064102 (2012).

    """
    name = 'ADM12'

    def __init__(self, rhomin, rhomax, epsinf):
        """Initialize parameters.

        Keyword arguments:
        rhomin -- mininum cutoff density value in 1 / Ang ** 3
        rhomin -- maximum cutoff density value in 1 / Ang ** 3
        epsinf -- dielectric constant outside the cavity

        """
        BaseSmoothedStep.__init__(self)
        self.rhomin = rhomin
        self.rhomax = rhomax
        self.epsinf = epsinf

    def get_theta_dtheta(self, rho):
        eps = self.epsinf
        theta = self.gd.empty()
        dtheta = self.gd.zeros()
        inside = rho > self.rhomax * Bohr ** 3
        outside = rho < self.rhomin * Bohr ** 3
        transition = np.logical_not(
            np.logical_or(inside, outside)
            )
        theta[inside] = 1.
        theta[outside] = .0
        t, dt = self._get_t_dt(np.log(rho[transition]))
        if eps == 1.0:
            # lim_{eps -> 1} (eps - eps ** t) / (eps - 1) = 1 - t
            theta[transition] = 1. - t
            dtheta[transition] = -dt / rho[transition]
        else:
            eps_to_t = eps ** t
            theta[transition] = (eps - eps_to_t) / (eps - 1.)
            dtheta[transition] = -(eps_to_t * np.log(eps) * dt) / \
                (rho[transition] * (eps - 1.))
        return (theta, dtheta)

    def _get_t_dt(self, x):
        lnmax = np.log(self.rhomax * Bohr ** 3)
        lnmin = np.log(self.rhomin * Bohr ** 3)
        twopi = 2. * np.pi
        arg = twopi * (lnmax - x) / (lnmax - lnmin)
        t = (arg - np.sin(arg)) / twopi
        dt = -2. * np.sin(arg / 2.) ** 2 / (lnmax - lnmin)
        return (t, dt)

    def print_parameters(self, text):
        text('rhomin: %11.6f' % (self.rhomin, ))
        text('rhomax: %11.6f' % (self.rhomax, ))
        text('epsinf: %11.6f' % (self.epsinf, ))


class BoltzmannSmoothedStep(BaseSmoothedStep):
    """Smoothed step function.

    1 - exp(-rho / rho0)

    """
    name = 'Boltzmann'

    def __init__(self, rho0):
        """Initialize parameters.

        Keyword arguments:
        rho0 -- cutoff density value in 1 / Ang ** 3

        """
        BaseSmoothedStep.__init__(self)
        self.rho0 = rho0

    def print_parameters(self, text):
        text('rho0: %11.6f' % (self.rho0, ))

    def get_theta_dtheta(self, rho):
        rho0 = self.rho0 * Bohr ** 3
        g = np.exp(-rho / rho0)
        theta = 1. - g
        dtheta = g / rho0
        return (theta, dtheta)
