from gpaw.utilities.tools import coordinates
from ase.units import Bohr
import numpy as np


def get_pbc_positions(atoms, r_max):
    """
    returns dict mapping atom index to positions in Bohr

    With periodic boundary conditions, it also includes neighbouring
    cells up to a distance of r_max (in Bohr).
    """
    # code snippet taken from ase/calculators/vdwcorrection.py
    pbc_c = atoms.get_pbc()
    cell_cv = atoms.get_cell() / Bohr
    Rcell_c = np.sqrt(np.sum(cell_cv ** 2, axis=1))
    ncells_c = np.ceil(np.where(pbc_c, 1. + r_max / Rcell_c, 1)).astype(int)
    pos_aav = {}
    # loop over all atoms in the cell (and neighbour cells for PBC)
    for index1, atom in enumerate(atoms):
        pos = atom.position / Bohr
        pos_aav[index1] = np.empty((np.prod(ncells_c * 2 - 1), 3))
        # loops over neighbour cells
        index2 = 0
        for ix in xrange(-ncells_c[0] + 1, ncells_c[0]):
            for iy in xrange(-ncells_c[1] + 1, ncells_c[1]):
                for iz in xrange(-ncells_c[2] + 1, ncells_c[2]):
                    i_c = np.array([ix, iy, iz])
                    pos_aav[index1][index2, :] = pos + np.dot(i_c, cell_cv)
                    index2 += 1
    return pos_aav


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

    def __init__(self, beta, n0):
        self.beta = beta
        self.n0 = n0
        BaseCavityDensity.__init__(self)

    def get_rho_drho(self, nt_g):
        n0 = self.n0 * Bohr ** 3
        n_scaled = nt_g / n0
        n_scaled[n_scaled < .0] = .0
        return (
            n_scaled ** self.beta,
            self.beta * n_scaled ** (self.beta - 1) / n0
            )

    def get_atomic_position_derivative(self, index):
        return np.zeros(3)

    def print_parameters(self, text):
        text('beta: %11.6f' % (self.beta, ))
        text('n0  : %11.6f' % (self.n0, ))


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
        if atoms.pbc.any():
            raise NotImplementedError(
                'Periodic boundary conditions are not yet implemented!'
                )
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
        if atoms.pbc.any():
            raise NotImplementedError(
                'Periodic boundary conditions are not yet implemented!'
                )
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


class Power12VdWCavityDensity(BaseCavityDensity):
    """Fake cavity density from van der Waals radii.

    Corresponds to an 1 / r ** 12 repulsive potential
    taking the value 1.0 at the van der Waals radius

    """
    name = 'Power12 van der Waals'

    def __init__(self, vdw_radii_map, r_max=10 * Bohr):
        BaseCavityDensity.__init__(self)
        self.vdw_radii_map = vdw_radii_map
        self.r_max = float(r_max)
        self._rho = None
        self._pos_aav = None
        self._vdw_radii = None

    def set_grid_descriptor(self, gd):
        BaseCavityDensity.set_grid_descriptor(self, gd)
        self._rho = gd.empty()
        self._r_vg = gd.get_grid_point_coordinates()

    def update_atoms(self, atoms):
        self._hack_non_pbc = not atoms.pbc.any()  # XXX remove after fix
        self._pos_aav = get_pbc_positions(atoms, self.r_max / Bohr)
        self._vdw_radii = [self.vdw_radii_map[n] for n in atoms.numbers]
        self._vdw_radii = np.array(self._vdw_radii) / Bohr
        self._rho.fill(.0)
        na = np.newaxis
        for index, pos_av in self._pos_aav.iteritems():
            r_vdw_12 = self._vdw_radii[index] ** 12
            for pos_v in pos_av:
                origin_vg = pos_v[:, na, na, na]
                r2_g = np.sum((self._r_vg - origin_vg) ** 2, axis=0)
                u = r_vdw_12 / r2_g ** 6
                self._rho += u
        self._rho[np.isnan(self._rho)] = np.inf

    def get_rho_drho(self, nt_g):
        return (self._rho, .0)

    def get_atomic_position_derivative(self, index):
        na = np.newaxis
        pos_av = self._pos_aav[index]
        r_vdw_12 = self._vdw_radii[index] ** 12
        if not self._hack_non_pbc:
            raise NotImplementedError(
                'Periodic boundary conditions are not yet implemented!'
                )
        # XXX hack for non PBC
        # XXX todo: implement for PBC
        pos_v = pos_av[0]
        origin_vg = pos_v[:, na, na, na]
        x_vg = self._r_vg - origin_vg
        r2_g = np.sum(x_vg ** 2, axis=0)
        mf_vg = 12. * r_vdw_12 / r2_g ** 7 * x_vg
        mf_vg[np.isnan(mf_vg)] = np.inf
        return mf_vg

    def print_parameters(self, text):
        if self._vdw_radii is not None:
            text('Van der Waals radii: %s' % (list(self._vdw_radii * Bohr), ))
        else:
            text('Van der Waals radii: %s' % ('not initialized', ))


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
