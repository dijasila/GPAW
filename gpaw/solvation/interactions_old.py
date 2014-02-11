from ase.units import Bohr, Hartree
from gpaw.fd_operators import Gradient
import numpy as np


class BaseInteraction:
    """Base class for additional solvent solute interactions."""

    name = 'unnamed'
    subscript = 'unnamed'

    def __init__(self):
        self.hamiltonian = None

    def init(self, hamiltonian):
        """Perform inexpensive initialization."""
        self.hamiltonian = hamiltonian

    def set_atoms(self, atoms):
        """Handle changes to atoms."""
        pass

    def allocate(self):
        """Perform expensive initialization."""
        pass

    def update_pseudo_potential(self, density):
        """Update the Kohn-Sham potential of the Hamiltonian.

        Return interaction energy in Hartree.

        """
        raise NotImplementedError

    def update_forces(self, dens, F_av):
        """Add interaction forces to F_av in Hartree / Bohr."""
        raise NotImplementedError

    def print_parameters(self, text):
        """Print parameters using text function."""
        pass


class QuantumVolumeInteraction(BaseInteraction):
    """Interaction proportional to quantum volume.

    Following O. Andreussi, I. Dabo, and N. Marzari,
    J. Chem. Phys. 136, 064102 (2012).

    """

    name = 'Quantum Volume'
    subscript = 'vol'

    def __init__(self, pressure):
        """Initialize parameters.

        Keyword arguments:
        pressure -- pressure in eV / Ang ** 3

        """
        BaseInteraction.__init__(self)
        self.pressure = pressure

    def update_pseudo_potential(self, density):
        pressure = self.pressure * Bohr ** 3 / Hartree
        vks = pressure * self.hamiltonian.dtheta * self.hamiltonian.drho
        for vt_g in self.hamiltonian.vt_sg[:self.hamiltonian.nspins]:
            vt_g += vks
        return pressure * self.hamiltonian.finegd.integrate(
            self.hamiltonian.theta,
            global_integral=False
            )

    def update_forces(self, dens, F_av):
        pressure = self.pressure * Bohr ** 3 / Hartree
        cavdens = self.hamiltonian.cavdens
        dtheta = self.hamiltonian.dtheta
        for a, fa in enumerate(F_av):
            drho_dRav = cavdens.get_atomic_position_derivative(a)
            for v in (0, 1, 2):
                fa[v] -= pressure * self.hamiltonian.finegd.integrate(
                    dtheta * drho_dRav[v],
                    global_integral=False
                    )

    def print_parameters(self, text):
        text('pressure: %11.6f' % (self.pressure, ))


class QuantumSurfaceInteraction(BaseInteraction):
    """Interaction proportional to quantum surface.

    Following O. Andreussi, I. Dabo, and N. Marzari,
    J. Chem. Phys. 136, 064102 (2012).

    """

    name = 'Quantum Surface'
    subscript = 'surf'

    def __init__(self, surface_tension, delta):
        """Initialize parameters.

        Keyword arguments:
        surface_tension -- surface tension in eV / Ang ** 2
        delta -- isosurface spacing in 1 / Ang ** 3

        """
        BaseInteraction.__init__(self)
        self.surface_tension = surface_tension
        self.delta = delta
        self.gradops = None

    def init(self, hamiltonian):
        BaseInteraction.init(self, hamiltonian)
        self.gradops = [
            Gradient(hamiltonian.finegd, i, 1.0, hamiltonian.poisson.nn) \
                for i in (0, 1, 2)
            ]

    def update_pseudo_potential(self, density):
        st = self.surface_tension * Bohr ** 2 / Hartree
        delta = self.delta * Bohr ** 3
        step = self.hamiltonian.smoothedstep
        rho = self.hamiltonian.rho.copy()
        forbidden = rho < 1e-12  # XXX check
        rho[forbidden] = 1e-12
        grho = self._gradient(rho)
        ngrho = np.sqrt(np.square(grho).sum(0))
        divarg = (grho[0] / ngrho, grho[1] / ngrho, grho[2] / ngrho)
        for c in divarg:
            c[forbidden] = .0
        div = self._div(divarg)
        f1 = step.get_theta_dtheta(rho + delta / 2.)[0]
        f1 -= step.get_theta_dtheta(rho - delta / 2.)[0]
        f1 *= st / delta
        vks = -f1 * div * self.hamiltonian.drho
        for vt_g in self.hamiltonian.vt_sg[:self.hamiltonian.nspins]:
            vt_g += vks
        return self.hamiltonian.finegd.integrate(
            f1 * ngrho,
            global_integral=False
            )

    def update_forces(self, dens, F_av):
        st = self.surface_tension * Bohr ** 2 / Hartree
        delta = self.delta * Bohr ** 3
        rho = self.hamiltonian.rho
        grho = self._gradient(rho)
        ngrho = np.sqrt(np.square(grho).sum(0))
        nngrho = (grho[0] / ngrho, grho[1] / ngrho, grho[2] / ngrho)
        cavdens = self.hamiltonian.cavdens
        step = self.hamiltonian.smoothedstep
        tp, dtp = step.get_theta_dtheta(rho + delta / 2.)
        tm, dtm = step.get_theta_dtheta(rho - delta / 2.)
        tdiff = tp - tm
        dtdiff = dtp - dtm
        for a, fa in enumerate(F_av):
            drho_dRav = cavdens.get_atomic_position_derivative(a)
            for v in (0, 1, 2):
                xav = drho_dRav[v]
                gxav = self._gradient(xav)
                dot = nngrho[0] * gxav[0]
                dot += nngrho[1] * gxav[1]
                dot += nngrho[2] * gxav[2]
                fa[v] -= st / delta * self.hamiltonian.finegd.integrate(
                    dtdiff * xav * ngrho + tdiff * dot,
                    global_integral=False
                    )

    def _gradient(self, x):
        g = (np.empty_like(x), np.empty_like(x), np.empty_like(x))
        for gi, opi in zip(g, self.gradops):
            opi.apply(x, gi)
        return g

    def _div(self, x):
        div = np.empty_like(x[0])
        tmp = np.empty_like(x[0])
        self.gradops[0].apply(x[0], div)
        self.gradops[1].apply(x[1], tmp)
        div += tmp
        self.gradops[2].apply(x[2], tmp)
        div += tmp
        return div

    def print_parameters(self, text):
        text('surface tension: %11.6f' % (self.surface_tension, ))
        text('Delta          : %11.6f' % (self.delta, ))


class LeakedDensityInteraction(BaseInteraction):
    """Interaction proportional to el. density leaking outside cavity."""

    name = 'Leaked Density'
    subscript = 'leak'

    def __init__(self, charging_energy):
        """Initialize parameters.

        Keyword arguments:
        charging_energy -- energy in eV needed for one electron
                           to leak outside the cavity

        """
        BaseInteraction.__init__(self)
        self.charging_energy = charging_energy

    def update_pseudo_potential(self, density):
        ce = self.charging_energy / Hartree
        g = 1. - self.hamiltonian.theta
        vks = ce * (g - density.nt_g * self.hamiltonian.dtheta * \
                        self.hamiltonian.drho)
        for vt_g in self.hamiltonian.vt_sg[:self.hamiltonian.nspins]:
            vt_g += vks
        return ce * self.hamiltonian.finegd.integrate(
            density.nt_g * g,
            global_integral=False
            )

    def update_forces(self, dens, F_av):
        ce = self.charging_energy / Hartree
        fixed = dens.nt_g * self.hamiltonian.dtheta
        for a, fa in enumerate(F_av):
            dRa = self.hamiltonian.cavdens.get_atomic_position_derivative(a)
            for v in (0, 1, 2):
                fa[v] += ce * self.hamiltonian.finegd.integrate(
                    fixed * dRa[v],
                    global_integral=False
                    )

    def print_parameters(self, text):
        text('E0: %11.6f' % (self.charging_energy, ))
