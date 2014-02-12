from gpaw.solvation.gridmem import NeedsGD
from ase.units import Bohr, Hartree
import numpy as np


class Interaction(NeedsGD):
    """Base class for non electrostatic solvent solute interactions."""

    subscript = 'unnamed'

    def __init__(self):
        NeedsGD.__init__(self)
        self.E = None
        self.delta_E_delta_n_g = None
        self.delta_E_delta_g_g = None

    def update(self, atoms, density, cavity):
        """Updates the Kohn-Sham potential and the energy.

        atoms and/or cavity are None iff they have not changed
        since the last call

        Returns whether the interaction has changed.
        """
        raise NotImplementedError

    @property
    def depends_on_atomic_positions(self):
        """returns whether the ia depends explicitly on atomic positions"""
        raise NotImplementedError

    @property
    def depends_on_el_density(self):
        """returns whether the ia depends explicitly on the electron density"""
        raise NotImplementedError

    def update_forces(self, dens, F_av):
        """Adds interaction forces to F_av in Hartree / Bohr."""
        raise NotImplementedError

    def print_parameters(self, text):
        """Prints parameters using text function."""
        pass


class SurfaceInteraction(Interaction):
    subscript = 'surf'

    def __init__(self, surface_tension):
        Interaction.__init__(self)
        self.surface_tension = float(surface_tension)

    def print_parameters(self, text):
        text('surface_tension: %s' % (self.surface_tension, ))


class VolumeInteraction(Interaction):
    """Interaction proportional to cavity volume"""
    subscript = 'vol'
    depends_on_el_density = False
    depends_on_atomic_positions = False

    def __init__(self, pressure):
        Interaction.__init__(self)
        self.pressure = float(pressure)

    def allocate(self):
        Interaction.allocate(self)
        self.delta_E_delta_g_g = self.gd.empty()

    def update(self, atoms, density, cavity):
        if cavity is None:
            return False
        vcalc = cavity.volume_calculator
        pressure = self.pressure * Bohr ** 3 / Hartree
        self.E = pressure * vcalc.V
        np.multiply(pressure, vcalc.delta_V_delta_g_g, self.delta_E_delta_g_g)
        return True


class LeakedDensityInteraction(Interaction):
    def __init__(self, charging_energy):
        Interaction.__init__(self)
