from gpaw.solvation.gridmem import NeedsGD


class Interaction(NeedsGD):
    """Base class for non electrostatic solvent solute interactions."""

    subscript = 'unnamed'

    def update_atoms(self, atoms):
        """Handles changes to atoms."""
        pass

    def update_pseudo_potential(self, density):
        """Updates the Kohn-Sham potential of the Hamiltonian.

        Returns interaction energy in Hartree.

        """
        raise NotImplementedError

    def update_forces(self, dens, F_av):
        """Adds interaction forces to F_av in Hartree / Bohr."""
        raise NotImplementedError

    def print_parameters(self, text):
        """Prints parameters using text function."""
        pass


class SurfaceInteraction(Interaction):
    def __init__(self, surface_tension):
        Interaction.__init__(self)


class VolumeInteraction(Interaction):
    def __init__(self, pressure):
        Interaction.__init__(self)


class LeakedDensityInteraction(Interaction):
    def __init__(self, charging_energy):
        Interaction.__init__(self)
