from ase.units import kB, Hartree
from gpaw.solvation.gridmem import NeedsGD
import numpy as np


class Cavity(NeedsGD):
    def __init__(self, surface_calculator=None, volume_calculator=None):
        NeedsGD.__init__(self)
        self.g_g = None
        self.del_g_del_n_g = None

    def update(self, atoms, density):
        """
        Updates the cavity.

        atoms are None, iff they have not changed.

        Returns whether the cavity has changed.
        """
        raise NotImplementedError()

    @property
    def depends_on_el_density(self):
        """returns whether the cavity depends on the electron density"""
        raise NotImplementedError()

    @property
    def depends_on_atomic_positions(self):
        """returns whether the cavity depends explicitly on atomic positions"""
        raise NotImplementedError()


class EffectivePotentialCavity(Cavity):
    def __init__(
        self,
        effective_potential,
        temperature,
        surface_calculator=None, volume_calculator=None
        ):
        Cavity.__init__(self, surface_calculator, volume_calculator)
        self.effective_potential = effective_potential
        self.temperature = temperature
        self.beta = 1. / (kB * temperature * Hartree)

    def allocate(self):
        Cavity.allocate(self)
        self.effective_potential.allocate()
        self.g_g = self.gd.empty()
        if self.depends_on_el_density:
            self.del_g_del_n_g = self.gd.empty()

    def update(self, atoms, density):
        if not self.effective_potential.update(atoms, density):
            return False
        u_g = self.effective_potential.u_g
        np.exp(u_g * self.beta, out=self.g_g)
        if self.depends_on_el_density:
            self.del_g_del_n_g.fill(-self.beta)
            self.del_g_del_n_g *= self.g_g
            self.del_g_del_n_g *= self.effective_potential.del_u_del_n_g
        return True

    @property
    def depends_on_el_density(self):
        return self.effective_potential.depends_on_el_density

    @property
    def depends_on_atomic_positions(self):
        return self.effective_potential.depends_on_atomic_positions


class Potential(NeedsGD):
    @property
    def depends_on_el_density(self):
        """returns whether the cavity depends on the electron density"""
        raise NotImplementedError()

    @property
    def depends_on_atomic_positions(self):
        """returns whether the cavity depends explicitly on atomic positions"""
        raise NotImplementedError()

    def update(self, atoms, density):
        """
        Updates the potential.

        atoms are None, iff they have not changed.

        Returns whether the potential has changed.
        """
        raise NotImplementedError()


class Power12Potential(Potential):
    depends_on_el_density = False
    depends_on_atomic_positions = True

    def __init__(self, vdw_radii, u0):
        Potential.__init__(self)


class DensityCavity(Cavity):
    def __init__(
        self,
        density, smooth_step,
        surface_calculator=None, volume_calculator=None
        ):
        Cavity.__init__(self, surface_calculator, volume_calculator)


class Density():
    def __init__(self):
        pass


class ElDensity(Density):
    pass


class SSS09Density(Density):
    def __init__(self, vdw_radii):
        Density.__init__(self)


class SmoothStep():
    def __init__(self):
        pass


class ADM12SmoothStep(SmoothStep):
    def __init__(self, rhomin, rhomax, epsinf):
        SmoothStep.__init__(self)


class FG02SmoothStep(SmoothStep):
    def __init__(self, rho0, beta):
        SmoothStep.__init__(self)


class SurfaceCalculator():
    def __init__(self):
        pass


class ADM12Surface(SurfaceCalculator):
    def __init__(self, delta):
        SurfaceCalculator.__init__(self)


class VolumeCalculator():
    def __init__(self):
        pass


class KB51Volume(VolumeCalculator):
    def __init__(self, compressibility, temperature):
        VolumeCalculator.__init__(self)
