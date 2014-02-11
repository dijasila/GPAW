class Cavity():
    def __init__(self, surface_calculator=None, volume_calculator=None):
        self.gd = None
        self.g_g = None
        self.del_g_del_n_g = None

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def update_atoms(self, atoms):
        """returns whether cavity has changed"""
        return False

    def update_el_density(self, density):
        """returns whether cavity has changed"""
        return False


class EffectivePotentialCavity(Cavity):
    def __init__(
        self,
        effective_potential,
        temperature,
        surface_calculator=None, volume_calculator=None
        ):
        Cavity.__init__(self, surface_calculator, volume_calculator)


class Potential():
    def __init__(self):
        pass


class Power12Potential(Potential):
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
