from ase.units import kB, Hartree, Bohr
from gpaw.solvation.gridmem import NeedsGD
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


class Cavity(NeedsGD):
    def __init__(self, surface_calculator=None, volume_calculator=None):
        NeedsGD.__init__(self)
        self.g_g = None
        self.del_g_del_n_g = None
        self.surface_calculator = surface_calculator
        self.volume_calculator = volume_calculator

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

    def print_parameters(self, text):
        """prints parameters using text function"""
        typ = self.surface_calculator and self.surface_calculator.__class__
        text('surface calculator: %s' % (typ, ))
        if self.surface_calculator is not None:
            self.surface_calculator.print_parameters(text)
        text()
        typ = self.volume_calculator and self.volume_calculator.__class__
        text('volume calculator: %s' % (typ, ))
        if self.volume_calculator is not None:
            self.volume_calculator.print_parameters(text)


class EffectivePotentialCavity(Cavity):
    def __init__(
        self,
        effective_potential,
        temperature,
        surface_calculator=None, volume_calculator=None
        ):
        Cavity.__init__(self, surface_calculator, volume_calculator)
        self.effective_potential = effective_potential
        self.temperature = float(temperature)
        self.minus_beta = -1. / (kB * temperature * Hartree)

    def set_grid_descriptor(self, gd):
        Cavity.set_grid_descriptor(self, gd)
        self.effective_potential.set_grid_descriptor(gd)

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
        np.exp(u_g * self.minus_beta, out=self.g_g)
        if self.depends_on_el_density:
            self.del_g_del_n_g.fill(self.minus_beta)
            self.del_g_del_n_g *= self.g_g
            self.del_g_del_n_g *= self.effective_potential.del_u_del_n_g
        return True

    @property
    def depends_on_el_density(self):
        return self.effective_potential.depends_on_el_density

    @property
    def depends_on_atomic_positions(self):
        return self.effective_potential.depends_on_atomic_positions

    def print_parameters(self, text):
        text('effective potential: %s' % (self.effective_potential.__class__))
        self.effective_potential.print_parameters(text)
        text()
        Cavity.print_parameters(self, text)


class Potential(NeedsGD):
    def __init__(self):
        NeedsGD.__init__(self)
        self.u_g = None
        self.del_u_del_n_g = None

    @property
    def depends_on_el_density(self):
        """returns whether the cavity depends on the electron density"""
        raise NotImplementedError()

    @property
    def depends_on_atomic_positions(self):
        """returns whether the cavity depends explicitly on atomic positions"""
        raise NotImplementedError()

    def allocate(self):
        NeedsGD.allocate(self)
        self.u_g = self.gd.empty()
        if self.depends_on_el_density:
            self.del_u_del_n_g = self.gd.empty()

    def update(self, atoms, density):
        """
        Updates the potential.

        atoms are None, iff they have not changed.

        Returns whether the potential has changed.
        """
        raise NotImplementedError()

    def print_parameters(self, text):
        pass


class Power12Potential(Potential):
    """
    Inverse power law potential.

    An 1 / r ** 12 repulsive potential
    taking the value u0 at the atomic radius.
    """
    depends_on_el_density = False
    depends_on_atomic_positions = True

    def __init__(self, atomic_radii, u0, r_max=10. * Bohr):
        Potential.__init__(self)
        self.atomic_radii = np.array(atomic_radii)
        self.u0 = float(u0)
        self.r_max = float(r_max)

    def update(self, atoms, density):
        if atoms is None:
            return False
        assert len(atoms) == len(self.atomic_radii)
        pos_aav = get_pbc_positions(atoms, self.r_max / Bohr)
        r_vg = self.gd.get_grid_point_coordinates()
        r12_a = (self.atomic_radii / Bohr) ** 12
        self.u_g.fill(.0)
        na = np.newaxis
        for index, pos_av in pos_aav.iteritems():
            r_12 = r12_a[index]
            for pos_v in pos_av:
                origin_vg = pos_v[:, na, na, na]
                r2_g = np.sum((r_vg - origin_vg) ** 2, axis=0)
                self.u_g += r_12 / r2_g ** 6
        self.u_g[np.isnan(self.u_g)] = np.inf
        return True

    def print_parameters(self, text):
        Potential.print_parameters(self, text)
        text('u0: %s' % (self.u0, ))
        text('atomic_radii: [%s]' % (', '.join(map(str, self.atomic_radii)), ))


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

    def print_parameters(self, text):
        pass


class ADM12Surface(SurfaceCalculator):
    def __init__(self, delta):
        SurfaceCalculator.__init__(self)
        self.delta = float(delta)

    def print_parameters(self, text):
        SurfaceCalculator.print_parameters(self, text)
        text('delta: %s' % (self.delta, ))


class VolumeCalculator():
    def __init__(self):
        pass

    def print_parameters(self, text):
        pass


class KB51Volume(VolumeCalculator):
    def __init__(self, compressibility, temperature):
        VolumeCalculator.__init__(self)
        self.compressibility = float(compressibility)
        self.temperature = float(temperature)

    def print_parameters(self, text):
        VolumeCalculator.print_parameters(self, text)
        text('compressibility: %s' % (self.compressibility, ))
        text('temperature:     %s' % (self.temperature, ))
