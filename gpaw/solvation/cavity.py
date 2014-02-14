from ase.units import kB, Hartree, Bohr
from gpaw.solvation.gridmem import NeedsGD
from gpaw.fd_operators import Gradient
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
        self.V = None  # global Volume
        self.A = None  # global Surface

    def update(self, atoms, density):
        """
        Updates the cavity.

        atoms are None, iff they have not changed.

        Returns whether the cavity has changed.
        """
        raise NotImplementedError()

    def update_vol_surf(self):
        if self.surface_calculator is not None:
            self.surface_calculator.update(self)
        if self.volume_calculator is not None:
            self.volume_calculator.update(self)

    def communicate_vol_surf(self, world):
        if self.surface_calculator is not None:
            A = np.array([self.surface_calculator.A])
            self.gd.comm.sum(A)
            world.broadcast(A, 0)
            self.A = A[0]
        else:
            self.A = None
        if self.volume_calculator is not None:
            V = np.array([self.volume_calculator.V])
            self.gd.comm.sum(V)
            world.broadcast(V, 0)
            self.V = V[0]
        else:
            self.V = None

    def allocate(self):
        NeedsGD.allocate(self)
        if self.surface_calculator is not None:
            self.surface_calculator.allocate()
        if self.volume_calculator is not None:
            self.volume_calculator.allocate()

    def set_grid_descriptor(self, gd):
        NeedsGD.set_grid_descriptor(self, gd)
        if self.surface_calculator is not None:
            self.surface_calculator.set_grid_descriptor(gd)
        if self.volume_calculator is not None:
            self.volume_calculator.set_grid_descriptor(gd)

    def get_del_r_vg(self, atom_index, density):
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
        self.minus_beta = -1. / (kB * temperature / Hartree)

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

    def get_del_r_vg(self, atom_index, density):
        u = self.effective_potential
        del_u_del_r_vg = u.get_del_r_vg(atom_index, density)
        for v in (0, 1, 2):
            assert (self.g_g[np.isnan(del_u_del_r_vg[v])] < 1e-10).all()  # XXX remove
            del_u_del_r_vg[v][np.isnan(del_u_del_r_vg[v])] = .0
        return self.minus_beta * self.g_g * del_u_del_r_vg

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

    def get_del_r_vg(self, atom_index, density):
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

    def __init__(self, atomic_radii, u0, pbc_cutoff=1e-6):
        Potential.__init__(self)
        self.atomic_radii = np.array(atomic_radii)
        self.u0 = float(u0)
        self.pbc_cutoff = float(pbc_cutoff)
        self.r12_a = None
        self.r_vg = None
        self.pos_aav = None
        self.del_u_del_r_vg = None

    def allocate(self):
        Potential.allocate(self)
        self.r_vg = self.gd.get_grid_point_coordinates()
        self.del_u_del_r_vg = self.gd.empty(3)

    def update(self, atoms, density):
        if atoms is None:
            return False
        assert len(atoms) == len(self.atomic_radii)
        self.r12_a = (self.atomic_radii / Bohr) ** 12
        r_cutoff = (self.r12_a.max() * self.u0 / self.pbc_cutoff) ** (1. / 12.)
        self.pos_aav = get_pbc_positions(atoms, r_cutoff)
        self.u_g.fill(.0)
        na = np.newaxis
        for index, pos_av in self.pos_aav.iteritems():
            r_12 = self.r12_a[index]
            for pos_v in pos_av:
                origin_vg = pos_v[:, na, na, na]
                r2_g = np.sum((self.r_vg - origin_vg) ** 2, axis=0)
                self.u_g += r_12 / r2_g ** 6
        self.u_g *= self.u0 / Hartree
        self.u_g[np.isnan(self.u_g)] = np.inf
        return True

    def get_del_r_vg(self, atom_index, density):
        u0 = self.u0 / Hartree
        r12 = self.r12_a[atom_index]
        na = np.newaxis
        self.del_u_del_r_vg.fill(.0)
        for pos_v in self.pos_aav[atom_index]:
            origin_vg = pos_v[:, na, na, na]
            diff_vg = self.r_vg - origin_vg
            r2_g = np.sum(diff_vg ** 2, axis=0)
            self.del_u_del_r_vg += diff_vg / r2_g ** 7
        self.del_u_del_r_vg *= (12. * u0 * r12)
        return self.del_u_del_r_vg

    def print_parameters(self, text):
        Potential.print_parameters(self, text)
        text('atomic_radii: [%s]' % (', '.join(map(str, self.atomic_radii)), ))
        text('u0: %s' % (self.u0, ))
        text('pbc_cutoff: %s' % (self.pbc_cutoff, ))


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


class SurfaceCalculator(NeedsGD):
    def __init__(self):
        NeedsGD.__init__(self)
        self.A = None
        self.delta_A_delta_g_g = None

    def print_parameters(self, text):
        pass

    def update(self, cavity):
        raise NotImplementedError()


class GradientSurface(SurfaceCalculator):
    def __init__(self, nn=3):
        SurfaceCalculator.__init__(self)
        self.nn = nn
        self.gradient = None
        self.gradient_in = None
        self.gradient_out = None
        self.norm_grad_out = None
        self.div_tmp = None

    def allocate(self):
        SurfaceCalculator.allocate(self)
        self.gradient = [
            Gradient(self.gd, i, 1.0, self.nn) for i in (0, 1, 2)
            ]
        self.gradient_in = self.gd.empty()
        self.gradient_out = (self.gd.empty(), self.gd.empty(), self.gd.empty())
        self.norm_grad_out = self.gd.empty()
        self.delta_A_delta_g_g = self.gd.empty()
        self.div_tmp = self.gd.empty()

    def update(self, cavity):
        self.calc_grad(cavity)
        self.A = self.gd.integrate(self.norm_grad_out, global_integral=False)
        mask = self.norm_grad_out > .0
        masked_norm_grad = self.norm_grad_out[mask]
        for i in (0, 1, 2):
            self.gradient_out[i][mask] /= masked_norm_grad
        self.calc_div(self.gradient_out, self.delta_A_delta_g_g)
        self.delta_A_delta_g_g *= -1

    def calc_grad(self, cavity):
        # zero on non-PBC boundary (cavity.g_g is 1 on non-PBC boundary)
        np.subtract(cavity.g_g, 1., self.gradient_in)
        self.norm_grad_out.fill(.0)
        for i in (0, 1, 2):
            self.gradient[i].apply(self.gradient_in, self.gradient_out[i])
            self.norm_grad_out += self.gradient_out[i] ** 2
        self.norm_grad_out **= .5

    def calc_div(self, vec, out):
        self.gradient[0].apply(vec[0], out)
        self.gradient[1].apply(vec[1], self.div_tmp)
        out += self.div_tmp
        self.gradient[2].apply(vec[2], self.div_tmp)
        out += self.div_tmp


class VolumeCalculator(NeedsGD):
    def __init__(self):
        NeedsGD.__init__(self)
        self.V = None
        self.delta_V_delta_g_g = None

    def print_parameters(self, text):
        pass

    def update(self, cavity):
        raise NotImplementedError()


class KB51Volume(VolumeCalculator):
    """
    KB51 Volume Calculator

    V = Integral(1 - g) + kappa_T * k_B * T

    Following

    J. G. Kirkwood and F. P. Buff,
    The Journal of Chemical Physics, vol. 19, no. 6, pp. 774--777, 1951
    """

    def __init__(self, compressibility=.0, temperature=.0):
        VolumeCalculator.__init__(self)
        self.compressibility = float(compressibility)
        self.temperature = float(temperature)

    def print_parameters(self, text):
        VolumeCalculator.print_parameters(self, text)
        text('compressibility: %s' % (self.compressibility, ))
        text('temperature:     %s' % (self.temperature, ))

    def update(self, cavity):
        self.V = self.gd.integrate(1. - cavity.g_g, global_integral=False)
        V_compress = self.compressibility * kB * self.temperature / Bohr ** 3
        self.V += V_compress / self.gd.comm.size
        self.delta_V_delta_g_g = -1.
