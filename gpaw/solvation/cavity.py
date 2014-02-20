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


def divide_silently(x, y):
    """
    Divides numpy arrays x / y ignoring all floating point errors.

    Use with caution!
    """
    old_err = np.seterr(all='ignore')
    result = x / y
    np.seterr(**old_err)
    return result


class Cavity(NeedsGD):
    def __init__(self, surface_calculator=None, volume_calculator=None):
        NeedsGD.__init__(self)
        self.g_g = None
        self.del_g_del_n_g = None
        self.surface_calculator = surface_calculator
        self.volume_calculator = volume_calculator
        self.V = None  # global Volume
        self.A = None  # global Surface

    def estimate_memory(self, mem):
        ngrids = 1 + self.depends_on_el_density
        mem.subnode('Distribution Function', ngrids * self.gd.bytecount())
        if self.surface_calculator is not None:
            self.surface_calculator.estimate_memory(
                mem.subnode('Surface Calculator')
                )
        if self.volume_calculator is not None:
            self.volume_calculator.estimate_memory(
                mem.subnode('Volume Calculator')
                )

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
        self.g_g = self.gd.empty()
        if self.depends_on_el_density:
            self.del_g_del_n_g = self.gd.empty()
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

    def estimate_memory(self, mem):
        Cavity.estimate_memory(self, mem)
        self.effective_potential.estimate_memory(
            mem.subnode('Effective Potential')
            )

    def set_grid_descriptor(self, gd):
        Cavity.set_grid_descriptor(self, gd)
        self.effective_potential.set_grid_descriptor(gd)

    def allocate(self):
        Cavity.allocate(self)
        self.effective_potential.allocate()

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
        # asserts lim_(||r - r_atom|| -> 0) dg/du * du/dr_atom = 0
        del_u_del_r_vg[np.isnan(del_u_del_r_vg)] = .0
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

    # --- BEGIN GradientSurface API ---
    def get_inner_function(self):
        return np.minimum(self.effective_potential.u_g, 1e20)

    def get_inner_function_boundary_value(self):
        if hasattr(self.effective_potential, 'grad_u_vg'):
            raise NotImplementedError
        else:
            return .0

    def get_grad_inner(self):
        if hasattr(self.effective_potential, 'grad_u_vg'):
            return self.effective_potential.grad_u_vg
        else:
            raise NotImplementedError

    def get_del_outer_del_inner(self):
        return self.minus_beta * self.g_g

    # --- END GradientSurface API ---


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

    def estimate_memory(self, mem):
        ngrids = 1 + self.depends_on_el_density
        mem.subnode('Potential', ngrids * self.gd.bytecount())

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
        self.atomic_radii = atomic_radii
        self.u0 = float(u0)
        self.pbc_cutoff = float(pbc_cutoff)
        self.r12_a = None
        self.r_vg = None
        self.pos_aav = None
        self.del_u_del_r_vg = None
        self.atomic_radii_output = None
        self.grad_u_vg = None

    def estimate_memory(self, mem):
        Potential.estimate_memory(self, mem)
        nbytes = self.gd.bytecount()
        mem.subnode('Coordinates', 6 * nbytes)
        mem.subnode('Gradient', 3 * nbytes)

    def allocate(self):
        Potential.allocate(self)
        self.r_vg = self.gd.get_grid_point_coordinates()
        self.del_u_del_r_vg = self.gd.empty(3)
        self.grad_u_vg = self.gd.empty(3)

    def update(self, atoms, density):
        if atoms is None:
            return False
        self.atomic_radii_output = np.array(self.atomic_radii(atoms))
        self.r12_a = (self.atomic_radii_output / Bohr) ** 12
        r_cutoff = (self.r12_a.max() * self.u0 / self.pbc_cutoff) ** (1. / 12.)
        self.pos_aav = get_pbc_positions(atoms, r_cutoff)
        self.u_g.fill(.0)
        self.grad_u_vg.fill(.0)
        na = np.newaxis
        for index, pos_av in self.pos_aav.iteritems():
            r12 = self.r12_a[index]
            for pos_v in pos_av:
                origin_vg = pos_v[:, na, na, na]
                r_diff_vg = self.r_vg - origin_vg
                r_diff2_g = (r_diff_vg ** 2).sum(0)
                r12_g = r_diff2_g ** 6
                r14_g = r12_g * r_diff2_g
                self.u_g += divide_silently(r12, r12_g)
                self.grad_u_vg += divide_silently(r12 * r_diff_vg, r14_g)
        self.u_g *= self.u0 / Hartree
        # np.exp(-np.inf) = .0
        self.u_g[np.isnan(self.u_g)] = np.inf

        self.grad_u_vg *= -12. * self.u0 / Hartree
        # mask points where the limit of all later
        # calculations is zero anyways
        self.grad_u_vg[...] = np.nan_to_num(self.grad_u_vg)
        # avoid overflow in norm calculation:
        self.grad_u_vg[self.grad_u_vg < -1e20] = -1e20
        self.grad_u_vg[self.grad_u_vg > 1e20] = 1e20
        return True

    def get_del_r_vg(self, atom_index, density):
        u0 = self.u0 / Hartree
        r12 = self.r12_a[atom_index]
        na = np.newaxis
        self.del_u_del_r_vg.fill(.0)
        for pos_v in self.pos_aav[atom_index]:
            origin_vg = pos_v[:, na, na, na]
            diff_vg = self.r_vg - origin_vg
            r14_g = np.sum(diff_vg ** 2, axis=0) ** 7
            self.del_u_del_r_vg += divide_silently(diff_vg, r14_g)
        self.del_u_del_r_vg *= (12. * u0 * r12)
        return self.del_u_del_r_vg

    def print_parameters(self, text):
        Potential.print_parameters(self, text)
        if self.atomic_radii_output is None:
            radiistr = 'not initialized (dry run)'
        else:
            radiistr = str(self.atomic_radii_output)
        text('atomic_radii: ' + radiistr)
        text('u0: %s' % (self.u0, ))
        text('pbc_cutoff: %s' % (self.pbc_cutoff, ))


class SmoothStepCavity(Cavity):
    def __init__(
        self,
        density,
        surface_calculator=None, volume_calculator=None
        ):
        Cavity.__init__(self, surface_calculator, volume_calculator)
        self.del_g_del_rho_g = None
        self.density = density

    @property
    def depends_on_el_density(self):
        return self.density.depends_on_el_density

    @property
    def depends_on_atomic_positions(self):
        return self.density.depends_on_atomic_positions

    def set_grid_descriptor(self, gd):
        Cavity.set_grid_descriptor(self, gd)
        self.density.set_grid_descriptor(gd)

    def estimate_memory(self, mem):
        Cavity.estimate_memory(self, mem)
        mem.subnode('Cavity Derivative', self.gd.bytecount())
        self.density.estimate_memory(mem.subnode('Density'))

    def allocate(self):
        Cavity.allocate(self)
        self.del_g_del_rho_g = self.gd.empty()
        self.density.allocate()

    def update(self, atoms, density):
        if not self.density.update(atoms, density):
            return False
        self.update_smooth_step(self.density.rho_g)
        if self.depends_on_el_density:
            np.multiply(
                self.del_g_del_rho_g,
                self.density.del_rho_del_n_g,
                self.del_g_del_n_g
                )
        return True

    def update_smooth_step(self, rho_g):
        """calculates self.g_g and self.del_g_del_rho_g"""
        raise NotImplementedError()

    def print_parameters(self, text):
        text('density: %s' % (self.density.__class__))
        self.density.print_parameters(text)
        text()
        Cavity.print_parameters(self, text)

    # --- BEGIN GradientSurface API ---
    def get_inner_function(self):
        return self.density.rho_g

    def get_inner_function_boundary_value(self):
        return .0

    def get_grad_inner(self):
        raise NotImplementedError

    def get_del_outer_del_inner(self):
        return self.del_g_del_rho_g

    # --- END GradientSurface API ---


class Density(NeedsGD):
    def __init__(self):
        NeedsGD.__init__(self)
        self.rho_g = None
        self.del_rho_del_n_g = None

    def estimate_memory(self, mem):
        nbytes = self.gd.bytecount()
        mem.subnode('Density', nbytes)
        if self.depends_on_el_density:
            mem.subnode('Density Derivative', nbytes)

    def allocate(self):
        NeedsGD.allocate(self)
        self.rho_g = self.gd.empty()
        if self.depends_on_el_density:
            self.del_rho_del_n_g = self.gd.empty()

    @property
    def depends_on_el_density(self):
        raise NotImplementedError()

    @property
    def depends_on_atomic_positions(self):
        raise NotImplementedError()

    def print_parameters(self, text):
        pass


class ElDensity(Density):
    depends_on_el_density = True
    depends_on_atomic_positions = False

    def allocate(self):
        Density.allocate(self)
        self.del_rho_del_n_g = 1.  # free array

    def update(self, atoms, density):
        self.rho_g[:] = density.nt_g
        self.rho_g[self.rho_g < .0] = .0
        return True


class SSS09Density(Density):
    def __init__(self, vdw_radii):
        Density.__init__(self)


class ADM12SmoothStepCavity(SmoothStepCavity):
    """Cavity from smooth step function.

    Following O. Andreussi, I. Dabo, and N. Marzari,
    J. Chem. Phys. 136, 064102 (2012).

    """

    def __init__(
        self,
        rhomin, rhomax, epsinf,
        density,
        surface_calculator=None, volume_calculator=None
        ):
        SmoothStepCavity.__init__(
            self, density, surface_calculator, volume_calculator
            )
        self.rhomin = float(rhomin)
        self.rhomax = float(rhomax)
        self.epsinf = float(epsinf)

    def update_smooth_step(self, rho_g):
        eps = self.epsinf
        inside = rho_g > self.rhomax * Bohr ** 3
        outside = rho_g < self.rhomin * Bohr ** 3
        transition = np.logical_not(
            np.logical_or(inside, outside)
            )
        self.g_g[inside] = .0
        self.g_g[outside] = 1.
        self.del_g_del_rho_g.fill(.0)
        t, dt = self._get_t_dt(np.log(rho_g[transition]))
        if eps == 1.0:
            # lim_{eps -> 1} (eps - eps ** t) / (eps - 1) = 1 - t
            self.g_g[transition] = t
            self.del_g_del_rho_g[transition] = dt / rho_g[transition]
        else:
            eps_to_t = eps ** t
            self.g_g[transition] = (eps_to_t - 1.) / (eps - 1.)
            self.del_g_del_rho_g[transition] = (
                eps_to_t * np.log(eps) * dt
                ) / (
                rho_g[transition] * (eps - 1.)
                )

    def _get_t_dt(self, x):
        lnmax = np.log(self.rhomax * Bohr ** 3)
        lnmin = np.log(self.rhomin * Bohr ** 3)
        twopi = 2. * np.pi
        arg = twopi * (lnmax - x) / (lnmax - lnmin)
        t = (arg - np.sin(arg)) / twopi
        dt = -2. * np.sin(arg / 2.) ** 2 / (lnmax - lnmin)
        return (t, dt)

    def print_parameters(self, text):
        text('rhomin: %s' % (self.rhomin, ))
        text('rhomax: %s' % (self.rhomax, ))
        text('epsinf: %s' % (self.epsinf, ))
        SmoothStepCavity.print_parameters(self, text)


class FG02SmoothStepCavity(SmoothStepCavity):
    """Cavity from smooth step function.

    Following J. Fattebert, and F. Gygi, J Comput Chem 23: 662-666, 2002.

    """

    def __init__(
        self,
        rho0,
        beta,
        density,
        surface_calculator=None, volume_calculator=None
        ):
        SmoothStepCavity.__init__(
            self, density, surface_calculator, volume_calculator
            )
        self.rho0 = float(rho0)
        self.beta = float(beta)

    def update_smooth_step(self, rho_g):
        rho0 = self.rho0 / (1. / Bohr ** 3)
        rho_scaled_g = rho_g / rho0
        exponent = 2. * self.beta
        np.divide(1., rho_scaled_g ** exponent + 1., self.g_g)
        np.multiply(
            (-exponent / rho0) * rho_scaled_g ** (exponent - 1.),
            self.g_g ** 2,
            self.del_g_del_rho_g
            )

    def print_parameters(self, text):
        text('rho0: %s' % (self.rho0, ))
        text('beta: %s' % (self.beta, ))
        SmoothStepCavity.print_parameters(self, text)


class SurfaceCalculator(NeedsGD):
    def __init__(self):
        NeedsGD.__init__(self)
        self.A = None
        self.delta_A_delta_g_g = None

    def estimate_memory(self, mem):
        mem.subnode('Functional Derivative', self.gd.bytecount())

    def allocate(self):
        NeedsGD.allocate(self)
        self.delta_A_delta_g_g = self.gd.empty()

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

    def estimate_memory(self, mem):
        SurfaceCalculator.estimate_memory(self, mem)
        nbytes = self.gd.bytecount()
        mem.subnode('Gradient', 5 * nbytes)
        mem.subnode('Divergence', nbytes)

    def allocate(self):
        SurfaceCalculator.allocate(self)
        self.gradient = [
            Gradient(self.gd, i, 1.0, self.nn) for i in (0, 1, 2)
            ]
        self.gradient_in = self.gd.empty()
        self.gradient_out = self.gd.empty(3)
        self.norm_grad_out = self.gd.empty()
        self.div_tmp = self.gd.empty()

    def update(self, cavity):
        inner = cavity.get_inner_function()
        del_outer_del_inner = cavity.get_del_outer_del_inner()
        sign = np.sign(del_outer_del_inner.max() + del_outer_del_inner.min())
        try:
            self.gradient_out[...] = cavity.get_grad_inner()
            self.norm_grad_out = (self.gradient_out ** 2).sum(0) ** .5
        except NotImplementedError:
            self.calc_grad(inner, cavity.get_inner_function_boundary_value())
        self.A = sign * self.gd.integrate(
            del_outer_del_inner * self.norm_grad_out, global_integral=False
            )
        mask = self.norm_grad_out > 1e-12  # avoid division by zero or overflow
        imask = np.logical_not(mask)
        masked_norm_grad = self.norm_grad_out[mask]
        for i in (0, 1, 2):
            self.gradient_out[i][mask] /= masked_norm_grad
            # set limit for later calculations:
            self.gradient_out[i][imask] = .0
        self.calc_div(self.gradient_out, self.delta_A_delta_g_g)
        if sign == 1:
            self.delta_A_delta_g_g *= -1.

    def calc_grad(self, x, boundary):
        if boundary != .0:
            np.subtract(x, boundary, self.gradient_in)
            gradient_in = self.gradient_in
        else:
            gradient_in = x
        self.norm_grad_out.fill(.0)
        for i in (0, 1, 2):
            self.gradient[i].apply(gradient_in, self.gradient_out[i])
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

    def estimate_memory(self, mem):
        mem.subnode('Functional Derivative', self.gd.bytecount())

    def allocate(self):
        NeedsGD.allocate(self)
        self.delta_V_delta_g_g = self.gd.empty()

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

    def allocate(self):
        VolumeCalculator.allocate(self)
        self.delta_V_delta_g_g = -1.  # frees array

    def update(self, cavity):
        self.V = self.gd.integrate(1. - cavity.g_g, global_integral=False)
        V_compress = self.compressibility * kB * self.temperature / Bohr ** 3
        self.V += V_compress / self.gd.comm.size
