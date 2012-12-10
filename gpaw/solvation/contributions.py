from gpaw.solvation.poisson import WeightedFDPoissonSolver
from gpaw.utilities.tools import coordinates
from gpaw.fd_operators import Gradient, Laplace
from gpaw.poisson import PoissonSolver
from types import StringTypes
from ase.units import Hartree, Bohr, kB
import numpy

MODE_ELECTRON_DENSITY_CUTOFF = 'electron_density_cutoff'
MODE_TEST = 'test'
MODE_RADII_CUTOFF = 'radii_cutoff'
MODE_SURFACE_TENSION = 'surface_tension'
MODE_AREA_VOLUME_VDW = 'area_volume_vdw'
MODE_C6_VDW = 'c6_vdw'
MODE_SPILLED_DENSITY_VDW = 'spilled_density_vdw'


class NoDefault:
    pass


class BaseContribution:
    """
    base class for all solvation contributions

    methods are called by the hamiltonian in the
    order they appear in the source code below
    """
    default_parameters = {}

    def __init__(self, hamiltonian, params):
        """
        should not be overwritten, use init() instead
        """
        self.hamiltonian = hamiltonian
        self.params = params
        self.init()

    def init(self):
        """
        inexpensive initialization
        """
        pass

    def set_atoms(self, atoms):
        """
        handles atoms changed
        """
        pass

    def allocate(self):
        """
        expensive initialization

        called before update_pseudo_potential, if hamiltonian.vt_sg was None
        """
        pass

    def update_pseudo_potential(self, density):
        """
        handles density changed
        updates the Kohn-Sham potential of the hamiltonian

        returns energy of the contribution in Hartree
        """
        raise NotImplementedError

    def calculate_forces(self, density, F_av):
        """
        adds forces of the contribution to F_av in Hartree / Bohr
        """
        raise NotImplementedError


class BaseElectronicContribution(BaseContribution):
    """
    base class for electronic solvation contributions

    dielectric -- list [epsr dx_epsr, dy_epsr, dz_epsr]
                  has to be updated after calls to set_atoms
                  and / or update_pseudo_potential
    """

    def init(self):
        self.dielectric = []

    def allocate(self):
        del self.dielectric[:]
        finegd = self.hamiltonian.finegd
        eps = finegd.empty()
        eps.fill(1.0)
        self.dielectric.append(eps)
        self.dielectric.extend([gd.zeros() for gd in (finegd, ) * 3])

    def make_poisson_solver(self, psolver):
        """
        creates and returns a Poisson solver

        or uses or modifies the one given by 'psolver'
        """
        if psolver is None:
            psolver = WeightedFDPoissonSolver(
                nn=3, relax='J', dielectric=self.dielectric
                )
        else:
            psolver.dielectric = self.dielectric
        return psolver

    def get_cavity_surface_area(self):
        """
        returns cavity surface area in Bohr ** 2
        """
        raise NotImplementedError

    def get_cavity_volume(self):
        """
        returns cavity volume in Bohr ** 3
        """
        raise NotImplementedError

    def get_cavity_surface_area_pseudo_potential(self):
        """
        returns contribution to the Kohn-Sham potential

        if a term Acav was included in the energy.
        """
        raise NotImplementedError

    def get_cavity_volume_pseudo_potential(self):
        """
        returns contribution to the Kohn-Sham potential

        if a term Vcav was included in the energy.
        """
        raise NotImplementedError

    def get_cavity_surface_area_forces(self):
        """
        returns contributions to the forces

        if a term Acav was included in the energy.
        """
        raise NotImplementedError

    def get_cavity_volume_forces(self):
        """
        returns contributions to the forces

        if a term Vcav was included in the energy.
        """
        raise NotImplementedError


class DummyContribution(BaseContribution):
    def update_pseudo_potential(self, density):
        return .0

    def calculate_forces(self, density, F_av):
        pass


class DummyElContribution(DummyContribution):
    def make_poisson_solver(self, psolver):
        if psolver is None:
            psolver = PoissonSolver(nn=3, relax='J')
        return psolver

    def get_cavity_surface_area(self):
        return 0.

    def get_cavity_volume(self):
        return 0.

    def get_cavity_surface_area_pseudo_potential(self):
        return None

    def get_cavity_volume_pseudo_potential(self):
        return None

    def get_cavity_surface_area_forces(self):
        return None

    def get_cavity_volume_forces(self):
        return None


class RadiiElContribution(BaseElectronicContribution):
    """
    Electronic contribution to the solvation

    with cavity from fake density constructed by spheres
    following Sanchez et al J. Chem. Phys. 131 (2009) 174108
    """

    default_parameters = {
        'radii'     : NoDefault,
        'centers'   : 'atoms',
        'exponent'  : NoDefault,
        'epsilon_r' : 1.0,
        'area_delta': 0.01
        }

    def init(self):
        BaseElectronicContribution.init(self)
        self.atoms = None
        self.cavity_dirty = True
        centers = self.params['centers']
        self.center_on_atoms = isinstance(centers, StringTypes) and \
                               centers.lower().strip() == 'atoms'
        self.Acav = None
        self.Vcav = None

    def allocate(self):
        BaseElectronicContribution.allocate(self)
        self.update_dielectric()

    def update_pseudo_potential(self, density):
        self.update_dielectric()
        return .0

    def set_atoms(self, atoms):
        if self.center_on_atoms:
            self.atoms = atoms
            self.cavity_dirty = True

    def get_centers(self):
        centers = self.params['centers']
        if self.center_on_atoms:
            centers = self.atoms.positions
        else:
            centers = numpy.array(centers, dtype=float)
            assert centers.ndim == 2
            assert centers.shape[1] == 3
        assert len(centers) == len(self.params['radii'])
        return centers

    def theta(self, c):
        beta = self.params['exponent']
        tmp = (self.gamma / c) ** (2. * beta)
        return tmp / (tmp + 1.)

    def update_dielectric(self):
        if not self.cavity_dirty:
            return
        centers = self.get_centers()
        ham = self.hamiltonian
        self.gamma = ham.finegd.zeros()
        for deps in self.dielectric[1:]:
            deps.fill(.0)
        for R, center in zip(self.params['radii'], centers):
            coords = coordinates(ham.finegd, origin=center / Bohr)
            distances = numpy.sqrt(coords[1])
            tmp = numpy.exp(R / Bohr - distances)
            self.gamma += tmp
            for i in (0, 1, 2):
                self.dielectric[1 + i] -= tmp * coords[0][i] / distances
        self.grad_gamma = [self.dielectric[1 + i].copy() for i in (0, 1, 2)]

        #XXX optimize numerics
        beta = self.params['exponent']
        epsinf = self.params['epsilon_r']
        delta = self.params['area_delta']
        self.norm_grad_gamma = .0
        for i in (0, 1, 2):
            self.norm_grad_gamma += self.dielectric[i + 1] ** 2
        numpy.sqrt(self.norm_grad_gamma, self.norm_grad_gamma)
        theta = self.theta
        integrand = theta(1. - delta / 2.) - theta(1. + delta / 2.)
        integrand *= self.norm_grad_gamma / delta
        self.Acav = self.hamiltonian.finegd.integrate(
            integrand,
            global_integral=False
            )
        self.Vcav = self.hamiltonian.finegd.integrate(
            theta(1.),
            global_integral=False
            )

        gamma_2b_p1 = self.gamma ** (2. * beta) + 1.
        gamma_2bm1 = self.gamma ** (2. * beta - 1.)
        self.dielectric[0][:] = 1. + (epsinf - 1.) / gamma_2b_p1
        for deps in self.dielectric[1:]:
            deps *= 2. * beta * (1. - epsinf) * \
                    gamma_2bm1 / gamma_2b_p1 ** 2

        self.update_cavity_forces()
        self.cavity_dirty = False

    def update_cavity_forces(self):
        #XXX optimize numerics
        #XXX merge with above function
        ham = self.hamiltonian
        el = self
        theta = el.theta
        gamma = el.gamma
        delta = el.params['area_delta']
        beta = el.params['exponent']
        ngg = el.norm_grad_gamma
        gg = el.grad_gamma
        term1 = theta(1. - delta / 2.) - theta(1. + delta / 2.)
        term2 = ngg / delta
        self.F_Acav = numpy.zeros((len(el.atoms), 3))
        self.F_Vcav = numpy.zeros((len(el.atoms), 3))
        for a, (R, p) in enumerate(
            zip(el.params['radii'], el.atoms.positions)
            ):
            coords = coordinates(ham.finegd, origin=p / Bohr)
            rRa = numpy.sqrt(coords[1])
            exp_a = numpy.exp(R / Bohr - rRa)
            for i in (0, 1, 2):
                rRai = coords[0][i]
                def del_ai_theta(c):
                    tmp = 2. * beta * gamma ** (2. * beta - 1.) * rRai
                    tmp *= exp_a
                    tmp /= c ** (2. * beta) * rRa
                    tmp /= ((gamma / c) ** (2. * beta) + 1.) ** 2
                    return tmp
                del_ai_ngg = gg[i] * exp_a
                del_ai_ngg *= rRa ** 2 - rRai ** 2 * (1. + rRa)
                tmp = ngg * rRa ** 3
                mask = del_ai_ngg == .0
                tmp[mask] = 1.
                del_ai_ngg /= tmp
                del_ai_ngg[mask] = .0
                del_ai_term2 = del_ai_ngg / delta
                del_ai_term1 = del_ai_theta(1. - delta / 2.)
                del_ai_term1 -= del_ai_theta(1. + delta / 2.)
                integrand = term1 * del_ai_term2 + term2 * del_ai_term1
                # - sign ???
                self.F_Acav[a][i] -= ham.finegd.integrate(
                    integrand,
                    global_integral=False
                    )
                self.F_Vcav[a][i] += ham.finegd.integrate(
                    del_ai_theta(1.),
                    global_integral=False
                    )

    def get_cavity_surface_area(self):
        return self.Acav

    def get_cavity_volume(self):
        return self.Vcav

    def calculate_forces(self, density, F_av):
        if not self.center_on_atoms:
            return
        ham = self.hamiltonian
        #XXX optimize numerics
        grad_phi_squared = ham.finegd.zeros()
        component = ham.finegd.empty()
        for i in (0, 1, 2):
            Gradient(ham.finegd, i, n=3).apply(ham.vHt_g, component)
            numpy.square(component, component)
            grad_phi_squared += component
        del component
        beta = self.params['exponent']
        epsinf = self.params['epsilon_r']
        fac = 2. * beta * (1. - epsinf) * self.gamma ** (2. * beta - 1.)
        fac /= (1. + self.gamma ** (2. * beta)) ** 2
        for a, (R, p) in enumerate(
            zip(self.params['radii'], self.atoms.positions)
            ):
            coords = coordinates(ham.finegd, origin=p / Bohr)
            distances = numpy.sqrt(coords[1])
            exp_a = numpy.exp(R / Bohr - distances)
            fac_a = fac * exp_a / distances
            for i in (0, 1, 2):
                del_ai_epsilon = fac_a * coords[0][i]
                integrand = del_ai_epsilon * grad_phi_squared
                F_av[a][i] += 1. / (8. * numpy.pi) * ham.finegd.integrate(
                    integrand, global_integral=False
                    )

    def get_cavity_surface_area_pseudo_potential(self):
        return None

    def get_cavity_volume_pseudo_potential(self):
        return None

    def get_cavity_surface_area_forces(self):
        return self.F_Acav

    def get_cavity_volume_forces(self):
        return self.F_Vcav


class ElDensElContribution(BaseElectronicContribution):
    """
    Electronic contribution to the solvation

    with cavity from density cutoff
    following Andreusi et al. J Chem Phys 136, 064102 (2012)
    """

    default_parameters = {
        'rho_min'   : NoDefault,
        'rho_max'   : NoDefault,
        'rho_delta' : 1e-6,
        'epsilon_r' : 1.0,
        }

    def set_atoms(self, atoms):
        self.n_atoms = len(atoms)

    def allocate(self):
        BaseElectronicContribution.allocate(self)
        self.finegd = self.hamiltonian.finegd
        self.E = None
        self.Acav = None
        self.Vcav = None
        self.F = numpy.empty((self.n_atoms, 3))
        self.FAcav = numpy.empty((self.n_atoms, 3))
        self.FVcav = numpy.empty((self.n_atoms, 3))
        self.VAcav = self.finegd.empty()
        self.VVcav = self.finegd.empty()
        self.gradient = [Gradient(self.finegd, i, 1.0, 3) for i in (0, 1, 2)]
        self.laplace = Laplace(self.finegd, 1.0, 3)

    def update_pseudo_potential(self, density):
        self.update_parameters()
        rho = density.nt_g
        theta, dtheta = self.theta(rho, derivative=True)
        self.update_dielectric(theta)
        self.update_volume(theta, dtheta)
        self.update_area(rho)
        Veps = (1. - self.epsinf) * dtheta
        Veps *= self.grad_squared(self.hamiltonian.vHt_g)
        Veps *= -1. / (8. * numpy.pi)
        for vt_g in self.hamiltonian.vt_sg[:self.hamiltonian.nspins]:
            vt_g += Veps
        return .0

    def update_volume(self, theta, dtheta):
        self.Vcav = theta.sum() * self.finegd.dv
        self.VVcav[...] = dtheta

    def update_area(self, rho):
        grad_rho_squared = self.grad_squared(rho)
        norm_grad_rho = numpy.sqrt(grad_rho_squared)
        d = self.rhodelta
        d2 = d * .5
        self.VAcav.fill(1. / d)
        self.VAcav *= (self.theta(rho + d2) - self.theta(rho - d2))
        self.Acav = (self.VAcav * norm_grad_rho).sum() * self.finegd.dv
        mask = self.VAcav == .0
        #mask = norm_grad_rho == .0
        grad_rho_squared[mask] = numpy.nan
        norm_grad_rho[mask] = numpy.nan
        laplace_rho = self.finegd.empty()
        self.laplace.apply(rho, laplace_rho)
        self.VAcav /= norm_grad_rho
        # XXX optimize (recycle gradient from above, use numpy, use symmetry)
        # new FD operator for di * dj * dij  ??
        grad_rho = [self.finegd.empty() for i in (0, 1, 2)]
        tmp = self.finegd.zeros()
        for i in (0, 1, 2):
            self.gradient[i].apply(rho, grad_rho[i])
        dij_rho = self.finegd.empty()
        for i in (0, 1, 2):
            for j in (0, 1, 2):
                self.gradient[i].apply(grad_rho[j], dij_rho)
                tmp += grad_rho[i] * grad_rho[j] * dij_rho
        self.VAcav *= tmp / grad_rho_squared - laplace_rho
        self.VAcav[mask] = .0

    def theta(self, rho, derivative=False):
        twopi = 2. * numpy.pi
        inside = rho > self.rhomax
        outside = rho < self.rhomin
        transition = numpy.logical_not(
            numpy.logical_or(inside, outside)
            )
        frac = (self.lnrhomax - numpy.log(rho[transition])) /\
               (self.lnrhomax - self.lnrhomin)
        t = 1. / twopi * (twopi * frac - numpy.sin(twopi * frac))
        if self.epsinf == 1.0:
            # lim_{epsinf -> 1} (epsinf - epsinf ** t) / (epsinf - 1) = 1 - t
            theta_trans = 1. - t
        else:
            power = self.epsinf ** t
            theta_trans = (self.epsinf - power) / (self.epsinf - 1.)
        theta = self.finegd.empty()
        theta[inside] = 1.
        theta[outside] = .0
        theta[transition] = theta_trans
        if not derivative:
            return theta
        dt = 1. / (self.lnrhomax - self.lnrhomin) * \
             (numpy.cos(twopi * frac) - 1.)
        dtheta = numpy.zeros_like(theta)
        if self.epsinf == 1.0:
            dtheta[transition] = -dt / rho[transition]
        else:
            dtheta[transition] = 1. / (1. - self.epsinf) * power * dt /\
                                 rho[transition]
        return theta, dtheta

    def update_parameters(self):
        self.rhomin = self.params['rho_min']
        self.rhomax = self.params['rho_max']
        self.rhodelta = self.params['rho_delta']
        self.epsinf = self.params['epsilon_r']
        self.lnrhomin = numpy.log(self.rhomin)
        self.lnrhomax = numpy.log(self.rhomax)
        self.lnepsinf = numpy.log(self.epsinf)

    def update_dielectric(self, theta):
        eps = self.dielectric[0]
        eps.fill(self.epsinf)
        eps += (1. - self.epsinf) * theta
        eps_hack = eps - self.epsinf # zero on boundary
        for i in (0, 1, 2):
            self.gradient[i].apply(eps_hack, self.dielectric[i + 1])

    def grad_squared(self, x):
        grad_x_squared = numpy.empty_like(x)
        tmp = numpy.empty_like(x)
        self.gradient[0].apply(x, grad_x_squared)
        numpy.square(grad_x_squared, grad_x_squared)
        self.gradient[1].apply(x, tmp)
        numpy.square(tmp, tmp)
        grad_x_squared += tmp
        self.gradient[2].apply(x, tmp)
        numpy.square(tmp, tmp)
        grad_x_squared += tmp
        return grad_x_squared

    def calculate_forces(self, density, F_av):
        raise NotImplementedError
        #F_av += self.F

    def get_cavity_surface_area(self):
        return self.Acav

    def get_cavity_volume(self):
        return self.Vcav

    def get_cavity_surface_area_pseudo_potential(self):
        return self.VAcav

    def get_cavity_volume_pseudo_potential(self):
        return self.VVcav


class TestElContribution(ElDensElContribution):
    default_parameters = {
        'rho_0'     : NoDefault,
        'eta'       : NoDefault,
        'rho_delta' : 1e-6,
        'epsilon_r' : 1.0,
        'T'         : NoDefault,
        }

    def update_parameters(self):
        self.rho0 = self.params['rho_0']
        self.eta = self.params['eta']
        self.rhodelta = self.params['rho_delta']
        self.epsinf = self.params['epsilon_r']
        self.kT = kB * self.params['T']

    def theta(self, rho, derivative=False):
        rho_tilde = rho / self.rho0
        u = rho_tilde ** self.eta
        g = numpy.exp(-u / self.kT)
        if derivative:
            dg = g.copy()
            dg *= rho_tilde ** (self.eta - 1.)
            dg *= -self.eta / (self.rho0 * self.kT)
            return 1. - g, -dg
        return 1. - g

    def update_dielectric(self, theta):
        eps = self.dielectric[0]
        t = (self.epsinf - 1.) * (1. - theta)
        eps[...] = (2. + self.epsinf + 2. * t) / (2. + self.epsinf - t)
        eps_hack = eps - self.epsinf # zero on boundary
        for i in (0, 1, 2):
            self.gradient[i].apply(eps_hack, self.dielectric[i + 1])

    def update_pseudo_potential(self, density):
        #if not hasattr(self, 'niter'):
        #    self.niter = 0
        #self.niter += 1
        #if self.niter < 6:
        #    self.Acav = .0
        #    self.Vcav = .0
        #    return .0
        self.update_parameters()
        rho = density.nt_g
        #rho[rho < .0] = .0
        theta, dtheta = self.theta(rho, derivative=True)
        self.update_dielectric(theta)
        self.update_volume(theta, dtheta)
        self.update_area(rho)
        g = 1. - theta
        dg = -dtheta
        Veps = (3. * (self.epsinf - 1.) * (self.epsinf + 2.)) / \
               ((self.epsinf - 1.) * g - self.epsinf - 2.) ** 2 * dg
        Veps *= self.grad_squared(self.hamiltonian.vHt_g)
        Veps *= -1. / (8. * numpy.pi)
        for vt_g in self.hamiltonian.vt_sg[:self.hamiltonian.nspins]:
            vt_g += Veps
        return .0




class STCavityContribution(BaseContribution):
    """
    Cavity formation contribution

    Calculates Ecav from macroscopic surface tension and cavity area.
    """
    default_parameters = {
        'surface_tension': NoDefault
        }

    def update_pseudo_potential(self, density):
        el = self.hamiltonian.contributions['el']
        st = self.params['surface_tension'] * Bohr ** 2 / Hartree
        E = st * el.get_cavity_surface_area()
        V = el.get_cavity_surface_area_pseudo_potential()
        if V is not None:
            Vst = st * V
            for vt_g in self.hamiltonian.vt_sg[:self.hamiltonian.nspins]:
                vt_g += Vst
        return E

    def calculate_forces(self, density, F_av):
        el = self.hamiltonian.contributions['el']
        dF_av = el.get_cavity_surface_area_forces()
        if dF_av is not None:
            gamma = self.params['surface_tension'] * Bohr ** 2 / Hartree
            F_av += gamma * dF_av


class AreaVolumeVdWContribution(BaseContribution):
    """
    two parameter model for vdW contributions

    E = alpha * A_vac + beta * V_cav

    following Andreusi et al. J Chem Phys 136, 064102 (2012)
    """
    default_parameters = {
        'surface_tension': NoDefault,
        'pressure': NoDefault
        }

    def update_pseudo_potential(self, density):
        el = self.hamiltonian.contributions['el']
        st = self.params['surface_tension'] * Bohr ** 2 / Hartree
        p = self.params['pressure'] * Bohr ** 3 / Hartree
        E = st * el.get_cavity_surface_area() + p * el.get_cavity_volume()
        VA = el.get_cavity_surface_area_pseudo_potential()
        VV = el.get_cavity_volume_pseudo_potential()
        if not (VA is None and VV is None):
            V = self.hamiltonian.finegd.zeros()
            if VA is not None:
                V += st * VA
            if VV is not None:
                V += p * VV
            for vt_g in self.hamiltonian.vt_sg[:self.hamiltonian.nspins]:
                vt_g += V
        return E

    def calculate_forces(self, density, F_av):
        el = self.hamiltonian.contributions['el']
        st = self.params['surface_tension'] * Bohr ** 2 / Hartree
        p = self.params['pressure'] * Bohr ** 3 / Hartree
        FA = el.get_cavity_surface_area_forces()
        FV = el.get_cavity_volume_forces()
        if FA is not None:
            F_av += st * FA
        if FV is not None:
            F_av += p * FV


class C6VdWContribution(BaseContribution):
    default_parameters = {
        'factor': NoDefault
        }

    def set_atoms(self, atoms):
        self.positions = atoms.positions / Bohr

    def update_pseudo_potential(self, density):
        # XXX Hack
        assert isinstance(self.hamiltonian.contributions['el'],
                          RadiiElContribution)
        theta = self.hamiltonian.contributions['el'].theta(1.)
        E = .0
        g = 1. - theta
        for pos in self.positions:
            r_square = coordinates(self.hamiltonian.finegd, origin=pos)[1]
            E += self.hamiltonian.finegd.integrate(
                g,
                1. / r_square ** 3,
                global_integral=False
                )
        E *= self.params['factor']
        # XXX todo: Kohn Sham potential in general case
        return E


class SpilledDensityVdWContribution(BaseContribution):
    default_parameters = {
        'factor': NoDefault
        }

    def update_pseudo_potential(self, density):
        # XXX Hack
        assert isinstance(self.hamiltonian.contributions['el'],
                          RadiiElContribution)
        theta = self.hamiltonian.contributions['el'].theta(1.)
        g = 1. - theta
        E = self.hamiltonian.finegd.integrate(
            g,
            density.nt_g,
            global_integral=False
            )
        E *= self.params['factor']
        for vt_g in self.hamiltonian.vt_sg[:self.hamiltonian.nspins]:
            vt_g += self.params['factor'] * g
        return E



#name -> mode -> contribution
CONTRIBUTIONS = {
    'el' : {
        None: DummyElContribution,
        MODE_RADII_CUTOFF: RadiiElContribution,
        MODE_ELECTRON_DENSITY_CUTOFF: ElDensElContribution,
        MODE_TEST: TestElContribution
        },
    'rep': {
        None: DummyContribution,
        MODE_AREA_VOLUME_VDW: AreaVolumeVdWContribution,
        MODE_C6_VDW: C6VdWContribution,
        MODE_SPILLED_DENSITY_VDW: SpilledDensityVdWContribution
        },
    'dis': {
        None: DummyContribution,
        MODE_AREA_VOLUME_VDW: AreaVolumeVdWContribution,
        MODE_C6_VDW: C6VdWContribution,
        MODE_SPILLED_DENSITY_VDW: SpilledDensityVdWContribution
        },
    'cav': {
        None: DummyContribution,
        MODE_SURFACE_TENSION: STCavityContribution
        },
    'tm' : {
        None:DummyContribution
        }
    }
