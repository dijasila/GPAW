from gpaw.solvation.poisson import SolvationPoissonSolver
from gpaw.utilities.tools import coordinates
from gpaw.fd_operators import Gradient
from gpaw.poisson import PoissonSolver
from types import StringTypes
from ase.units import Hartree, Bohr
import numpy

MODE_ELECTRON_DENSITY_CUTOFF = 'electron_density_cutoff'
MODE_RADII_CUTOFF = 'radii_cutoff'
MODE_SURFACE_TENSION = 'surface_tension'


class NoDefault:
    pass


class BaseContribution:
    """
    base class for all solvation contributions

    methods are called by the hamiltonian in the
    order they appear in the source code below
    set_* is called only if the corresponding value has changed
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
        updates the pseudo potential

        returns energy of the contribution in Hartree
        """
        raise NotImplementedError

    def calculate_forces(self, density, F_av):
        """
        updates F_av
        """
        raise NotImplementedError


class BaseElectronicContribution(BaseContribution):
    """
    base class for electronic solvation contributions
    """

    def make_poisson_solver(self, psolver):
        """
        creates and returns a Poisson solver

        or uses or modifies the one given by 'psolver'
        """
        raise NotImplementedError

    def update_pseudo_potential(self, density):
        """
        updates the pseudo potential

        return value is ignored, since the additional energy
        is already included in the Hartree energy of the modified
        Poisson equation!
        """
        raise NotImplementedError

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


class DummyContribution(BaseContribution):
    def update_pseudo_potential(self, density):
        return 0.

    def calculate_forces(self, density, F_av):
        pass


class DummyElContribution(BaseElectronicContribution):
    def make_poisson_solver(self, psolver):
        if psolver is None:
            psolver = PoissonSolver(nn=3, relax='J')
        return psolver

    def update_pseudo_potential(self, density):
        pass

    def get_cavity_surface_area(self):
        return 0.

    def get_cavity_volume(self):
        return 0.

    def calculate_forces(self, density, F_av):
        pass


class SmearedCavityElectronicContribution(BaseElectronicContribution):
    def init(self):
        BaseElectronicContribution.init(self)
        self.weights = []

    def make_poisson_solver(self, psolver):
        if psolver is None:
            psolver = SolvationPoissonSolver(
                nn=3, relax='J', op_weights=self.weights
                )
        else:
            psolver.op_weights = self.weights
        return psolver

    def allocate(self):
        del self.weights[:]
        finegd = self.hamiltonian.finegd
        self.weights.extend([gd.empty() for gd in (finegd, ) * 4])
        self.update_weights(None)

    def update_weights(self, density):
        raise NotImplementedError

    def update_pseudo_potential(self, density):
        self.update_weights(density)


class RadiiElContribution(SmearedCavityElectronicContribution):
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
        SmearedCavityElectronicContribution.init(self)
        self.atoms = None
        self.cavity_dirty = True
        centers = self.params['centers']
        self.center_on_atoms = isinstance(centers, StringTypes) and \
                               centers.lower().strip() == 'atoms'
        self.Acav = None

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

    def update_weights(self, density):
        if not self.cavity_dirty:
            return
        centers = self.get_centers()
        ham = self.hamiltonian
        self.gamma = ham.finegd.zeros()
        for deps in self.weights[1:]:
            deps.fill(.0)
        for R, center in zip(self.params['radii'], centers):
            coords = coordinates(ham.finegd, origin=center / Bohr)
            distances = numpy.sqrt(coords[1])
            tmp = numpy.exp(R / Bohr - distances)
            self.gamma += tmp
            for i in (0, 1, 2):
                self.weights[1 + i] -= tmp * coords[0][i] / distances
        self.grad_gamma = [self.weights[1 + i].copy() for i in (0, 1, 2)]

        #XXX optimize numerics
        beta = self.params['exponent']
        epsinf = self.params['epsilon_r']
        delta = self.params['area_delta']
        self.norm_grad_gamma = .0
        for i in (0, 1, 2):
            self.norm_grad_gamma += self.weights[i + 1] ** 2
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
        self.weights[0][:] = 1. + (epsinf - 1.) / gamma_2b_p1
        for deps in self.weights[1:]:
            deps *= 2. * beta * (1. - epsinf) * \
                    gamma_2bm1 / gamma_2b_p1 ** 2
        self.cavity_dirty = False

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


class ElDensElContribution(SmearedCavityElectronicContribution):
    """
    Electronic contribution to the solvation

    with cavity from density cutoff
    following Sanchez et al J. Chem. Phys. 131 (2009) 174108
    """

    default_parameters = {
        'cutoff'    : NoDefault,
        'exponent'  : NoDefault,
        'epsilon_r' : 1.0,
        'area_delta': NoDefault
        }

    def init(self):
        raise NotImplementedError
        #SmearedCavityElectronicContribution.init(self)

        #self.hamiltonian not initialized at this point!
        #if self.hamiltonian.nspins != 1 or not self.hamiltonian.collinear:
        #    raise NotImplementedError(
        #        'Solvation only supports spin-paired calculations '
        #        'up to now for electron density cutoff mode.'
        #        )

    def update_weights(self, density):
        raise NotImplementedError
        ## epsinf = self.pcm_params['epsinf']
        ## rho0 = self.pcm_params['cutoff']
        ## beta = self.pcm_params['beta']
        ## tmp = (density.nt_g / rho0) ** (2. * beta)
        ## self.weights[0][:] = 1. + (epsinf - 1.) / 2. * \
        ##                      (1. + (1. - tmp) / (1. + tmp))
        ## for op, w in zip(self.weight_ops, self.weights[1:]):
        ##     op.apply(self.weights[0], w)


class STCavityContribution(BaseContribution):
    """
    Cavity formation contribution

    Calculates Ecav from macroscopic surface tension and cavity area.
    """
    default_parameters = {
        'surface_tension': NoDefault
        }

    def update_pseudo_potential(self, density):
        return self.params['surface_tension'] * Bohr ** 2 / Hartree * \
               self.hamiltonian.contributions['el'].get_cavity_surface_area()

    def calculate_forces(self, density, F_av):
        #XXX optimize numerics
        ham = self.hamiltonian
        el = ham.contributions['el']
        theta = el.theta
        gamma = el.gamma
        delta = el.params['area_delta']
        beta = el.params['exponent']
        ngg = el.norm_grad_gamma
        gg = el.grad_gamma
        term1 = theta(1. - delta / 2.) - theta(1. + delta / 2.)
        term2 = ngg / delta
        stension = self.params['surface_tension'] * Bohr ** 2 / Hartree
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
                F_av[a][i] -= stension * ham.finegd.integrate(
                    integrand,
                    global_integral=False
                    )


#name -> mode -> contribution
CONTRIBUTIONS = {
    'el' : {
        None: DummyElContribution,
        MODE_RADII_CUTOFF: RadiiElContribution,
        MODE_ELECTRON_DENSITY_CUTOFF: ElDensElContribution
        },
    'rep': {
        None: DummyContribution
        },
    'dis': {
        None: DummyContribution
        },
    'cav': {
        None: DummyContribution,
        MODE_SURFACE_TENSION: STCavityContribution
        },
    'tm' : {
        None:DummyContribution
        }
    }
