from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.solvation.poisson import WeightedFDPoissonSolver
from gpaw.fd_operators import Gradient
import numpy as np


class SolvationRealSpaceHamiltonian(RealSpaceHamiltonian):
    def __init__(
        self, cavity, dielectric, interactions,
        gd, finegd, nspins, setups, timer, xc,
        vext=None, collinear=True, psolver=None,
        stencil=3, world=None
        ):
        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions
        cavity.set_grid_descriptor(finegd)
        dielectric.set_grid_descriptor(finegd)
        if psolver is None:
            psolver = WeightedFDPoissonSolver()
        psolver.set_dielectric(self.dielectric)
        self.gradient = None
        self.cav_dirty = True
        RealSpaceHamiltonian.__init__(
            self,
            gd, finegd, nspins, setups, timer, xc,
            vext, collinear, psolver,
            stencil, world
            )
        for ia in interactions:
            setattr(self, 'E_' + ia.subscript, None)
            ia.init(self)

    def update_atoms(self, atoms):
        cav_dirty = self.cavity.update_atoms(atoms)
        self.cav_dirty = self.cav_dirty or cav_dirty
        for ia in self.interactions:
            ia.update_atoms(atoms)

    def initialize(self):
        self.gradient = [
            Gradient(self.finegd, i, 1.0, self.poisson.nn) for i in (0, 1, 2)
            ]
        self.dielectric.allocate()
        for ia in self.interactions:
            ia.allocate()
        RealSpaceHamiltonian.initialize(self)

    def update(self, density):
        self.timer.start('Hamiltonian')
        if self.vt_sg is None:
            self.timer.start('Initialize Hamiltonian')
            self.initialize()
            self.timer.stop('Initialize Hamiltonian')

        cav_dirty = self.cavity.update_el_density(density)
        self.cav_dirty = self.cav_dirty or cav_dirty
        if self.cav_dirty:
            self.dielectric.update_cavity(self.cavity)
            self.cav_dirty = False

        Epot, Ebar, Eext, Exc = self.update_pseudo_potential(density)
        Eias = [i.update_pseudo_potential(density) for i in self.interactions]

        Ekin = self.calculate_kinetic_energy(density)
        W_aL = self.calculate_atomic_hamiltonians(density)
        Ekin, Epot, Ebar, Eext, Exc = self.update_corrections(
            density, Ekin, Epot, Ebar, Eext, Exc, W_aL
            )

        energies = np.array([Ekin, Epot, Ebar, Eext, Exc] + Eias)
        self.timer.start('Communicate energies')
        self.gd.comm.sum(energies)
        # Make sure that all CPUs have the same energies
        self.world.broadcast(energies, 0)
        self.timer.stop('Communicate energies')
        (self.Ekin0, self.Epot, self.Ebar, self.Eext, self.Exc) = energies[:5]
        for E, ia in zip(energies[5:], self.interactions):
            setattr(self, 'E_' + ia.subscript, E)

        #self.Exc += self.Enlxc
        #self.Ekin0 += self.Enlkin

        self.timer.stop('Hamiltonian')

    def update_pseudo_potential(self, density):
        ret = RealSpaceHamiltonian.update_pseudo_potential(self, density)
        del_g_del_n_g = self.cavity.del_g_del_n_g
        # does cavity depend on electron density?
        # XXX optimize numerics
        if np.any(del_g_del_n_g != .0):
            del_eps_del_g_g = self.dielectric.del_eps_del_g_g
            Veps = -1. / (8. * np.pi) * del_eps_del_g_g * del_g_del_n_g
            Veps *= self.grad_squared(self.vHt_g)
            for vt_g in self.vt_sg[:self.nspins]:
                vt_g += Veps
        return ret

    def calculate_forces(self, dens, F_av):
        self.el_force_correction(dens, F_av)
        for ia in self.interactions:
            ia.update_forces(dens, F_av)
        return RealSpaceHamiltonian.calculate_forces(
            self, dens, F_av
            )

    def el_force_correction(self, dens, F_av):
        del_eps_del_g_g = self.dielectric.del_eps_del_g_g
        fixed = 1. / (8. * np.pi) * del_eps_del_g_g * \
            self.grad_squared(self.vHt_g)  # XXX grad_vHt_g inexact in bmgs
        for a, fa in enumerate(F_av):
            dRa = self.cavity.get_atomic_position_derivative(a)
            # does cavity depend on atomic positions?
            if np.any(dRa != 0):
                for v in (0, 1, 2):
                    fa[v] += self.finegd.integrate(
                        fixed * dRa[v],
                        global_integral=False
                        )

    def get_energy(self, occupations):
        self.Ekin = self.Ekin0 + occupations.e_band
        self.S = occupations.e_entropy
        self.Eel = self.Ekin + self.Epot + self.Eext + \
                   self.Ebar + self.Exc - self.S
        Etot = self.Eel
        for ia in self.interactions:
            Etot += getattr(self, 'E_' + ia.subscript)
        self.Etot = Etot
        return self.Etot

    def grad_squared(self, x):
        gs = np.empty_like(x)
        tmp = np.empty_like(x)
        self.gradient[0].apply(x, gs)
        np.square(gs, gs)
        self.gradient[1].apply(x, tmp)
        np.square(tmp, tmp)
        gs += tmp
        self.gradient[2].apply(x, tmp)
        np.square(tmp, tmp)
        gs += tmp
        return gs
