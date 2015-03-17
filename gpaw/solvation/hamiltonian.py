from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.solvation.poisson import WeightedFDPoissonSolver
from gpaw.fd_operators import Gradient
import numpy as np


class SolvationRealSpaceHamiltonian(RealSpaceHamiltonian):
    def __init__(
        self, cavity, dielectric, interactions,
        gd, finegd, nspins, setups, timer, xc, world,
        kptband_comm, vext=None, collinear=True, psolver=None,
        stencil=3
        ):
        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions
        cavity.set_grid_descriptor(finegd)
        dielectric.set_grid_descriptor(finegd)
        for ia in interactions:
            ia.set_grid_descriptor(finegd)
        if psolver is None:
            psolver = WeightedFDPoissonSolver()
        psolver.set_dielectric(self.dielectric)
        self.gradient = None
        RealSpaceHamiltonian.__init__(
            self,
            gd, finegd, nspins, setups, timer, xc, world,
            kptband_comm, vext, collinear, psolver,
            stencil
            )
        for ia in interactions:
            setattr(self, 'E_' + ia.subscript, None)
        self.new_atoms = None
        self.vt_ia_g = None

    def estimate_memory(self, mem):
        RealSpaceHamiltonian.estimate_memory(self, mem)
        solvation = mem.subnode('Solvation')
        for name, obj in [
            ('Cavity', self.cavity),
            ('Dielectric', self.dielectric),
            ] + \
            [('Interaction: ' + ia.subscript, ia) for ia in self.interactions]:
            obj.estimate_memory(solvation.subnode(name))

    def update_atoms(self, atoms):
        self.new_atoms = atoms.copy()

    def initialize(self):
        self.gradient = [
            Gradient(self.finegd, i, 1.0, self.poisson.nn) for i in (0, 1, 2)
            ]
        self.vt_ia_g = self.finegd.zeros()
        self.cavity.allocate()
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

        cavity_changed = self.cavity.update(self.new_atoms, density)
        if cavity_changed:
            self.cavity.update_vol_surf()
            self.dielectric.update(self.cavity)

        Epot, Ebar, Eext, Exc = self.update_pseudo_potential(density)
        ia_changed = [
            ia.update(
                self.new_atoms,
                density,
                self.cavity if cavity_changed else None
                ) for ia in self.interactions
            ]
        if np.any(ia_changed):
            self.vt_ia_g.fill(.0)
            for ia in self.interactions:
                if ia.depends_on_el_density:
                    self.vt_ia_g += ia.delta_E_delta_n_g
                if self.cavity.depends_on_el_density:
                    self.vt_ia_g += (ia.delta_E_delta_g_g *
                                     self.cavity.del_g_del_n_g)
        if len(self.interactions) > 0:
            for vt_g in self.vt_sg[:self.nspins]:
                vt_g += self.vt_ia_g
        Eias = [ia.E for ia in self.interactions]

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
        self.cavity.communicate_vol_surf(self.world)
        self.timer.stop('Communicate energies')
        (self.Ekin0, self.Epot, self.Ebar, self.Eext, self.Exc) = energies[:5]
        for E, ia in zip(energies[5:], self.interactions):
            setattr(self, 'E_' + ia.subscript, E)

        #self.Exc += self.Enlxc
        #self.Ekin0 += self.Enlkin

        self.new_atoms = None
        self.timer.stop('Hamiltonian')

    def update_pseudo_potential(self, density):
        ret = RealSpaceHamiltonian.update_pseudo_potential(self, density)
        if not self.cavity.depends_on_el_density:
            return ret
        del_g_del_n_g = self.cavity.del_g_del_n_g
        # XXX optimize numerics
        del_eps_del_g_g = self.dielectric.del_eps_del_g_g
        Veps = -1. / (8. * np.pi) * del_eps_del_g_g * del_g_del_n_g
        Veps *= self.grad_squared(self.vHt_g)
        for vt_g in self.vt_sg[:self.nspins]:
            vt_g += Veps
        return ret

    def calculate_forces(self, dens, F_av):
        # XXX reorganize
        self.el_force_correction(dens, F_av)
        for ia in self.interactions:
            if self.cavity.depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    del_g_del_r_vg = self.cavity.get_del_r_vg(a, dens)
                    for v in (0, 1, 2):
                        F_v[v] -= self.finegd.integrate(
                            ia.delta_E_delta_g_g * del_g_del_r_vg[v],
                            global_integral=False
                            )
            if ia.depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    del_E_del_r_vg = ia.get_del_r_vg(a, dens)
                    for v in (0, 1, 2):
                        F_v[v] -= self.finegd.integrate(
                            del_E_del_r_vg[v],
                            global_integral=False
                            )
        return RealSpaceHamiltonian.calculate_forces(
            self, dens, F_av
            )

    def el_force_correction(self, dens, F_av):
        if not self.cavity.depends_on_atomic_positions:
            return
        del_eps_del_g_g = self.dielectric.del_eps_del_g_g
        fixed = 1. / (8. * np.pi) * del_eps_del_g_g * \
            self.grad_squared(self.vHt_g)  # XXX grad_vHt_g inexact in bmgs
        for a, F_v in enumerate(F_av):
            del_g_del_r_vg = self.cavity.get_del_r_vg(a, dens)
            for v in (0, 1, 2):
                F_v[v] += self.finegd.integrate(
                    fixed * del_g_del_r_vg[v],
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
        # XXX ugly
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
