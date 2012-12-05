from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.solvation.contributions import CONTRIBUTIONS
import numpy


class SolvationRealSpaceHamiltonian(RealSpaceHamiltonian):
    def __init__(self, solvation_parameters,
                 gd, finegd, nspins, setups, timer, xc,
                 vext=None, collinear=True, psolver=None, stencil=3):
        sparams = solvation_parameters
        self.contributions = dict(
            [(c, CONTRIBUTIONS[c][p['mode']](self, p)) \
             for c, p in sparams.iteritems()]
            )
        for cname in self.contributions:
            setattr(self, 'E' + cname, None)
        self.Acav = None
        self.Vcav = None
        RealSpaceHamiltonian.__init__(
            self, gd, finegd, nspins, setups, timer, xc,
            vext, collinear, psolver, stencil
            )

    def set_atoms(self, atoms):
        for contrib in self.contributions.itervalues():
            contrib.set_atoms(atoms)

    def init_psolver(self, psolver):
        self.poisson = self.contributions['el'].make_poisson_solver(psolver)
        self.poisson.set_grid_descriptor(self.finegd)

    def initialize(self):
        for contrib in self.contributions.itervalues():
            contrib.allocate()
        RealSpaceHamiltonian.initialize(self)

    def update(self, density):
        self.timer.start('Hamiltonian')
        if self.vt_sg is None:
            self.timer.start('Initialize Hamiltonian')
            self.initialize()
            self.timer.stop('Initialize Hamiltonian')

        E = {}
        E['pot'], E['bar'], E['ext'], E['xc'] = \
                  self.update_pseudo_potential(density)
        E['pot'] += self.contributions['el'].update_pseudo_potential(density)
        Acav = self.contributions['el'].get_cavity_surface_area()
        Vcav = self.contributions['el'].get_cavity_volume()
        for cname, contrib in self.contributions.iteritems():
            if cname == 'el':
                continue
            E[cname] = contrib.update_pseudo_potential(density)
        E['kin0'] = self.calculate_kinetic_energy(density)
        W_aL = self.calculate_atomic_hamiltonians(density)
        E['kin0'], E['pot'], E['bar'], E['ext'], E['xc'] = \
                  self.update_corrections(
            density, E['kin0'], E['pot'], E['bar'], E['ext'], E['xc'], W_aL
            )

        self.timer.start('Communicate energies')
        cnames = E.keys()
        # assure same order on every rank
        cnames.sort()
        energies = numpy.array([E[cname] for cname in cnames] + [Acav, Vcav])
        self.gd.comm.sum(energies)
        for key, value in zip(cnames, energies[:-2]):
            setattr(self, 'E' + key, value)
        self.Acav = energies[-2]
        self.Vcav = energies[-1]
        self.timer.stop('Communicate energies')
        self.timer.stop('Hamiltonian')

    def calculate_forces(self, dens, F_av):
        for contrib in self.contributions.itervalues():
            contrib.calculate_forces(dens, F_av)
        RealSpaceHamiltonian.calculate_forces(self, dens, F_av)

    def get_energy(self, occupations):
        self.Ekin = self.Ekin0 + occupations.e_band
        self.S = occupations.e_entropy
        self.Eel = self.Ekin + self.Epot + self.Eext + \
                   self.Ebar + self.Exc - self.S
        Etot = .0
        for cname, contrib in self.contributions.iteritems():
            Etot += getattr(self, 'E' + cname)
        self.Etot = Etot
        return self.Etot
