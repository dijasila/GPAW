import numpy as np
from ase.units import Bohr, Ha

from gpaw.hamiltonian import Hamiltonian
from gpaw.tb.repulsion import evaluate_pair_potential


class TBPoissonSolver:
    def get_description(self):
        return 'TB'


class LFC:
    def set_positions(self, spos_ac, atom_partition):
        pass


class TBHamiltonian(Hamiltonian):
    poisson = TBPoissonSolver()
    npoisson = 0

    def __init__(self, repulsion_parameters, *args, **kwargs):
        Hamiltonian.__init__(self, *args, **kwargs)
        self.vbar = LFC()
        self.repulsion_parameters = repulsion_parameters

    def set_positions(self, spos_ac, atom_partition):
        cell_cv = self.gd.cell_cv * Bohr
        position_av = spos_ac @ cell_cv

        energy, forces = evaluate_pair_potential(
            self.repulsion_parameters,
            [setup.symbol for setup in self.setups],
            position_av, cell_cv, self.gd.pbc_c)

        self.e_pair = energy / Ha
        self.force_av = forces * (Bohr / Ha)

        Hamiltonian.set_positions(self, spos_ac, atom_partition)

    def update_pseudo_potential(self, dens):
        energies = np.array([self.e_pair, 0.0, 0.0, 0.0])
        return energies

    def calculate_kinetic_energy(self, density):
        return 0.0

    def calculate_atomic_hamiltonians(self, dens):
        from gpaw.arraydict import ArrayDict

        def getshape(a):
            return sum(2 * l + 1 for l, _ in enumerate(self.setups[a].ghat_l)),

        W_aL = ArrayDict(self.atomdist.aux_partition, getshape, float)
        for W_L in W_aL.values():
            W_L[:] = 0.0

        return self.atomdist.to_work(self.atomdist.from_aux(W_aL))
