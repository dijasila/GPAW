import numpy as np
from ase.units import Bohr, Ha

from gpaw.tb.repulsion import evaluate_pair_potential
from gpaw.hamiltonian import Hamiltonian


class TBPoissonSolver:
    def get_description(self):
        return 'TB'


class LFC:
    def set_positions(self, spos_ac, atom_partition):
        pass


class TBHamiltonian(Hamiltonian):
    poisson = TBPoissonSolver()
    xnpoisson = 0

    def __init__(self,
                 repulsion_parameters,
                 **kwargs):
        self.repulsion_parameters = repulsion_parameters
        Hamiltonian.__init__(self, **kwargs)

    def set_positions(self, spos_ac, atom_partition):
        cell_cv = self.gd.cell_cv * Bohr
        position_av = spos_ac @ cell_cv

        energy, forces = evaluate_pair_potential(
            self.repulsion_parameters,
            [setup.symbol for setup in self.setups],
            position_av, cell_cv, self.gd.pbc_c)

        self.e_pair = energy / Ha
        self.force_av = forces * (Bohr / Ha)

        self.atom_partition = atom_partition
        self.atomdist = self.redistributor.get_atom_distributions(spos_ac)

        self.positions_set = True

    def update(self, dens):
        D_asp = self.atomdist.to_work(dens.D_asp)
        self.dH_asp = self.setups.empty_atomic_matrix(1, D_asp.partition)

        e_kinetic = 0.0
        e_zero = 0.0
        e_xc = 0.0
        for a, D_sp in D_asp.items():
            setup = self.setups[a]
            D_p = D_sp[0]
            dH_p = setup.K_p + setup.MB_p
            e_kinetic += np.dot(setup.K_p, D_p) + setup.Kc
            e_zero += setup.MB + np.dot(setup.MB_p, D_p)
            self.dH_asp[a][0] = dH_p

        e_coulomb = self.e_pair
        e_external = 0.0
        return np.array([e_kinetic, e_coulomb, e_zero, e_external, e_xc])

    def get_energy(self, e_entropy, wfs):
        """Sum up all eigenvalues weighted with occupation numbers"""
        self.e_band = wfs.calculate_band_energy()
        self.e_kinetic = self.e_kinetic0 + self.e_band
        self.e_entropy = e_entropy

        self.e_total_free = (self.e_kinetic + self.e_coulomb +
                             self.e_external + self.e_zero + self.e_xc +
                             self.e_entropy)
        self.e_total_extrapolated = (
            self.e_total_free +
            wfs.occupations.extrapolate_factor * e_entropy)

        return self.e_total_free

    def xupdate_pseudo_potential(self, dens):
        energies = np.array([self.e_pair, 0.0, 0.0, 0.0])
        return energies

    def xcalculate_kinetic_energy(self, density):
        return 0.0

    def xcalculate_atomic_hamiltonians(self, dens):
        from gpaw.arraydict import ArrayDict

        def getshape(a):
            return sum(2 * l + 1 for l, _ in enumerate(self.setups[a].ghat_l)),

        W_aL = ArrayDict(self.atomdist.aux_partition, getshape, float)
        for W_L in W_aL.values():
            W_L[:] = 0.0

        return self.atomdist.to_work(self.atomdist.from_aux(W_aL))
