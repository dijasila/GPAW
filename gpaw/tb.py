import numpy as np

from gpaw.density import Density
from gpaw.hamiltonian import Hamiltonian
from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.wavefunctions.lcao import LCAOWaveFunctions
from gpaw.wavefunctions.mode import Mode


class TB(Mode):
    name = 'tb'
    interpolation = 1
    force_complex_dtype = False

    def __init__(self) -> None:
        pass

    def __call__(self, ksl, **kwargs) -> 'TBWaveFunctions':
        return TBWaveFunctions(ksl, **kwargs)


class TBWaveFunctions(LCAOWaveFunctions):
    mode = 'tb'


class TBEigenSolver(DirectLCAO):
    def iterate(self, ham, wfs):
        Vt_xMM = np.array([[[-0.2]]])
        for kpt in wfs.kpt_u:
            assert kpt.s == 0
            self.iterate_one_k_point(ham, wfs, kpt, Vt_xMM)


class TBDensity(Density):
    def set_positions(self, spos_ac, atom_partition):
        self.set_positions_without_ruining_everything(spos_ac, atom_partition)

    def initialize_from_atomic_densities(self, basis_functions):
        """Initialize D_asp and Q_aL from atomic densities."""

        self.log('Density initialized from atomic densities')

        self.update_atomic_density_matrices(
            self.setups.empty_atomic_matrix(self.ncomponents,
                                            self.atom_partition))

        f_asi = {}
        for a in basis_functions.atom_indices:
            f_asi[a] = self.get_initial_occupations(a)

        # D_asp does not have the same distribution as the basis functions,
        # so we have to loop over atoms separately.
        for a in self.D_asp:
            f_si = f_asi.get(a)
            if f_si is None:
                f_si = self.get_initial_occupations(a)
            self.D_asp[a][:] = self.setups[a].initialize_density_matrix(f_si)

        self.calculate_normalized_charges_and_mix()

    def mix(self, comp_charge):
        self.error = 0.0

    def normalize(self, comp_charge):
        pass

    def calculate_dipole_moment(self):
        return np.zeros(3)


class TBPoissonSolver:
    def get_description(self):
        return 'TB'


class TBHamiltonian(Hamiltonian):
    poisson = TBPoissonSolver()
    npoisson = 0

    def set_positions(self, spos_ac, atom_partition):
        self.xc.set_positions(spos_ac)
        self.set_positions_without_ruining_everything(spos_ac, atom_partition)
        self.positions_set = True

    def update_pseudo_potential(self, dens):
        energies = np.array([0.0, 0.0, 0.0, 0.0])
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

if __name__ == '__main__':
    from ase import Atoms
    from gpaw import GPAW
    a = Atoms('H')
    a.calc = GPAW(mode='tb')
    a.get_potential_energy()
