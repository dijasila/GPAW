from math import pi

import numpy as np

from .density import Density
from .hamiltonian import Hamiltonian
from .lcao.eigensolver import DirectLCAO
from .lcao.tci import TCIExpansions
from .spline import Spline
from .wavefunctions.lcao import LCAOWaveFunctions
from .wavefunctions.mode import Mode


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

    def __init__(self, *args, **kwargs):
        LCAOWaveFunctions.__init__(self, *args, **kwargs)

        vtphit = {}  # Dict[Setup, List[Spline]]
        for setup in self.setups.setups.values():
            vt = setup.vt
            vtphit_j = []
            for phit in setup.phit_j:
                rc = phit.get_cutoff()
                r_g = np.linspace(0, rc, 150)
                vt_g = vt.map(r_g) / (4 * pi)**0.5
                vtphit_j.append(Spline(phit.l, rc, vt_g * phit.map(r_g)))
            vtphit[setup] = vtphit_j

        self.vtciexpansions = TCIExpansions([s.phit_j for s in self.setups],
                                            [vtphit[s] for s in self.setups],
                                            self.tciexpansions.I_a)

    def set_positions(self, spos_ac, *args, **kwargs):
        LCAOWaveFunctions.set_positions(self, spos_ac, *args, **kwargs)
        manytci = self.vtciexpansions.get_manytci_calculator(
            self.setups, self.gd, spos_ac, self.kd.ibzk_qc, self.dtype,
            self.timer)
        manytci.Pindices = manytci.Mindices
        my_atom_indices = self.basis_functions.my_atom_indices
        self.Vt_qMM = [Vt_MM.toarray()
                       for Vt_MM in manytci.P_qIM(my_atom_indices)]
        print(self.Vt_qMM[0]);asdf


class TBEigenSolver(DirectLCAO):
    def iterate(self, ham, wfs):
        for kpt in wfs.kpt_u:
            self.iterate_one_k_point(ham, wfs, kpt, [wfs.Vt_qMM[kpt.q]])


class TBDensity(Density):
    def set_positions(self, spos_ac, atom_partition):
        self.set_positions_without_ruining_everything(spos_ac, atom_partition)

    def initialize_density_from_atomic_densities(self, basis_functions, f_asi):
        pass

    def mix(self, comp_charge):
        self.error = 0.0

    def normalize(self, comp_charge):
        pass

    def calculate_pseudo_density(self, wfs):
        pass

    def calculate_dipole_moment(self):
        return np.zeros(3)


class TBPoissonSolver:
    def get_description(self):
        return 'TB'


class LFC:
    def set_positions(self, spos_ac, atom_partition):
        pass


class TBHamiltonian(Hamiltonian):
    poisson = TBPoissonSolver()
    npoisson = 0

    def __init__(self, *args, **kwargs):
        Hamiltonian.__init__(self, *args, **kwargs)
        self.vbar = LFC()

    def update_pseudo_potential(self, dens):
        e_coulomb = 0.0
        energies = np.array([e_coulomb, 0.0, 0.0, 0.0])
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
