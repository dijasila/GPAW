from collections import Counter
from typing import Dict, List

import numpy as np
from ase.units import Bohr, Ha

from gpaw.tb.repulsion import evaluate_pair_potential
from gpaw.hamiltonian import Hamiltonian
from gpaw.density import Density
from gpaw.hints import Array2D
from gpaw.utilities import pack


class TBPoissonSolver:
    def get_description(self):
        return 'TB'


class LFC:
    def set_positions(self, spos_ac, atom_partition):
        pass


class TBXC:
    def __init__(self, xc):
        self.xc = xc
        self.name = xc.name
        self.type = xc.type

        self.e_xc: float = np.nan
        self.dH_asp: Dict[int, Array2D] = {}

    def set_positions(self, spos_ac):
        pass

    def calculate_paw_correction(self, setup, D_sp, dH_sp, a):
        if a not in self.dH_asp:
            dH0_sp = np.zeros_like(D_sp)
            self.e_xc = self.xc.calculate_paw_correction(
                setup, D_sp, dH0_sp, a)
            self.dH_asp[a] = dH0_sp

        dH_sp += self.dH_asp[a]

        return self.e_xc

    def get_kinetic_energy_correction(self):
        return 0.0

    def summary(self, logger):
        pass


def reference_occupation_numbers(setup) -> List[float]:
    f_i: List[float] = sum(([f] * (2 * l + 1)
                            for f, l in zip(setup.f_j, setup.l_j)),
                           [])
    return f_i


def calculate_reference_energies(setups, xc):
    count = Counter(setups)
    e_kin = 0.0
    e_zero = 0.0
    e_coulomb = 0.0
    e_xc = 0.0
    for setup, n in count.items():
        f_i = reference_occupation_numbers(setup)
        D_p = pack(np.diag(f_i))
        D_sp = D_p[np.newaxis]
        dH_sp = np.zeros_like(D_sp)

        e_xc += n * xc.calculate_paw_correction(setup, D_sp, dH_sp, a=None)
        e_kin += n * (np.dot(setup.K_p, D_p) + setup.Kc)
        e_zero += n * (setup.MB + np.dot(setup.MB_p, D_p))
        e_coulomb += n * (setup.M + D_p.dot(setup.M_p + setup.M_pp.dot(D_p)))

        e_kin -= n * D_p @ (dH_sp[0] +
                            setup.K_p +
                            setup.MB_p +
                            setup.M_p + 2 * setup.M_pp @ D_p +
                            setup.Delta_pL[:, 0] * setup.W)
        e_kin += n * np.dot(setup.data.eps_j, setup.f_j)

    return e_xc, e_kin, e_zero, e_coulomb


class TBHamiltonian(Hamiltonian):
    poisson = TBPoissonSolver()
    npoisson = 0

    def __init__(self,
                 repulsion_parameters,
                 xc,
                 **kwargs):
        self.repulsion_parameters = repulsion_parameters
        self.vbar = LFC()

        xc = TBXC(xc)
        Hamiltonian.__init__(self, xc=xc, **kwargs)

        self.e_xc_0, self.e_kin_0, self.e_zero_0, self.e_coulomb_0 = (
            calculate_reference_energies(self.setups, xc))

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

    def update_pseudo_potential(self, dens: Density):
        e_coulomb = self.e_pair - self.e_coulomb_0
        e_zero = -self.e_zero_0
        e_external = 0.0
        e_xc = -self.e_xc_0
        return np.array([e_coulomb, e_zero, e_external, e_xc])

    def calculate_kinetic_energy(self, dens: Density) -> float:
        return -self.e_kin_0

    def calculate_atomic_hamiltonians(self, dens):
        from gpaw.arraydict import ArrayDict

        def getshape(a):
            return sum(2 * l + 1 for l, _ in enumerate(self.setups[a].ghat_l)),

        W_aL = ArrayDict(self.atomdist.aux_partition, getshape, float)
        for a, W_L in W_aL.items():
            W_L[:] = 0.0
            W_L[0] = self.setups[a].W

        return self.atomdist.to_work(self.atomdist.from_aux(W_aL))
