from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase.data import covalent_radii, atomic_numbers
from ase.neighborlist import neighbor_list
from ase.units import Bohr, Ha
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.core.domain import Domain
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.wave_functions import WaveFunctions


class NoGrid(Domain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gd = SimpleNamespace(
            get_grid_spacings=lambda: [0, 0, 0],
            cell_cv=self.cell_cv,
            N_c=[0, 0, 0],
            dv=0.0)

    def empty(self, shape=(), comm=serial_comm):
        return DummyFunctions(self, shape, comm)


class DummyFunctions(DistributedArrays[NoGrid]):
    def __init__(self,
                 grid: NoGrid,
                 dims: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm):
        DistributedArrays. __init__(self, dims, (),
                                    comm, grid.comm, None, np.nan,
                                    grid.dtype, transposed=False)
        self.desc = grid

    def integrate(self):
        return np.ones(self.dims)

    def new(self):
        return self


class PSCoreDensities:
    def __init__(self, grid, fracpos_ac):
        self.layout = AtomArraysLayout([1] * len(fracpos_ac),
                                       grid.comm)

    def to_uniform_grid(self, out, scale):
        pass


class DummyBasisSet:
    def add_to_density(self, data, f_asi):
        pass


class FakePotentialCalculator(PotentialCalculator):
    def __init__(self,
                 xc,
                 setups,
                 nct_R,
                 atoms):
        super().__init__(xc, None, setups, nct_R)
        self.atoms = atoms.copy()
        self.force_av = None
        self.stress_vv = None

    def calculate_charges(self, vHt_r):
        return {a: np.zeros(9) for a, setup in enumerate(self.setups)}

    def _calculate(self, density, vHt_r):
        vt_sR = density.nt_sR

        atoms = self.atoms
        energy, force_av, stress_vv = pairpot(atoms)
        energy /= Ha
        self.force_av = force_av * Bohr / Ha

        vol = abs(np.linalg.det(atoms.cell[atoms.pbc][:, atoms.pbc]))
        self.stress_vv = stress_vv / vol * Bohr**atoms.pbc.sum() / Ha

        return {'kinetic': 0.0,
                'coulomb': 0.0,
                'zero': 0.0,
                'xc': energy,
                'external': 0.0}, vt_sR, vHt_r

    def _move(self, fracpos_ac, ndensities):
        self.atoms.set_scaled_positions(fracpos_ac)
        self.force_av = None
        self.stress_vv = None

    def force_contributions(self, state):
        return {}, {}, {a: force_v[np.newaxis]
                        for a, force_v
                        in enumerate(self.force_av)}

    def stress_contribution(self, state):
        return self.stress_vv


class DummyXC:
    no_forces = False
    xc = None

    def calculate_paw_correction(self, setup, D_sp, dH_sp):
        return 0.0


class DummyWaveFunctions(WaveFunctions):
    def __init__(self,
                 domain_comm,
                 band_comm,
                 nbands,
                 spin: int | None,
                 setups,
                 fracpos_ac,
                 weight: float = 1.0,
                 spin_degeneracy: int = 2):
        super().__init__(spin, setups, fracpos_ac, weight, spin_degeneracy)
        self.domain_comm = domain_comm
        self.band_comm = band_comm
        self.nbands = nbands

    def force_contribution(self, dH_asii, F_av):
        pass

    def move(self, fracpos_ac):
        pass


class DummySCFLoop:
    def __init__(self, occ_calc, pot_calc):
        self.occ_calc = occ_calc
        self.pot_calc = pot_calc

    def iterate(self,
                state,
                convergence=None,
                maxiter=None,
                log=None):
        for wfs in state.ibzwfs:
            wfs._eig_n = np.arange(wfs.nbands)
        state.ibzwfs.calculate_occs(self.occ_calc)
        yield
        state.potential, state.vHt_x, _ = self.pot_calc.calculate(
            state.density, state.vHt_x)


class FakeDFTComponentsBuilder(DFTComponentsBuilder):
    name = 'test'
    interpolation = ''

    def check_cell(self, cell):
        pass

    def create_basis_set(self):
        return DummyBasisSet()

    def create_uniform_grids(self):
        grid = NoGrid(
            self.atoms.cell,
            self.atoms.pbc,
            dtype=self.dtype,
            comm=self.communicators['d'])
        return grid, grid

    def get_pseudo_core_densities(self):
        return PSCoreDensities(self.grid, self.fracpos_ac)

    def create_potential_calculator(self):
        xc = DummyXC()
        return FakePotentialCalculator(xc, self.setups, self.nct_R, self.atoms)

    def create_ibz_wave_functions(self, basis_set, potential):
        ibz = self.ibz
        kpt_comm = self.communicators['k']
        band_comm = self.communicators['b']
        domain_comm = self.communicators['d']
        spin_degeneracy = 2 if self.ncomponents == 1 else 1
        rank_k = ibz.ranks(kpt_comm)
        wfs_qs = []
        for kpt_c, weight, rank in zip(ibz.kpt_kc, ibz.weight_k, rank_k):
            if rank != kpt_comm.rank:
                continue
            wfs_s = []
            for s in range(self.ncomponents):
                wfs_s.append(
                    DummyWaveFunctions(
                        domain_comm,
                        band_comm,
                        self.nbands,
                        s, self.setups,
                        self.fracpos_ac, weight,
                        spin_degeneracy=spin_degeneracy))
            wfs_qs.append(wfs_s)

        ibzwfs = IBZWaveFunctions(ibz, rank_k, kpt_comm, wfs_qs,
                                  self.nelectrons,
                                  spin_degeneracy)
        return ibzwfs

    def create_scf_loop(self, pot_calc):
        occ_calc = self.create_occupation_number_calculator()
        return DummySCFLoop(occ_calc, pot_calc)


def pairpot(atoms):
    """Simple pair-potential for testing.

    >>> from ase import Atoms
    >>> r = covalent_radii[1]
    >>> atoms = Atoms('H2', [(0, 0, 0), (0, 0, 2 * r)])
    >>> e, f, s = pairpot(atoms)
    >>> print(f'{e:.6f} eV')
    -9.677419 eV
    >>> f
    array([[0., 0., 0.],
           [0., 0., 0.]])

    """
    radii = {}
    symbol_a = atoms.symbols
    for symbol in symbol_a:
        radii[symbol] = covalent_radii[atomic_numbers[symbol]]
    r0 = {}
    for s1, r1 in radii.items():
        for s2, r2 in radii.items():
            r0[(s1, s2)] = r1 + r2
    rcutmax = 2 * max(r0.values())
    energy = 0.0
    force_av = np.zeros((len(atoms), 3))
    stress_vv = np.zeros((3, 3))
    for i, j, d, D_v in zip(*neighbor_list('ijdD', atoms, rcutmax)):
        d0 = r0[(symbol_a[i], symbol_a[j])]
        e0 = 6.0 / d0
        x = d0 / d
        if x > 0.5:
            energy += 0.5 * e0 * (-5 + x * (24 + x * (-36 + 16 * x)))
            f = -0.5 * e0 * (24 + x * (-72 + 48 * x)) * d0 / d**2
            F_v = D_v * f / d
            force_av[i] += F_v
            force_av[j] -= F_v
            # print(i, j, d, D_v, F_v)
            stress_vv += np.outer(F_v, D_v)

    return energy, force_av, stress_vv


def poly():
    """Polynomium used for pair potential."""
    import matplotlib.pyplot as plt
    c = np.linalg.solve([[1, 0.5, 0.25, 0.125],
                         [1, 1, 1, 1],
                         [0, 1, 1, 0.75],
                         [0, 1, 2, 3]],
                        [0, -1, 0, 0])
    print(c)
    d = np.linspace(0.5, 2, 101)
    plt.plot(d, c[0] + c[1] / d + c[2] / d**2 + c[3] / d**3)
    plt.show()


if __name__ == '__main__':
    poly()
