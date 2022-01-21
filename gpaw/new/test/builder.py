from __future__ import annotations

import numpy as np
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.core.domain import Domain
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.wave_functions import WaveFunctions
from gpaw.new.ibzwfs import IBZWaveFunctions


class NoGrid(Domain):
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


class PSCoreDensities:
    def __init__(self, grid, fracpos_ac):
        self.layout = AtomArraysLayout([1] * len(fracpos_ac),
                                       grid.comm)

    def to_uniform_grid(self, out, scale):
        pass


class DummyBasisSet:
    def add_to_density(self, data, f_asi):
        pass


class TestPotentialCalculator(PotentialCalculator):
    def __init__(self,
                 xc,
                 setups,
                 nct_R):
        super().__init__(xc, None, setups, nct_R)

    def calculate_charges(self, vHt_r):
        return {a: np.zeros(9) for a, setup in enumerate(self.setups)}

    def _calculate(self, density, vHt_r):
        vt_sR = density.nt_sR

        return {'kinetic': 0.0,
                'coulomb': 0.0,
                'zero': 0.0,
                'xc': 0.0,
                'external': 0.0}, vt_sR, vHt_r


class DummyXC:
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


class TestDFTComponentsBuilder(DFTComponentsBuilder):
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
        return TestPotentialCalculator(xc, self.setups, self.nct_R)

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
