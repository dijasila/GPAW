import numpy as np
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.new import cached_property
from gpaw.new.calculation import DFTCalculation
from gpaw.utilities import pack
from gpaw.band_descriptor import BandDescriptor
from ase import Atoms
from gpaw.utilities.partition import AtomPartition


class FakeWFS:
    def __init__(self, calculation: DFTCalculation, atoms: Atoms):
        self.setups = calculation.setups
        self.state = calculation.state
        ibzwfs = self.state.ibzwfs
        self.kd = KPointDescriptor(ibzwfs.ibz.bz.kpt_Kc,
                                   ibzwfs.nspins)
        self.kd.set_symmetry(atoms,
                             ibzwfs.ibz.symmetries.symmetry)
        self.kd.set_communicator(ibzwfs.kpt_comm)
        self.bd = BandDescriptor(ibzwfs.nbands, ibzwfs.band_comm)
        self.gd = self.state.density.nt_sR.desc._gd

    @property
    def kpt_u(self):
        return [KPT(wfs)
                for wfs_s in self.state.ibzwfs.wfs_qs
                for wfs in wfs_s]

    @property
    def kpt_qs(self):
        return [[KPT(wfs)
                 for wfs in wfs_s]
                for wfs_s in self.state.ibzwfs.wfs_qs]


class KPT:
    def __init__(self, wfs):
        self.wfs = wfs
        self.projections = wfs.P_ani


class FakeDensity:
    def __init__(self, calculation: DFTCalculation):
        self.setups = calculation.setups
        self.state = calculation.state
        self.D_asii = self.state.density.D_asii
        atomdist = self.D_asii.layout.atomdist
        self.atom_partition = AtomPartition(atomdist.comm,
                                            atomdist.rank_a)

    @cached_property
    def D_asp(self):
        return {a: np.array([pack(D_ii) for D_ii in D_sii])
                for a, D_sii in self.D_asii.items()}


class FakeHamiltonian:
    def __init__(self, calculation: DFTCalculation):
        self.pot_calc = calculation.pot_calc

    @property
    def xc(self):
        return self.pot_calc.xc.xc
