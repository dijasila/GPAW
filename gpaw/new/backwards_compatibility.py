import numpy as np
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.new import cached_property
from gpaw.new.calculation import DFTCalculation
from gpaw.utilities import pack
from gpaw.band_descriptor import BandDescriptor
from ase import Atoms
from gpaw.projections import Projections


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
        self.atom_partition = calculation._atom_partition
        self.setups.set_symmetry(ibzwfs.ibz.symmetries.symmetry)
        self.occupations = calculation.scf_loop.occ_calc.occ
        self.nvalence = ibzwfs.nelectrons

    @property
    def xxxkpt_u(self):
        return [KPT(wfs)
                for wfs_s in self.state.ibzwfs.wfs_qs
                for wfs in wfs_s]

    @property
    def kpt_qs(self):
        return [[KPT(wfs, self.atom_partition)
                 for wfs in wfs_s]
                for wfs_s in self.state.ibzwfs.wfs_qs]


class KPT:
    def __init__(self, wfs, atom_partition):
        self.wfs = wfs
        self.projections = Projections(
            wfs.nbands,
            [I2 - I1 for (a, I1, I2) in wfs.P_ani.layout.myindices],
            atom_partition,
            wfs.P_ani.comm,
            wfs.ncomponents < 4,
            wfs.spin,
            data=wfs.P_ani.data)
        self.eps_n = wfs.eig_n
        self.s = wfs.spin if wfs.ncomponents < 4 else None
        self.k = wfs.k


class FakeDensity:
    def __init__(self, calculation: DFTCalculation):
        self.setups = calculation.setups
        self.state = calculation.state
        self.D_asii = self.state.density.D_asii
        self.atom_partition = calculation._atom_partition

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
