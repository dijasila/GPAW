import numpy as np
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.new import cached_property
from gpaw.new.calculation import DFTCalculation
from gpaw.utilities import pack
from gpaw.band_descriptor import BandDescriptor
from ase import Atoms
from gpaw.projections import Projections
from gpaw.pw.descriptor import PWDescriptor


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
        self.nvalence = int(round(ibzwfs.nelectrons))
        assert self.nvalence == ibzwfs.nelectrons
        self.world = calculation.scf_loop.world
        self.fermi_level, = ibzwfs.fermi_levels
        self.nspins = ibzwfs.nspins
        self.dtype = ibzwfs.dtype
        self.pd = PWDescriptor(ibzwfs.wfs_qs[0][0].psit_nX.desc.ecut,
                               self.gd, self.dtype, self.kd)

    def _get_wave_function_array(self, u, n, realspace):
        return self.kpt_u[u]

    @cached_property
    def kpt_u(self):
        return [kpt
                for kpt_s in self.kpt_qs
                for kpt in kpt_s]

    @cached_property
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
        self.weight = wfs.spin_degeneracy * wfs.weight
        self.f_n = wfs.occ_n * self.weight
        self.psit_nG = wfs.psit_nX.data
        self.P_ani = wfs.P_ani


class FakeDensity:
    def __init__(self, calculation: DFTCalculation):
        self.setups = calculation.setups
        self.state = calculation.state
        self.D_asii = self.state.density.D_asii
        self.atom_partition = calculation._atom_partition
        self.nt_sg = None
        self.interpolate = calculation.pot_calc._interpolate_density
        self.nt_sR = self.state.density.nt_sR
        self.finegd = calculation.pot_calc.fine_grid._gd

    @cached_property
    def D_asp(self):
        return {a: np.array([pack(D_ii) for D_ii in D_sii])
                for a, D_sii in self.D_asii.items()}

    def interpolate_pseudo_density(self):
        self.nt_sg = self.interpolate(self.nt_sR)[0].data


class FakeHamiltonian:
    def __init__(self, calculation: DFTCalculation):
        self.pot_calc = calculation.pot_calc
        self.finegd = self.pot_calc.fine_grid._gd
        self.grid = calculation.state.potential.vt_sR.desc

    def restrict_and_collect(self, vxct_sg):
        fine_grid = self.pot_calc.fine_grid
        vxct_sr = fine_grid.empty(len(vxct_sg))
        vxct_sr.data[:] = vxct_sg
        vxct_sR = self.grid.zeros(vxct_sr.dims)
        self.pot_calc._restrict(vxct_sr, vxct_sR)
        return vxct_sR.data

    @property
    def xc(self):
        return self.pot_calc.xc.xc
