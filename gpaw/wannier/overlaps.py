from typing import Tuple, Dict, Any, Sequence, List

import numpy as np
from ase import Atoms

from gpaw import GPAW
from gpaw.projections import Projections
from gpaw.utilities.partition import AtomPartition
from gpaw.setup import Setup
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.utilities.ibz2bz import construct_symmetry_operators
from .functions import WannierFunctions

Array1D = Any
Array2D = Any
Array4D = Any


class WannierOverlaps:
    def __init__(self,
                 atoms: Atoms,
                 monkhorst_pack_size: Sequence[int],
                 fermi_level: float,
                 directions: Dict[Tuple[int, int, int], int],
                 overlaps: Array4D):

        self.atoms = atoms
        self.monkhorst_pack_size = tuple(monkhorst_pack_size)
        self.fermi_level = fermi_level
        self.directions = directions

        nkpts, ndirs, self.nbands, nbands = overlaps.shape
        assert nbands == self.nbands
        assert nkpts == np.prod(monkhorst_pack_size)
        assert ndirs == len(directions)

        self._overlaps = overlaps

    def overlap(self,
                bz_index: int,
                direction: Tuple[int, int, int]) -> Array2D:
        return self._overlaps[bz_index, self.directions[direction]]

    def localize(self,
                 maxiter: int = 100,
                 tolerance: float = 1e-5,
                 verbose: bool = not False) -> WannierFunctions:
        from .edmiston_ruedenberg import localize
        return localize(self, maxiter, tolerance, verbose)

    def write_wannier90_input_files(self, prefix, **kwargs):
        import gpaw.wannier.w90 as w90
        w90.write_win(prefix, self)
        w90.write_mmn(prefix, self)


def calculate_overlaps(calc: GPAW,
                       n1: int = 0,
                       n2: int = 0,
                       soc: bool = False,
                       spin: int = 0) -> WannierOverlaps:
    if n2 <= 0:
        n2 += calc.get_number_of_bands()

    bzwfs = BZRealSpaceWaveFunctions.from_calculation(calc, n1, n2, spin)

    kd = bzwfs.kd
    gd = bzwfs.gd
    size = kd.N_c

    icell = calc.atoms.cell.reciprocal()
    directions = {direction: i
                  for i, direction
                  in enumerate(find_directions(icell, size))}
    Z_kdnn = np.empty((kd.nbzkpts, len(directions), n2 - n1, n2 - n1), complex)

    spos_ac = calc.spos_ac
    setups = calc.wfs.setups

    for bz_index1 in range(kd.nbzkpts):
        wf1 = bzwfs[bz_index1]
        i1_c = np.unravel_index(bz_index1, size)
        for direction, d in directions.items():
            i2_c = np.array(i1_c) + direction
            bz_index2 = np.ravel_multi_index(i2_c, size, 'wrap')
            wf2 = bzwfs[bz_index2]
            phase_c = (i2_c % size - i2_c) // size
            u2_nR = wf2.u_nR
            if phase_c.any():
                u2_nR = u2_nR * gd.plane_wave(phase_c)
            Z_kdnn[bz_index1, d] = gd.integrate(wf1.u_nR, u2_nR,
                                                global_integral=False)
            for a, P1_ni in wf1.projections.items():
                dO_ii = setups[a].dO_ii
                P2_ni = wf2.projections[a]
                Z_nn = P1_ni.conj().dot(dO_ii).dot(P2_ni.T).astype(complex)
                if phase_c.any():
                    Z_nn *= np.exp(2j * np.pi * phase_c.dot(spos_ac[a]))
                Z_kdnn[bz_index1, d] += Z_nn

    gd.comm.sum(Z_kdnn)

    overlaps = WannierOverlaps(calc.atoms,
                               kd.N_c,
                               calc.get_fermi_level(),
                               directions,
                               Z_kdnn)
    return overlaps


def find_directions(icell: Array2D,
                    mpsize: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Find nearest neighbors k-points.

    icell:
        Reciprocal cell.
    mpsize:
        Size of Monkhorst-Pack grid.

    If dk is a vector pointing at a neighbor k-points then we don't
    also include -dk in the list.  Examples: for simple cubic there
    will be 3 neighbors and for FCC there will be 6.

    For a hexagonal cell you get three directions in plane and one
    out of plane:

    >>> hex = np.array([[1, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 1]])
    >>> dirs = find_directions(hex, (4, 4, 4))
    >>> sorted(dirs)
    [(0, 0, 1), (0, 1, 0), (1, -1, 0), (1, 0, 0)]
    """

    from scipy.spatial import Voronoi

    d_ic = np.indices((3, 3, 3)).reshape((3, -1)).T - 1
    d_iv = d_ic.dot((icell.T / mpsize).T)
    voro = Voronoi(d_iv)
    directions: List[Tuple[int, int, int]] = []
    for i1, i2 in voro.ridge_points:
        if i1 == 13 and i2 > 13:
            directions.append(tuple(d_ic[i2]))  # type: ignore
        elif i2 == 13 and i1 > 13:
            directions.append(tuple(d_ic[i1]))  # type: ignore
    return directions


class WaveFunction:
    def __init__(self,
                 u_nR,
                 projections: Projections):
        self.u_nR = u_nR
        self.projections = projections

    def transform(self,
                  kd: KPointDescriptor,
                  gd,
                  setups: List[Setup],
                  spos_ac: Array2D,
                  bz_index: int) -> 'WaveFunction':
        """Transforms PAW projections from IBZ to BZ k-point."""
        a_a, U_aii, time_rev = construct_symmetry_operators(
            kd, setups, spos_ac, bz_index)

        projections = self.projections.new()

        if projections.atom_partition.comm.rank == 0:
            a = 0
            for b, U_ii in zip(a_a, U_aii):
                P_msi = self.projections[b].dot(U_ii)
                if time_rev:
                    P_msi = P_msi.conj()
                projections[a][:] = P_msi
                a += 1
        else:
            assert len(projections.indices) == 0

        u_nR = 1 / 0
        return WaveFunction(u_nR, projections)

    def redistribute_atoms(self,
                           gd,
                           atom_partition: AtomPartition
                           ) -> 'WaveFunction':
        projections = self.projections.redist(atom_partition)
        u_nR = gd.distribute(self.u_nR)
        return WaveFunction(u_nR, projections)


class BZRealSpaceWaveFunctions:
    """Container for wave-functions and PAW projections (all of BZ)."""
    def __init__(self,
                 kd: KPointDescriptor,
                 gd,
                 wfs: Dict[int, WaveFunction]):
        self.kd = kd
        self.gd = gd
        self.wfs = wfs

    def __getitem__(self, bz_index):
        return self.wfs[bz_index]

    @classmethod
    def from_calculation(cls,
                         calc: GPAW,
                         n1: int = 0,
                         n2: int = 0,
                         spin=0) -> 'BZRealSpaceWaveFunctions':
        wfs = calc.wfs
        kd = wfs.kd

        if wfs.mode == 'lcao' and not wfs.positions_set:
            calc.initialize_positions()

        gd = wfs.gd.new_descriptor(comm=calc.world)

        nproj_a = wfs.kpt_qs[0][0].projections.nproj_a
        # All atoms on rank-0:
        rank_a = np.zeros_like(nproj_a)
        atom_partition = AtomPartition(gd.comm, rank_a)

        rank_a = np.arange(len(rank_a)) % gd.comm.size
        atom_partition2 = AtomPartition(gd.comm, rank_a)

        spos_ac = calc.spos_ac
        setups = calc.wfs.setups

        u_nR = gd.empty((n2 - n1), complex, global_array=True)

        bzwfs = {}
        for ibz_index in range(kd.nibzkpts):
            for n in range(n1, n2):
                u_nR[n - n1] = wfs.get_wave_function_array(n=n,
                                                           k=ibz_index,
                                                           s=spin,
                                                           periodic=True)
            P_nI = wfs.collect_projections(ibz_index, spin)
            if P_nI is not None:
                P_nI = P_nI[n1:n2]
            projections = Projections(
                nbands=n2 - n1,
                nproj_a=nproj_a,
                atom_partition=atom_partition,
                data=P_nI)

            wf = WaveFunction(u_nR, projections)

            for bz_index, ibz_index2 in enumerate(kd.bz2ibz_k):
                if ibz_index2 != ibz_index:
                    continue
                if kd.ibz2bz_k[ibz_index] == bz_index:
                    wf1 = wf
                else:
                    wf1 = wf.transform(kd, gd, setups, spos_ac, bz_index)

                bzwfs[bz_index] = wf1.redistribute_atoms(gd, atom_partition2)

        return BZRealSpaceWaveFunctions(kd, gd, bzwfs)
