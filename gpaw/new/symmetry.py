import numpy as np

from gpaw.mpi import MPIComm
from gpaw.new.brillouin import IBZ, BZPoints
from gpaw.rotation import rotation


class Symmetries:
    def __init__(self, symmetry):
        self.symmetry = symmetry
        self.rotation_scc = symmetry.op_scc
        self.translation_sc = symmetry.ft_sc
        self.a_sa = symmetry.a_sa
        cell_cv = symmetry.cell_cv
        self.rotation_svv = np.einsum('vc, scd, dw -> svw',
                                      np.linalg.inv(cell_cv),
                                      self.rotation_scc,
                                      cell_cv)
        self.rotation_lsmm = [
            np.array([rotation(l, r_vv) for r_vv in self.rotation_svv])
            for l in range(4)]
        self._rotations = {}

    def __len__(self):
        return len(self.rotation_scc)

    def __str__(self):
        return str(self.symmetry)

    def reduce(self,
               bz: BZPoints,
               comm: MPIComm = None) -> IBZ:
        (_, weight_k, sym_k, time_reversal_k, bz2ibz_K, ibz2bz_k,
         bz2bz_Ks) = self.symmetry.reduce(bz.kpt_Kc, comm)

        return IBZ(self, bz, ibz2bz_k, bz2ibz_K, weight_k)

    def check_positions(self, fracpos_ac):
        self.symmetry.check(fracpos_ac)

    def symmetrize_forces(self, F_av):
        return self.symmetry.symmetrize_forces(F_av)

    def rotations(self, l_j):
        ells = tuple(l_j)
        rotation_sii = self._rotations.get(ells)
        if rotation_sii is None:
            ni = sum(2 * l + 1 for l in l_j)
            rotation_sii = np.zeros((len(self), ni, ni))
            i1 = 0
            for l in l_j:
                i2 = i1 + 2 * l + 1
                rotation_sii[:, i1:i2, i1:i2] = self.rotation_lsmm[l]
                i1 = i2
            self._rotations[ells] = rotation_sii
        return rotation_sii
