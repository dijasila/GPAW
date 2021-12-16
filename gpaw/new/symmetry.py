from gpaw.new.brillouin import IBZ, BZPoints
from gpaw.mpi import MPIComm


class Symmetries:
    def __init__(self, symmetry):
        self.symmetry = symmetry

    def __str__(self):
        return str(self.symmetry)

    def reduce(self,
               bz: BZPoints,
               comm: MPIComm = None) -> IBZ:
        (_, weight_i, sym_i, time_reversal_i, bz2ibz_k, ibz2bz_i,
         bz2bz_ks) = self.symmetry.reduce(bz.kpt_kc, comm)

        return IBZ(self, bz, ibz2bz_i, bz2ibz_k, weight_i)

    def check_positions(self, fracpos_ac):
        self.symmetry.check(fracpos_ac)
