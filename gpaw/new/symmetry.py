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
        (_, weight_k, sym_k, time_reversal_k, bz2ibz_K, ibz2bz_k,
         bz2bz_Ks) = self.symmetry.reduce(bz.kpt_Kc, comm)

        return IBZ(self, bz, ibz2bz_k, bz2ibz_K, weight_k)

    def check_positions(self, fracpos_ac):
        self.symmetry.check(fracpos_ac)
