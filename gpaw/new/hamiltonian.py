from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.uniform_grid import UGArray as UGArray


class Hamiltonian:
    def apply(self,
              vt_sR: UGArray,
              dedtaut_sR: UGArray | None,
              psit_nG: XArray,
              out: XArray,
              spin: int) -> XArray:
        raise NotImplementedError

    def create_preconditioner(self, blocksize):
        raise NotImplementedError
