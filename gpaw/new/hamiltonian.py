from __future__ import annotations
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core import UGArray


class Hamiltonian:
    def apply(self,
              vt_sR: UGArray,
              dedtaut_sR: UGArray | None,
              psit_nG: XArray,
              out: XArray,
              spin: int) -> XArray:
        self.apply_local_potential(vt_sR[spin], psit_nG, out)
        if dedtaut_sR is not None:
            self.apply_mgga(dedtaut_sR[spin], psit_nG, out)
        return out

    def apply_local_potential(self,
                              vt_R: UGArray,
                              psit_nG: XArray,
                              out: XArray
                              ) -> None:
        raise NotImplementedError

    def apply_mgga(self,
                   dedtaut_R: UGArray,
                   psit_nG: XArray,
                   vt_nG: XArray) -> None:
        raise NotImplementedError

    def create_preconditioner(self, blocksize):
        raise NotImplementedError
