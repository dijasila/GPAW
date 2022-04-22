from gpaw.core.domain import Domain
from gpaw.core.arrays import DistributedArrays


class SpinorWaveFunctionDescriptor(Domain):
    def __init__(self, desc: Domain):
        self.desc = desc
        Domain.__init__(self, desc.cell_cv, desc.pbc_c, desc.kpt_c, desc.comm,
                        complex)
        self.myshape = (2,) + desc.myshape

    def new(self, *, kpt):
        return SpinorWaveFunctionDescriptor(self.desc.new(kpt=kpt))

    def empty(self, nbands, band_comm):
        psit_nsX = self.desc.empty((nbands, 2), band_comm)
        return psit_nsX#SpinorWaveFunctions(psit_nsX, self)

    def atom_centered_functions(self,
                                pt_aj,
                                positions,
                                *,
                                atomdist=None):
        pt_aiX = self.desc.atom_centered_functions(pt_aj, positions,
                                                   atomdist=atomdist)
        return pt_aiX#SpinorProjectors(pt_aiX)


class SpinorProjectors:
    def __init__(self, pt_aiX):
        self.pt_aiX = pt_aiX

    def empty(self, dims, comm, transposed):
        assert transposed
        return self.pt_aiX.empty(dims + (2,), comm, transposed)

    def integrate(self, psit_nX, P_ains):
        return self.pt_aiX.integrate(psit_nX.psit_nsX, P_ains)


class SpinorWaveFunctions(DistributedArrays[SpinorWaveFunctionDescriptor]):
    def __init__(self, psit_nsX, desc):
        self.desc = desc
        self.psit_nsX = psit_nsX
        DistributedArrays. __init__(self, psit_nsX.dims[0], desc.myshape,
                                    psit_nsX.comm, desc.comm,
                                    psit_nsX.data, desc.desc.dv, complex,
                                    transposed=False)
        # self._matrix: Matrix | None

    def new(self, data):
        return SpinorWaveFunctions(self.psit_nsX.new(data=data), self.desc)
