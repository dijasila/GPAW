from gpaw.core.domain import Domain


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
        return psit_nsX

    def atom_centered_functions(self,
                                pt_aj,
                                positions,
                                *,
                                atomdist=None):
        pt_aiX = self.desc.atom_centered_functions(pt_aj, positions,
                                                   atomdist=atomdist)
        return pt_aiX
