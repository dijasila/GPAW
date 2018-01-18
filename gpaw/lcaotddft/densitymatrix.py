import numpy as np


class DensityMatrix(object):

    def __init__(self, paw):
        self.wfs = paw.wfs
        self.using_blacs = self.wfs.ksl.using_blacs
        self.tag = None

    def zeros(self, dtype):
        ksl = self.wfs.ksl
        if self.using_blacs:
            return ksl.mmdescriptor.zeros(dtype=dtype)
        else:
            return np.zeros((ksl.mynao, ksl.nao), dtype=dtype)

    def _calculate_density_matrix(self, wfs, kpt):
        if self.using_blacs:
            ksl = wfs.ksl
            rho_MM = ksl.calculate_blocked_density_matrix(kpt.f_n, kpt.C_nM)
        else:
            rho_MM = wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM)
            wfs.bd.comm.sum(rho_MM, root=0)
            # TODO: should the sum over bands be moved to
            # OrbitalLayouts.calculate_density_matrix()
        return rho_MM

    def get_density_matrix(self, tag=None):
        if tag is None or self.tag != tag:
            self.rho_uMM = []
            for u, kpt in enumerate(self.wfs.kpt_u):
                rho_MM = self._calculate_density_matrix(self.wfs, kpt)
                self.rho_uMM.append(rho_MM)
            self.tag = tag
        return self.rho_uMM
