import os

import numpy as np

from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.lcaotddft.utilities import collect_uMM
from gpaw.tddft.units import au_to_as


class TimeDensityMatrix(TDDFTObserver):

    def __init__(self, paw, dmat, ksd, interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.dmat = dmat
        if paw.wfs.world.rank == 0:
            self.ksd = ksd
            self.dpath = 'rho'
            if not os.path.isdir(self.dpath):
                os.mkdir(self.dpath)

    def _update(self, paw):
        time = paw.time
        rho_uMM = self.dmat.get_density_matrix((paw.niter, paw.action))
        rho_uMM = [collect_uMM(paw.wfs, rho_uMM, 0, 0)]
        if paw.wfs.world.rank == 0:
            fpath = os.path.join(self.dpath, 't%09.1f.npy' % (time * au_to_as))
            rho_p = self.ksd.transform(rho_uMM)[0]
            np.save(fpath, rho_p)
