from gpaw.directmin.fdpw.inner_loop import InnerLoop
from ase.parallel import parprint
import numpy as np

class DirectMinLocalize:

    def __init__(self, obj_f, wfs, maxiter=300, g_tol=1.0e-3):

        self.obj_f = obj_f
        self.iloop = InnerLoop(
            obj_f, wfs, maxiter=maxiter,
            g_tol=g_tol)

    def run(self, wfs, dens, log=None, max_iter=None,
            g_tol=None, rewritepsi=True):

        if g_tol is not None:
            self.iloop.tol = g_tol
        if max_iter is not None:
            self.iloop.max_iter = max_iter
        # if log is None:
        #     log = parprint

        wfs.timer.start('Inner loop')

        counter = self.iloop.run(0.0, wfs, dens, log, 0)
        if rewritepsi:
            for kpt in wfs.kpt_u:
                k = self.iloop.n_kps * kpt.s + kpt.q
                # n_occ = self.n_occ[k]
                dim1 = self.iloop.U_k[k].shape[0]
                kpt.psit_nG[:dim1] = np.tensordot(self.iloop.U_k[k].T,
                                                kpt.psit_nG[:dim1],
                                                axes=1)
        wfs.timer.stop('Inner loop')

        return counter
