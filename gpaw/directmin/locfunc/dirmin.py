from gpaw.directmin.fd.inner_loop import InnerLoop
from ase.parallel import parprint


class DirectMinLocalize:

    def __init__(self, obj_f, wfs, maxiter=333, g_tol=1.0e-4):

        self.obj_f = obj_f
        self.iloop = InnerLoop(
            obj_f, wfs, maxiter=maxiter,
            g_tol=g_tol)

    def run(self, wfs, dens, log=None, max_iter=None, g_tol=None):

        if g_tol is not None:
            self.iloop.g_tol = g_tol
        if max_iter is not None:
            self.iloop.max_iter = max_iter
        # if log is None:
        #     log = parprint

        wfs.timer.start('Inner loop')

        psi_copy = {}
        n_kps = wfs.kd.nks // wfs.kd.nspins
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            psi_copy[k] = kpt.psit_nG[:].copy()

        counter = self.iloop.run(0.0, psi_copy, wfs, dens, log, 0)
        del psi_copy

        wfs.timer.stop('Inner loop')

        return counter
