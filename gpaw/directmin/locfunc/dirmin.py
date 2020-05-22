from gpaw.directmin.fd.inner_loop import InnerLoop
from ase.parallel import parprint


class DirectMinLocalize:

    def __init__(self, obj_f, wfs, maxiter=333, g_tol=1.0e-3):

        self.obj_f = obj_f
        self.iloop = InnerLoop(
            obj_f, wfs, maxiter=maxiter,
            tol=g_tol)

    def run(self, wfs, dens, log=None, max_iter=None, g_tol=None):

        if g_tol is not None:
            self.iloop.tol = g_tol
        if max_iter is not None:
            self.iloop.max_iter = max_iter
        # if log is None:
        #     log = parprint

        wfs.timer.start('Inner loop')

        counter = self.iloop.run(0.0, wfs, dens, log, 0)

        wfs.timer.stop('Inner loop')

        return counter
