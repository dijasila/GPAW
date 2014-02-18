from gpaw.fd_operators import Gradient
from gpaw.solvation.gridmem import NeedsGD
import numpy as np


class Dielectric(NeedsGD):
    def __init__(self, epsinf):
        """Initialize paramters.

        Keyword arguments:
        epsinf -- dielectric constant at infinite distance from the cavity
        """
        NeedsGD.__init__(self)
        self.epsinf = float(epsinf)
        self.eps_gradeps = None  # eps_g, dxeps_g, dyeps_g, dzeps_g
        self.del_eps_del_g_g = None

    def estimate_memory(self, mem):
        nbytes = self.gd.bytecount()
        mem.subnode('Permittivity', nbytes)
        mem.subnode('Permittivity Gradient', 3 * nbytes)
        mem.subnode('Permittivity Derivative', nbytes)

    def allocate(self):
        NeedsGD.allocate(self)
        self.eps_gradeps = []
        eps_g = self.gd.empty()
        eps_g.fill(1.0)
        self.eps_gradeps.append(eps_g)
        self.eps_gradeps.extend([gd.zeros() for gd in (self.gd, ) * 3])
        self.del_eps_del_g_g = self.gd.empty()

    def update(self, cavity):
        raise NotImplementedError

    def print_parameters(self, text):
        """Print parameters using text function."""
        text('epsilon_inf: %s' % (self.epsinf, ))


class FDGradientDielectric(Dielectric):
    def __init__(self, epsinf, nn=3):
        Dielectric.__init__(self, epsinf)
        self.nn = nn
        self.eps_hack_g = None
        self.gradient = None

    def estimate_memory(self, mem):
        Dielectric.estimate_memory(self, mem)
        mem.subnode('Boundary Correction', self.gd.bytecount())

    def allocate(self):
        Dielectric.allocate(self)
        self.eps_hack_g = self.gd.empty()
        self.gradient = [
            Gradient(self.gd, i, 1.0, self.nn) for i in (0, 1, 2)
            ]

    def update_gradient(self):
        # zero on boundary, since bmgs support only zero or periodic BC
        np.subtract(self.eps_gradeps[0], self.epsinf, out=self.eps_hack_g)
        for i in (0, 1, 2):
            self.gradient[i].apply(self.eps_hack_g, self.eps_gradeps[i + 1])


class LinearDielectric(FDGradientDielectric):
    def allocate(self):
        FDGradientDielectric.allocate(self)
        self.del_eps_del_g_g = self.epsinf - 1.  # frees array

    def update(self, cavity):
        np.multiply(cavity.g_g, self.epsinf - 1., self.eps_gradeps[0])
        self.eps_gradeps[0] += 1.
        self.update_gradient()


class CMDielectric(FDGradientDielectric):
    def update(self, cavity):
        ei = self.epsinf
        t = 1. - cavity.g_g
        self.eps_gradeps[0][:] = (3. * (ei + 2.)) / ((ei - 1.) * t + 3.) - 2.
        self.del_eps_del_g_g[:] = (3. * (ei - 1.) * (ei + 2.)) / ((ei - 1.) * t + 3.) ** 2
        self.update_gradient()
