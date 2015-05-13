from gpaw.fd_operators import Gradient
from gpaw.solvation.gridmem import NeedsGD
import numpy as np


class Dielectric(NeedsGD):
    """Class representing a spatially varying permittivity.

    Attributes:
    eps_gradeps     -- List [eps_g, dxeps_g, dyeps_g, dzeps_g]
                       with
                       eps_g:   permittivity on the fine grid
                       dieps_g: gradient of eps_g.
    del_eps_del_g_g -- Partial derivative with respect to
                       the cavity. (A point-wise dependence
                       on the cavity is assumed).
    """

    def __init__(self, epsinf):
        """Constructor for the Dielectric class.

        Arguments:
        epsinf -- Static dielectric constant
                  at infinite distance from the solute.
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
        """Calculate eps_gradeps and del_eps_del_g_g from the cavity."""
        raise NotImplementedError

    def print_parameters(self, text):
        """Print parameters using text function."""
        text('epsilon_inf: %s' % (self.epsinf, ))


class FDGradientDielectric(Dielectric):
    """Dielectric with finite difference gradient."""

    def __init__(self, epsinf, nn=3):
        """Constructor for the FDGradientDielectric class.

        Additional arguments not present in the base Dielectric class:
        nn -- Stencil size for finite difference gradient.
        """
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
        """Update the gradient.

        Usage note:
        Call this method at the end of the update method of a subclass.
        """
        # zero on boundary, since bmgs supports only zero or periodic BC
        np.subtract(self.eps_gradeps[0], self.epsinf, out=self.eps_hack_g)
        for i in (0, 1, 2):
            self.gradient[i].apply(self.eps_hack_g, self.eps_gradeps[i + 1])


class LinearDielectric(FDGradientDielectric):
    """Dielectric depending (affine) linearly on the cavity.

    See also
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """

    def allocate(self):
        FDGradientDielectric.allocate(self)
        self.del_eps_del_g_g = self.epsinf - 1.  # frees array

    def update(self, cavity):
        np.multiply(cavity.g_g, self.epsinf - 1., self.eps_gradeps[0])
        self.eps_gradeps[0] += 1.
        self.update_gradient()


class CMDielectric(FDGradientDielectric):
    """Clausius-Mossotti like dielectric.

    Untested, use at own risk!
    """

    def update(self, cavity):
        ei = self.epsinf
        t = 1. - cavity.g_g
        self.eps_gradeps[0][:] = (3. * (ei + 2.)) / ((ei - 1.) * t + 3.) - 2.
        self.del_eps_del_g_g[:] = (
            (3. * (ei - 1.) * (ei + 2.)) / ((ei - 1.) * t + 3.) ** 2
        )
        self.update_gradient()
