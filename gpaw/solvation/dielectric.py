class BaseDielectric:
    name = 'unnamed'

    def __init__(self, epsinf):
        """Initialize paramters.

        Keyword arguments:
        epsinf -- dielectric constant at infinite distance from the cavity
        """
        self.epsinf = epsinf
        self.gd = None

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def get_eps_deps(self, theta):
        """Return dielectric and derivative from smoothed step function."""
        raise NotImplementedError

    def print_parameters(self, text):
        """Print parameters using text function."""
        text('epsilon_inf: %11.6f' % (self.epsinf, ))


class LinearDielectric(BaseDielectric):
    name = 'Linear Dielectric'

    def get_eps_deps(self, theta):
        eps = self.epsinf + (1. - self.epsinf) * theta
        deps = 1. - self.epsinf
        return (eps, deps)


class CMDielectric(BaseDielectric):
    name = 'Clausius-Mossotti Dielectric'

    def get_eps_deps(self, theta):
        ei = self.epsinf
        eps = (3. * (ei + 2.)) / ((ei - 1.) * theta + 3.) - 2.
        deps = - (3. * (ei - 1.) * (ei + 2.)) / ((ei - 1.) * theta + 3.) ** 2
        return (eps, deps)
