import numpy as np
from gpaw.tddft.units import as_to_au, eV_to_au


def create_laser(name, **kwargs):
    if isinstance(name, Laser):
        return name
    elif isinstance(name, dict):
        kwargs.update(name)
        return create_laser(**kwargs)
    elif name == 'GaussianImpulse':
        return GaussianImpulse(**kwargs)
    else:
        raise ValueError('Unknown laser: %s' % name)


class Laser(object):
    def __init__(self):
        pass

    def strength(self, time):
        return 0.0

    def fourier(self, omega):
        return 0.0


class GaussianImpulse(Laser):
    """
    Laser with Gaussian envelope.

    Parameters:
    """

    def __init__(self, strength, time0, frequency, sigma, sincos='sin'):
        self.dict = dict(name='GaussianImpulse',
                         strength=strength,
                         time0=time0,
                         frequency=frequency,
                         sigma=sigma,
                         sincos=sincos)
        self.s0 = strength
        self.t0 = time0 * as_to_au
        self.omega0 = frequency * eV_to_au
        self.sigma = sigma * eV_to_au
        assert sincos in ['sin', 'cos']
        self.sincos = sincos

    def strength(self, t):
        """
        t: au
        """
        s = self.s0 * np.exp(-0.5 * self.sigma**2 * (t - self.t0)**2)
        if self.sincos == 'sin':
            s *= np.sin(self.omega0 * (t - self.t0))
        else:
            s *= np.cos(self.omega0 * (t - self.t0))
        return s

    def derivative(self, t):
        """
        t: au
        """
        dt = t - self.t0
        s = self.s0 * np.exp(-0.5 * self.sigma**2 * dt**2)
        if self.sincos == 'sin':
            s *= (-self.sigma**2 * dt * np.sin(self.omega0 * dt) +
                  self.omega0 * np.cos(self.omega0 * dt))
        else:
            s *= (-self.sigma**2 * dt * np.cos(self.omega0 * dt) +
                  -self.omega0 * np.sin(self.omega0 * dt))
        return s

    def fourier(self, omega):
        """
        omega: au
        """
        s = (self.s0 * np.sqrt(np.pi / 2) / self.sigma *
             np.exp(-0.5 * (omega - self.omega0)**2 / self.sigma**2) *
             np.exp(1.0j * self.t0 * omega))
        if self.sincos == 'sin':
            s *= 1.0j
        return s

    def todict(self):
        return self.dict
