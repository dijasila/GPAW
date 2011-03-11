from math import pi

from scipy.optimize import fsolve
import numpy as np


class RadialGridDescriptor:
    def __init__(self, r_g, dr_g):
        """Grid descriptor for radial grid."""
        self.r_g = r_g
        self.dr_g = dr_g
        self.N = len(r_g)
        self.dv_g = 4 * pi * r_g**2 * dr_g

    def zeros(self, x=()):
        if isinstance(x, int):
            x = (x,)
        return np.zeros(x + (self.N,))

    def empty(self, x=()):
        if isinstance(x, int):
            x = (x,)
        return np.zeros(x + (self.N,))

    def integrate(self, a_xg, n=0):
        assert n >= -2
        return np.dot(a_xg[..., 1:],
                      (self.r_g**(2 + n) * self.dr_g)[1:]) * (4 * pi)

    def derivative(self, n_g, dndr_g):
        """Finite-difference derivative of radial function."""
        dndr_g[0] = n_g[1] - n_g[0]
        dndr_g[1:-1] = 0.5 * (n_g[2:] - n_g[:-2])
        dndr_g[-1] = n_g[-1] - n_g[-2]
        dndr_g /= self.dr_g

    def derivative2(self, a_g, b_g):
        """Finite-difference derivative of radial function.

        For an infinitely dense grid, this method would be identical
        to the `derivative` method."""
        
        c_g = a_g / self.dr_g
        b_g[0] = 0.5 * c_g[1] + c_g[0]
        b_g[1] = 0.5 * c_g[2] - c_g[0]
        b_g[1:-1] = 0.5 * (c_g[2:] - c_g[:-2])
        b_g[-2] = c_g[-1] - 0.5 * c_g[-3]
        b_g[-1] = -c_g[-1] - 0.5 * c_g[-2]

    def poisson(self, n_g):
        a_g = -4 * pi * n_g * self.r_g * self.dr_g
        A_g = np.add.accumulate(a_g)
        vr_g = self.zeros()
        vr_g[1:] = A_g[:-1] + 0.5 * a_g[1:]
        vr_g -= A_g[-1]
        vr_g *= self.r_g
        a_g *= self.r_g
        A_g = np.add.accumulate(a_g)
        vr_g[1:] -= A_g[:-1] + 0.5 * a_g[1:]
        return vr_g

    def pseudize(self, a_g, gc, l=0, points=4):
        """Construct smooth continuation of a_g for g<gc.
        
        Returns (b_g, c_p) such that b_g=a_g for g >= gc::
        
                   P-1      2(P-1-p)+l
            b(r) = Sum c_p r 
                   p=0
        """
        assert isinstance(gc, int) and gc > 10
        
        r_g = self.r_g
        g = [0] + range(gc, gc + points)
        c_p = np.polyfit(r_g[g]**2,
                         a_g[g] * r_g[g]**(2 - l), points)[:-1]
        b_g = a_g.copy()
        b_g[:gc] = np.polyval(c_p, r_g[:gc]**2) * r_g[:gc]**l
        return b_g, c_p

    def pseudize_normalized(self, a_g, gc, l=0, points=3):
        """Construct normalized smooth continuation of a_g for g<gc.
        
        Returns (b_g, c_p) such that b_g=a_g for g >= gc and::
        
            /        2  /        2
            | dr b(r) = | dr a(r)
            /           /
        """
        b_g, c_x = self.pseudize(a_g, gc, l, points + 1)
        gc0 = gc // 2
        x0 = b_g[gc0]
        r_g = self.r_g
        g = [0, gc0] + range(gc, gc + points)
        norm = self.integrate(a_g**2)
        def f(x):
            b_g[gc0] = x
            c_x[:] = np.polyfit(r_g[g]**2,
                                b_g[g] * r_g[g]**(2 - l), points + 1)[:-1]
            b_g[:gc] = np.polyval(c_x, r_g[:gc]**2) * r_g[:gc]**l
            return self.integrate(b_g**2) - norm
        fsolve(f, x0)
        return b_g, c_x
        
    def plot(self, a_g, n=0, rc=4.0, show=False):
        import matplotlib.pyplot as plt
        plt.plot(self.r_g, a_g * self.r_g**n)
        plt.axis(xmax=rc)
        if show:
            plt.show()

    def floor(self, r):
        return int(np.floor(self.r2g(r)))
    
    def round(self, r):
        return int(round(self.r2g(r)))
    
    def ceil(self, r):
        return int(np.ceil(self.r2g(r)))


class EquidistantRGD(RadialGridDescriptor):
    def __init__(self, h, N=1000, h0=0.0):
        """Equidistant radial grid descriptor.

        The radial grid is r(g) = h0 + g*h,  g = 0, 1, ..., N - 1."""

        RadialGridDescriptor.__init__(self, h * np.arange(N) + h0, h)

    def r2g(self, r):
        return (r - self.r_g[0]) / (self.r_g[1] - self.r_g[0])


class RGDNew(RadialGridDescriptor):
    def __init__(self, r1, rN=50.0, N=1000):
        """Radial grid descriptor for all-electron calculation.

        The radial grid is::

                     a g
            r(g) = -------,  g = 0, 1, ..., N - 1
                   1 - b g
        
        so that r(0)=0, r(1)=r1 and r(N)=rN."""

        self.a = (1 - 1.0 / N) / (1.0 / r1 - 1.0 / rN)
        self.b = 1.0 - self.a / r1
        g = np.arange(N)
        r_g = self.a * g / (1 - self.b * g)
        dr_g = (self.b * r_g + self.a)**2 / self.a
        RadialGridDescriptor.__init__(self, r_g, dr_g)
                                      

    def r2g(self, r):
        return 1 / (self.b + self.a / r)


class RGDOld(RadialGridDescriptor):
    def __init__(self, beta, N, default_spline_points=25, _noarrays=False):
        """Radial grid descriptor for old all-electron calculation.

        The radial grid is::

                   beta g
            r(g) = ------,  g = 0, 1, ..., N - 1
                   N - g
        """
        self.beta = beta
        g = np.arange(N, dtype=float)
        r_g = beta * g / (N - g)
        dr_g = beta * N / (N - g)**2
        RadialGridDescriptor.__init__(self, r_g, dr_g)

    def r2g(self, r):
        return r * self.N / (self.beta + r)
