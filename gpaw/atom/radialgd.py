from math import pi

import numpy as np

from gpaw.spline import Spline
from gpaw.utilities import hartree, divrl


class RadialGridDescriptor:
    def __init__(self, r_g, dr_g, default_spline_points=25):
        """Grid descriptor for radial grid."""
        self.r_g = r_g
        self.dr_g = dr_g
        self.N = len(r_g)
        self.dv_g = 4 * pi * r_g**2 * dr_g
        self.default_spline_points = default_spline_points
        
    def zeros(self, x=()):
        a_xg = self.empty(x)
        a_xg[:] = 0
        return a_xg

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

    def purepythonpoisson(self, n_g, l=0):
        r_g = self.r_g
        dr_g = self.dr_g
        a_g = -4 * pi * n_g * r_g * dr_g
        a_g[1:] /= r_g[1:]**l
        A_g = np.add.accumulate(a_g)
        vr_g = self.zeros()
        vr_g[1:] = A_g[:-1] + 0.5 * a_g[1:]
        vr_g -= A_g[-1]
        vr_g *= r_g**(1 + l)
        a_g *= r_g**(2 * l + 1)
        A_g = np.add.accumulate(a_g)
        vr_g[1:] -= A_g[:-1] + 0.5 * a_g[1:]
        vr_g[1:] /= r_g[1:]**l
        return vr_g
    
    def poisson(self, n_g, l=0):  # Old C version
        vr_g = self.zeros()
        nrdr_g = n_g * self.r_g * self.dr_g
        beta = self.a / self.b
        ng = int(round(1.0 / self.b))
        assert abs(ng - 1 / self.b) < 1e-5
        hartree(l, nrdr_g, beta, ng, vr_g)
        vrp_g = self.purepythonpoisson(n_g,l)
        assert abs(vr_g-vrp_g).max() < 1e-12
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
        from scipy.optimize import fsolve
        fsolve(f, x0)
        return b_g, c_x
        
    def plot(self, a_g, n=0, rc=4.0, show=False):
        import matplotlib.pyplot as plt
        r_g = self.r_g[:len(a_g)]
        plt.plot(r_g, a_g * r_g**n)
        plt.axis(xmax=rc)
        if show:
            plt.show()

    def floor(self, r):
        return np.floor(self.r2g(r)).astype(int)
    
    def round(self, r):
        return np.around(self.r2g(r)).astype(int)
    
    def ceil(self, r):
        return np.ceil(self.r2g(r)).astype(int)

    def spline(self, a_g, rcut, l=0, points=None):
        if points is None:
            points = self.default_spline_points

        b_g = a_g.copy()
        N = len(b_g)
        if l > 0:
            b_g = divrl(b_g, l, self.r_g[:N])
            #b_g[1:] /= self.r_g[1:]**l
            #b_g[0] = b_g[1]
            
        r_i = np.linspace(0, rcut, points + 1)
        #g_i = np.clip(self.ceil(r_i), 1, self.N - 2)
        #g_i = np.clip(self.round(r_i), 1, self.N - 2)
        g_i = np.clip((self.r2g(r_i)+0.5).astype(int), 1, N - 2)
        if 0:#a_g[0] < 0:
            print a_g[[0,1,2,-10,-2,-1]]
            print rcut,l,points, len(a_g)
            print g_i;dcvg
        r1_i = self.r_g[g_i - 1]
        r2_i = self.r_g[g_i]
        r3_i = self.r_g[g_i + 1]
        x1_i = (r_i - r2_i) * (r_i - r3_i) / (r1_i - r2_i) / (r1_i - r3_i)
        x2_i = (r_i - r1_i) * (r_i - r3_i) / (r2_i - r1_i) / (r2_i - r3_i)
        x3_i = (r_i - r1_i) * (r_i - r2_i) / (r3_i - r1_i) / (r3_i - r2_i)
        b1_i = b_g[g_i - 1]
        b2_i = b_g[g_i]
        b3_i = b_g[g_i + 1]
        b_i = b1_i * x1_i + b2_i * x2_i + b3_i * x3_i
        return Spline(l, rcut, b_i)


class EquidistantRadialGridDescriptor(RadialGridDescriptor):
    def __init__(self, h, N=1000, h0=0.0):
        """Equidistant radial grid descriptor.

        The radial grid is r(g) = h0 + g*h,  g = 0, 1, ..., N - 1."""

        RadialGridDescriptor.__init__(self, h * np.arange(N) + h0, h)

    def r2g(self, r):
        return (r - self.r_g[0]) / (self.r_g[1] - self.r_g[0])

    def spline(self, a_g, l=0):
        b_g = a_g.copy()
        if l > 0:
            b_g = divrl(b_g, l, self.r_g[:len(a_g)])
            #b_g[1:] /= self.r_g[1:]**l
            #b_g[0] = b_g[1]
        return Spline(l, self.r_g[len(a_g) - 1], b_g)


class AERadialGridDescriptor(RadialGridDescriptor):
    def __init__(self, a, b, N=1000, default_spline_points=25):
        """Radial grid descriptor for all-electron calculation.

        The radial grid is::

                     a g
            r(g) = -------,  g = 0, 1, ..., N - 1
                   1 - b g
        """

        self.a = a
        self.b = b
        g = np.arange(N)
        r_g = self.a * g / (1 - self.b * g)
        dr_g = (self.b * r_g + self.a)**2 / self.a
        RadialGridDescriptor.__init__(self, r_g, dr_g, default_spline_points)
                                      

    def r2g(self, r):
        # return r / (r * self.b + self.a)
        # Hack to preserve backwards compatibility:
        ng = 1.0 / self.b
        beta = self.a / self.b
        return ng * r / (beta + r)

    def xml(self, id='grid1'):
        if abs(self.N - 1 / self.b) < 1e-5:
            return (('<radial_grid eq="r=a*i/(n-i)" a="%r" n="%d" ' +
                     'istart="0" iend="%d" id="%s"/>') % 
                    (self.a * self.N, self.N, self.N - 1, id))
        return (('<radial_grid eq="r=a*i/(1-b*i)" a="%r" b="%r" n="%d" ' +
                 'istart="0" iend="%d" id="%s"/>') % 
                (self.a, self.b, self.N, self.N - 1, id))

    def d2gdr2(self):
        return -2 * self.a * self.b / (self.b * self.r_g + self.a)**3
