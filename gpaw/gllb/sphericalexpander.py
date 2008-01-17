from math import pi, sqrt

from numpy import array, dot, newaxis, zeros, transpose
import numpy as num

from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import YL
# load points and weights for the angular integration
from gpaw.sphere import Y_nL, points, weights

A_Liy = zeros((25, 3, len(points)))

y = 0
for R in points:
    for l in range(5):
        for m in range(2 * l + 1):
            L = l**2 + m
            for c, n in YL[L]:
                for i in range(3):
                    ni = n[i]
                    if ni > 0:
                        a = ni * c * R[i]**(ni - 1)
                        for ii in range(3):
                            if ii != i:
                                a *= R[ii]**n[ii]
                        A_Liy[L, i, y] += a
            A_Liy[L, :, y] -= l * R * Y_nL[y, L]
    y += 1

class SphericalExpander:
    """
       Expands wave functions, pseudo wavefunctions to radial and angular grid.
       Can calculate densities, or more special things.
    """

    def __init__(self, rgd, lmax, jl, w_j,  nc_g):
        # Get the integration weights

        self.nc_g = nc_g
        
        self.Lmax = (lmax + 1)**2
        if lmax == 0:
            self.weights = [1.0]
            self.Y_yL = num.array([[1.0 / num.sqrt(4.0 * num.pi)]])
        else:
            self.weights = weights
            self.Y_yL = Y_nL[:, :self.Lmax].copy()

        # Store j,l and the combined index to table
        jlL = []
        for j, l in jl:
            for m in range(2 * l + 1):
                jlL.append((j, l, l**2 + m))

        # Number of radial gridpoints
        ng = len(nc_g)
        self.ng = ng

        # Number of states (counting m-degeneracy)
        ni = len(jlL)
        # Number of states (without counting m-degeneracy)
        nj = len(jl)
        # Number of state pairs (counting m-degeneracy)
        np = ni * (ni + 1) // 2
        self.np = np

        # Number of state pairs (without counting m-degeneracy)
        nq = nj * (nj + 1) // 2

        # Make Gaunt's coefficient table
        self.B_Lqp = zeros((self.Lmax, nq, np), float)
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL:
            for j2, l2, L2 in jlL[i1:]:
                if j1 < j2:
                    q = j2 + j1 * nj - j1 * (j1 + 1) // 2
                else:
                    q = j1 + j2 * nj - j2 * (j2 + 1) // 2
                self.B_Lqp[:, q, p] = gaunt[L1, L2, :self.Lmax]
                p += 1
            i1 += 1
        self.B_pqL = num.transpose(self.B_Lqp).copy()
        self.dv_g = rgd.dv_g
        self.n_qg = zeros((nq, ng), float)
        q = 0
        for j1, l1 in jl:
            for j2, l2 in jl[j1:]:
                rl1l2 = rgd.r_g**(l1 + l2)
                self.n_qg[q] = rl1l2 * w_j[j1] * w_j[j2]
                q += 1
        self.rgd = rgd

    def get_slice_length(self):
        return self.ng

    def get_iterator(self, D_p, core = True, gradient=True):
        return SphericalIterator(self, D_p, core, gradient)

    def get_rgd(self):
        return self.rgd

class SphericalIterator:
    def  __init__(self, expander, D_p, core, need_gradient):
        self.expander = expander
        self.D_p = D_p
        self.core = core
        self.slice = 0
        self.slices = len(self.expander.weights)
        D_Lq = dot3(self.expander.B_Lqp, D_p)
        self.n_Lg = num.dot(D_Lq, self.expander.n_qg)
        if self.core:
            self.n_Lg[0] += self.expander.nc_g * num.sqrt(4 * num.pi)

        if need_gradient:
            self.dndr_Lg = zeros((self.expander.Lmax, self.expander.ng), float)
       
            for L in range(self.expander.Lmax):
                self.expander.rgd.derivative(self.n_Lg[L], self.dndr_Lg[L])

    def get_density(self, n_g):
        Y_L = self.expander.Y_yL[self.slice]
        n_g[:] = num.dot(Y_L, self.n_Lg)   

    def get_gradient(self, a2_g):
        A_Li = A_Liy[:self.expander.Lmax, :, self.slice]
        Y_L = self.expander.Y_yL[self.slice]
        # Calculate gradients in each direction
        a1x_g = num.dot(A_Li[:, 0], self.n_Lg)
        a1y_g = num.dot(A_Li[:, 1], self.n_Lg)
        a1z_g = num.dot(A_Li[:, 2], self.n_Lg)
        a2_g[:] = a1x_g[:]**2 + a1y_g[:]**2 + a1z_g[:]**2
        a2_g[1:] /= self.expander.rgd.r_g[1:]**2
        a2_g[0] = a2_g[1]
        a1_g = num.dot(Y_L, self.dndr_Lg)
        a2_g[:] += a1_g[:]**2

    def get_weight(self):
        return self.expander.weights[self.slice]
    
    def integrate(self, coeff, v_g, H_p):
        #print "H_p needs to be zeroed at first?"
        Y_L = self.expander.Y_yL[self.slice]
        w = self.get_weight()
        # Integrate the slice with respect to orbitals
        H_p += coeff * w * num.dot(dot3(self.expander.B_pqL, Y_L),
                           num.dot(self.expander.n_qg, v_g * self.expander.rgd.dv_g))

    def has_next(self):
        return not (self.slice == self.slices)

    def next(self):
        self.slice += 1

    def get_rgd(self):
        return self.expander.get_rgd()
