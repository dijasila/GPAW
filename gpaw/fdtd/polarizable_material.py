# Useful routines for the Electrodynamics module, by Arto Sakko (Aalto University)

from ase.parallel import parprint
from ase.units import Hartree, Bohr, _eps0, _c, _aut
from gpaw import PoissonConvergenceError
from gpaw.fd_operators import Gradient
#from gpaw.poisson import PoissonSolver
from poisson_corr import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.io import open as gpaw_io_open
from gpaw.tddft.units import attosec_to_autime, autime_to_attosec
from gpaw.utilities.blas import axpy
from gpaw.utilities.ewald import madelung
#from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.tools import construct_reciprocal
from gpaw.utilities import mlsqr
from math import pi
from numpy.fft import fftn, ifftn, fft2, ifft2
from string import split
import _gpaw
import gpaw.mpi as mpi
import numpy as np
import sys
from gpaw.tddft.units import autime_to_attosec

# in atomic units, 1/(4*pi*e_0) = 1
_eps0_au = 1.0 / (4.0 * np.pi)

# Base class for the classical polarizable material:
#    -holds various functions at each point in the calculation grid:
#        1) the dielectric function (permittivity)
#        2) electric field
#        3) classical polarization charge density
#    -contains routines for calculating them from each other and/or external potential
class PolarizableMaterial:
    def __init__(self, components=[], sign = -1.0):
        self.gd          = None
        self.initialized = False
        self.components  = components
        self.sign        = sign
    
    def addComponent(self, component):
        self.components.append(component)

    def permittivityValue(self, omega=0.0):
        return self.epsInfty + _eps0_au * np.sum(self.beta / (self.barOmega**2.0 - 1J * self.alpha * omega - omega**2.0), axis=0)

    def getStaticPermittivity(self):
        return self.gd.collect(self.permittivityValue(0.0))

    def initialize(self, gd):
        self.initialized = True
        parprint("Initializing Polarizable Material")
        
        try:
            self.Nj = max(component.permittivity.Nj for component in self.components)
        except:
            self.Nj = 0
            
        self.gd = gd
        
        # 3-dimensional scalar array: rho, epsInfty
        self.rhoCl    = self.gd.zeros()
        self.epsInfty = np.ones(self.gd.empty().shape) * _eps0_au
        
        # 3-dimensional vector arrays:
        #        electric field, total polarization density
        dims = [3] + list(self.gd.empty().shape)
        self.eField = np.zeros(dims)
        self.pTotal = np.zeros(dims)
        
        # 4-dimensional vector arrays:
        #        currents, polarizations
        
        dims = [3, self.Nj] + list(self.gd.empty().shape)
        self.currents      = np.zeros(dims)
        self.polarizations = np.zeros(dims)
        
        # 4-dimensional scalar arrays:
        #        oscillator parameters alpha, beta, barOmega, epsInfty
        dims = [self.Nj] + list(self.gd.empty().shape)
        self.alpha    = np.zeros(dims)
        self.beta     = np.zeros(dims)
        self.barOmega = np.ones(dims)
        
        # Set the permittivity for each grid point
        for component in self.components:
            self.applyMask(mask = component.getMask(self.gd),
                           permittivity = component.permittivity)
    
            
    # Here the 3D-arrays are filled with material-specific information
    def applyMask(self, mask, permittivity):
        for j in range(permittivity.Nj):
            self.barOmega   [j] = np.logical_not(mask) * self.barOmega[j] + mask * permittivity.oscillators[j].barOmega
            self.alpha      [j] = np.logical_not(mask) * self.alpha[j]    + mask * permittivity.oscillators[j].alpha
            self.beta       [j] = np.logical_not(mask) * self.beta[j]     + mask * permittivity.oscillators[j].beta
        
        # Add dummy oscillators if needed
        for j in range(permittivity.Nj, self.Nj):
            self.barOmega   [j] = np.logical_not(mask) * self.barOmega[j] + mask * 1.0
            self.alpha      [j] = np.logical_not(mask) * self.alpha[j]    + mask * 0.0
            self.beta       [j] = np.logical_not(mask) * self.beta[j]     + mask * 0.0

        # Print the permittivity information
        parprint("  Permittivity data:")
        parprint("    baromega         alpha          beta")
        parprint("  ----------------------------------------")
        for j in range(permittivity.Nj):
            parprint("%12.6f  %12.6f  %12.6f" % (permittivity.oscillators[j].barOmega,
                                                 permittivity.oscillators[j].alpha,
                                                 permittivity.oscillators[j].beta))
        parprint("  ----------------------------------------")
        parprint("...done initializing Polarizable Material")
        masksum  = self.gd.comm.sum(int(np.sum(mask)))
        masksize = self.gd.comm.sum(int(np.size(mask)))
        parprint("Fill ratio: %f percent" % (100.0 * float(masksum)/float(masksize)))
        
        
    # E(r) = -Grad V(r)
    # NB: Here -V(r) is used (if/when sign=-1), because in GPAW the
    #     electron unit charge is +1 so that the calculated V(r) is
    #     positive around negative charge. In order to get the correct
    #     direction of the electric field, the sign must be changed.
    def solveElectricField(self, phi):
        for v in range(3):
            Gradient(self.gd, v, n=3).apply(-1.0 * self.sign * phi, self.eField[v])

    # n(r) = -Div P(r)
    def solveRho(self):
        self.rhoCl *= 0.0
        dmy         = self.gd.empty()
        for v in range(3):
            Gradient(self.gd, v, n=3).apply(self.pTotal[v], dmy)
            self.rhoCl -= dmy

    # P(r, omega) = [eps(r, omega) - eps0] E(r, omega)
    # P0(r) = [eps_inf(r) - eps0] E0(r) + sum_j P0_j(r) // Gao2012, Eq. 10
    def solvePolarizations(self):
        for v in range(3):
            self.polarizations[v] = _eps0_au * self.beta / (self.barOmega**2.0) * self.eField[v]
        self.pTotal = np.sum(self.polarizations, axis=1) + (self.epsInfty - _eps0_au ) * self.eField

    def propagatePolarizations(self, timestep):
        for v in range(3):
            self.polarizations[v] = self.polarizations[v] + timestep * self.currents[v]
        self.pTotal = np.sum(self.polarizations, axis=1)
    
    def propagateCurrents(self, timestep):
        c1 = (1.0 - 0.5 * self.alpha*timestep)/(1.0 + 0.5 * self.alpha*timestep)
        c2 = - timestep / (1.0 + 0.5 * self.alpha*timestep) * (self.barOmega**2.0)
        c3 = - timestep / (1.0 + 0.5 * self.alpha*timestep) * (-1.0) * _eps0_au * self.beta
        for v in range(3):
            self.currents[v] = c1 * self.currents[v] + c2 * self.polarizations[v] + c3 * self.eField[v]

    def kickElectricField(self, timestep, kick):
        for v in range(3):
            self.eField[v] = self.eField[v] + kick[v] / timestep

    def plotPermittivity(self, omega = 0.0, figtitle=None):
        try:
            from matplotlib.pyplot import figure, rcParams, plot, title, xlim, legend, show
            from plot_functions import plot_projection
            plotData = self.gd.collect(self.permittivityValue(omega))
            if self.gd.comm.rank == 0:
                figure(1, figsize = (19, 10))
                rcParams['font.size'] = 26
                plot(range(0, plotData.shape[0]), np.real(plotData[:, plotData.shape[1]/2, plotData.shape[2]/2])/_eps0_au, label = 'permittivity')
                print np.real(plotData[:, plotData.shape[1]/2, plotData.shape[2]/2])/_eps0_au
                if figtitle is None:
                    title('Permittivity along the x-axis for omega=%.2f eV' % (omega*Hartree))
                else:
                    title(figtitle)
                xlim(0, plotData.shape[0])
                legend()
                show()
        except:
            parprint('Plotting with pyplot failed!')
        self.gd.comm.barrier()



# Box-shaped classical material
class PolarizableBox():
    def __init__(self, vector1, vector2, permittivity):
        # sanity check
        assert(len(vector1)==3)
        assert(len(vector2)==3)

        self.vector1      = np.array(vector1)/Bohr # from Angstroms to atomic units
        self.vector2      = np.array(vector2)/Bohr # from Angstroms to atomic units
        self.permittivity = permittivity

    # Setup grid descriptor and the permittivity values inside the box
    def getMask(self, gd):
        parprint("Initializing Polarizable Box")

        # 3D coordinates at each grid point
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        
        # inside or outside
        return np.logical_and(np.logical_and( # z
                np.logical_and(np.logical_and( # y
                 np.logical_and( #x
                                r_gv[:, :, :, 0] >= self.vector1[0],
                                r_gv[:, :, :, 0] <= self.vector2[0]),
                                 r_gv[:, :, :, 1] >= self.vector1[1]),
                                 r_gv[:, :, :, 1] <= self.vector2[1]),
                                  r_gv[:, :, :, 2] >= self.vector1[2]),
                                  r_gv[:, :, :, 2] <= self.vector2[2])

# Sphere-shaped classical material
class PolarizableSphere():
    def __init__(self, vector1, radius1, permittivity):
        # sanity check
        assert(len(vector1)==3)
        
        self.vector1      = np.array(vector1)/Bohr # from Angstroms to atomic units
        self.radius1      = radius1/Bohr           # from Angstroms to atomic units
        self.permittivity = permittivity

    def getMask(self, gd):
        parprint("Initializing Polarizable Sphere")
        
        # 3D coordinates at each grid point
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        
        # inside or outside
        return  np.array( (r_gv[:, :, :, 0] - self.vector1[0])**2.0 +
                          (r_gv[:, :, :, 1] - self.vector1[1])**2.0 +
                          (r_gv[:, :, :, 2] - self.vector1[2])**2.0 <= self.radius1**2 )

# Sphere-shaped classical material
class PolarizableEllipsoid():
    def __init__(self, vector1, radii, permittivity):
        # sanity check
        assert(len(vector1)==3)
        assert(len(radii)==3)
        
        self.vector1      = np.array(vector1)/Bohr # from Angstroms to atomic units
        self.radii        = np.array(radii)/Bohr   # from Angstroms to atomic units
        self.permittivity = permittivity

    def getMask(self, gd):
        parprint("Initializing Polarizable Ellipsoid")
        
        # 3D coordinates at each grid point
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        
        # inside or outside
        return  np.array( (r_gv[:, :, :, 0] - self.vector1[0])**2.0/self.radii[0]**2.0 +
                          (r_gv[:, :, :, 1] - self.vector1[1])**2.0/self.radii[1]**2.0 +
                          (r_gv[:, :, :, 2] - self.vector1[2])**2.0/self.radii[2]**2.0 <= 1.0)

 # Rod-shaped classical material
class PolarizableRod():
    def __init__(self, corners, radius, permittivity, roundCorners=True):
        # sanity check
        assert(np.array(corners).shape[0]>1)  # at least two points
        assert(np.array(corners).shape[1]==3) # 3D
        
        self.corners      = np.array(corners)/Bohr # from Angstroms to atomic units
        self.radius       = radius/Bohr  # from Angstroms to atomic units
        self.roundCorners = roundCorners
        self.permittivity = permittivity

    def getMask(self, gd):
        parprint("Initializing Polarizable Rod (%i corners)" % len(self.corners))
        
        # 3D coordinates at each grid point
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        ng = r_gv.shape[0:-1]
        ngv = r_gv.shape
        
        a = self.corners[0]

        mask = False * np.ones(ng)

        for p in self.corners[1:]:
            #http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line:
            # d = |(a-p)-((a-p).n)n|   point p, line a+tn  (|n|=1)
            n = (p-a)/np.sqrt((p-a).dot((p-a)))
            v1 = np.array([a[w]-r_gv[:, :, :, w] for w in range(3)]).transpose((1, 2, 3, 0)) # a-p

            v2 = np.sum(np.array([v1[:, :, :, w]*n[w] for w in range(3)]), axis=0)  # (a-p).n

            v3 = np.array([v2*n[w] for w in range(3)]).transpose(1, 2, 3, 0)       # ((a-p).n)n

            d = np.zeros(ng)
            for ind, idx in np.ndenumerate(v3[:, :, :, 0]):
                d[ind] = np.array(v1[ind]-v3[ind]).dot(v1[ind]-v3[ind])

            # angle between (p-a) and (r-a):
            pa = p-a # (3)
            ra = np.array([r_gv[:, :, :, w]-a[w] for w in range(3)]).transpose((1, 2, 3, 0)) # (ng1, ng2, ng3, 3)
            para = np.sum([pa[w]*ra[:, :, :, w] for w in range(3)], axis=0)
            ll2   = pa.dot(pa)*np.sum([ra[:, :, :, w]*ra[:, :, :, w] for w in range(3)], axis=0)
            angle1 = np.arccos(para/(1.0e-9+np.sqrt(ll2)))
            
            # angle between (a-p) and (r-p):
            ap = a-p # (3)
            rp = np.array([r_gv[:, :, :, w]-p[w] for w in range(3)]).transpose((1, 2, 3, 0)) # (ng1, ng2, ng3, 3)
            aprp = np.sum([ap[w]*rp[:, :, :, w] for w in range(3)], axis=0)
            ll2   = ap.dot(ap)*np.sum([rp[:, :, :, w]*rp[:, :, :, w] for w in range(3)], axis=0)
            angle2 = np.arccos(aprp/(1.0e-9+np.sqrt(ll2)))

            # Include in the mask
            thisMask = np.logical_and(np.logical_and(angle1 < 0.5*np.pi, angle2 < 0.5*np.pi),
                                      d <= self.radius**2.0 )

            # Add spheres around current end points 
            if self.roundCorners:
                # |r-a| and |r-p|
                raDist = np.sum([ra[:, :, :, w]*ra[:, :, :, w] for w in range(3)], axis=0)
                rpDist = np.sum([rp[:, :, :, w]*rp[:, :, :, w] for w in range(3)], axis=0)
                thisMask = np.logical_or(thisMask,
                                         np.logical_or(raDist <= self.radius**2.0, rpDist <= self.radius**2.0))

            mask =  np.logical_or(mask, thisMask)

            # move to next point
            a = p

        return mask


 # Polyhedron with height
class PolarizableDeepConvexPolyhedron():
    def __init__(self, corners, height, permittivity):
        # sanity check
        assert(np.array(corners).shape[0]>2)  # at least three points
        assert(np.array(corners).shape[1]==3) # 3D
        assert(height>0)
        
        self.corners      = np.array(corners)/Bohr # from Angstroms to atomic units
        self.height       = height/Bohr  # from Angstroms to atomic units
        self.permittivity = permittivity

    def getMask(self, gd):
        parprint("Initializing Polarizable Deep Convex Polyhedron (%i corners)" % len(self.corners))
        
        # Vector (perpendicular to the plane) defining the plane 
        perpVector = np.cross(self.corners[1]-self.corners[0],
                              self.corners[-1]-self.corners[0])
        perpVector = -perpVector/np.linalg.norm(perpVector)
        vUp        = np.max(perpVector)>0
        
        # Ensure that all corners are in the same plane
        for k in range(len(self.corners)):
            assert 0 == np.linalg.norm(np.cross(perpVector, np.cross(self.corners[k]-self.corners[0],
                                                                     self.corners[-1]-self.corners[0])))
        
        # 3D coordinates at each grid point
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        ng = r_gv.shape[0:-1]
        ngv = r_gv.shape
        
        # Calculate the distances of all points to the plane
        dists = np.sum([perpVector[w]*(r_gv[:, :, :, w]-self.corners[0][w]) for w in range(3)], axis=0)
        mask = np.logical_and(0.0 <= dists, dists <= self.height)
        
        # Then calculate the projections of all points into our 2D plane
        projs = np.array([r_gv[:, :, :, w]-dists[:, :, :]*perpVector[w] for w in range(3)])
        
        # Then check the angles between all 2D vertices and the projected points: if any is >180dgr, the point is outside.
        # Here I use the condition that angle>180 if the points are given in clockwise order and the
        # minimum of the cross product vector coordinates is <0. If points are given in counter-clockwise order,
        # the maximum must be >0.
        num = len(self.corners)
        for p in range(num):
            m1 = np.array([self.corners[(p  )%num][w]-projs[w, :, :, :] for w in range(3)])
            m2 = np.array([self.corners[(p+1)%num][w]-projs[w, :, :, :] for w in range(3)])
            
            if vUp:
                mask = np.logical_and(mask, np.min(np.cross(m1, m2, axis=0), axis=0)<0)
            else:
                mask = np.logical_and(mask, np.max(np.cross(m1, m2, axis=0), axis=0)>0)

        return mask

class PolarizableTetrahedron():
    #http://steve.hollasch.net/cgindex/geometry/ptintet.html
    #obrecht@imagen.com (Doug Obrecht) writes:
    #
    # Can someone point me to an algorithm that determines if a point is within a tetrahedron?
    #
    # Let the tetrahedron have vertices
    #     V1 = (x1, y1, z1)
    #    V2 = (x2, y2, z2)
    #    V3 = (x3, y3, z3)
    #    V4 = (x4, y4, z4)
    #
    #and your test point be
    #
    #        P = (x, y, z).
    #Then the point P is in the tetrahedron if following five determinants all have the same sign.
    #
    #             |x1 y1 z1 1|
    #        D0 = |x2 y2 z2 1|
    #             |x3 y3 z3 1|
    #             |x4 y4 z4 1|
    #
    #             |x  y  z  1|
    #        D1 = |x2 y2 z2 1|
    #             |x3 y3 z3 1|
    #             |x4 y4 z4 1|
    #
    #             |x1 y1 z1 1|
    #        D2 = |x  y  z  1|
    #             |x3 y3 z3 1|
    #             |x4 y4 z4 1|
    # 
    #             |x1 y1 z1 1|
    #        D3 = |x2 y2 z2 1|
    #             |x  y  z  1|
    #             |x4 y4 z4 1|
    #
    #             |x1 y1 z1 1|
    #        D4 = |x2 y2 z2 1|
    #             |x3 y3 z3 1|
    #             |x  y  z  1|
    #
    # Some additional notes:
    #
    # If by chance the D0=0, then your tetrahedron is degenerate (the points are coplanar).
    # If any other Di=0, then P lies on boundary i (boundary i being that boundary formed by the three points other than Vi).
    # If the sign of any Di differs from that of D0 then P is outside boundary i.
    # If the sign of any Di equals that of D0 then P is inside boundary i.
    # If P is inside all 4 boundaries, then it is inside the tetrahedron.
    # As a check, it must be that D0 = D1+D2+D3+D4.
    # The pattern here should be clear; the computations can be extended to simplicies of any dimension. (The 2D and 3D case are the triangle and the tetrahedron).
    # If it is meaningful to you, the quantities bi = Di/D0 are the usual barycentric coordinates.
    # Comparing signs of Di and D0 is only a check that P and Vi are on the same side of boundary i.

    def __init__(self, corners, permittivity):
        # sanity check
        assert(len(corners)==4)     # exactly 4 points
        assert(len(corners[0])==3)  # 3D
        
        self.corners      = np.array(corners)/Bohr # from Angstroms to atomic units
        self.permittivity = permittivity

    def detValue(self, x, y, z,
                       x1, y1, z1,
                       x2, y2, z2,
                       x3, y3, z3,
                       x4, y4, z4, ind):
        mat = np.array([[x1, y1, z1, 1], [x2, y2, z2, 1], [x3, y3, z3, 1], [x4, y4, z4, 1]])
        mat[ind][:] = np.array([x, y, z, 1])
        return np.linalg.det(mat)

    def getMask(self, gd):
        parprint("Initializing Polarizable Tetrahedron")
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        ng = r_gv.shape[0:-1]
        ngv = r_gv.shape
        x1, y1, z1 = self.corners[0]
        x2, y2, z2 = self.corners[1]
        x3, y3, z3 = self.corners[2]
        x4, y4, z4 = self.corners[3]
        
        mask = np.ones(ng)==0
        
        # TODO: associate a determinant for each point, and use numpy tools to determine
        #       the mask without the *very slow* loop over grid points
        for ind, pt in np.ndenumerate(r_gv[:, :, :, 0]):
            x, y, z = r_gv[ind][:]
            d0 = np.array([[x1, y1, z1, 1],
                           [x2, y2, z2, 1],
                           [x3, y3, z3, 1],
                           [x4, y4, z4, 1]])
            d1 = np.array([[x,  y,  z, 1],
                           [x2, y2, z2, 1],
                           [x3, y3, z3, 1],
                           [x4, y4, z4, 1]])
            d2 = np.array([[x1, y1, z1, 1],
                           [x,  y,  z,  1],
                           [x3, y3, z3, 1],
                           [x4, y4, z4, 1]])
            d3 = np.array([[x1, y1, z1, 1],
                           [x2, y2, z2, 1],
                           [x,  y,  z,  1],
                           [x4, y4, z4, 1]])
            d4 = np.array([[x1, y1, z1, 1],
                           [x2, y2, z2, 1],
                           [x3, y3, z3, 1],
                           [x,  y,  z,  1]])
            s0 = np.linalg.det(d0)
            s1 = np.linalg.det(d1)
            s2 = np.linalg.det(d2)
            s3 = np.linalg.det(d3)
            s4 = np.linalg.det(d4)
            
            if (np.sign(s0)==np.sign(s1) or abs(s1)<1e-12) and \
               (np.sign(s0)==np.sign(s2) or abs(s2)<1e-12) and \
               (np.sign(s0)==np.sign(s3) or abs(s3)<1e-12) and \
               (np.sign(s0)==np.sign(s4) or abs(s4)<1e-12):
                mask[ind] = True
        return mask

        
       
class simpleMixer():
    def __init__(self, alpha, data):
        self.alpha = alpha
        self.data  = np.copy(data)
    
    def mix(self, data):
        self.data = self.alpha * data + (1.0-self.alpha) * self.data
        return np.copy(self.data)


# Lorentzian oscillator function: L(omega) = eps0 * beta / (w**2 - i*alpha*omega - omega**2)    // Coomar2011, Eq. 2
class LorentzOscillator:
    def __init__(self, barOmega, alpha, beta):
        self.barOmega = barOmega
        self.alpha    = alpha
        self.beta     = beta

    def value(self, omega):
        return _eps0_au * self.beta / (self.barOmega**2 - 1J * self.alpha * omega - omega**2)

# Dieletric function: e(omega) = eps_inf + sum_j L_j(omega) // Coomar2011, Eq. 2
class Permittivity:
    def __init__(self, fname=None, epsInfty = _eps0_au ):
        self.epsInfty = epsInfty

        if fname == None:
            # constant (vacuum?) permittivity
            self.Nj = 0
            self.oscillators = []
        else:
            # read permittivity from a 3-column file
            fp = open(fname, 'r')
            lines = fp.readlines()
            fp.close()

            self.Nj = len(lines)
            self.oscillators = []

            for line in lines:
                barOmega = float(split(line)[0]) / Hartree
                alpha    = float(split(line)[1]) / Hartree
                beta     = float(split(line)[2]) / Hartree / Hartree
                self.oscillators.append(LorentzOscillator(barOmega, alpha, beta))

    def value(self, omega = 0):
        return self.epsInfty + sum([osc.value(omega) for osc in self.oscillators])
    
    def plot(self, emin=1.0, emax=10.0, figtitle='Dielectric function', fname=None):
        try:
            from matplotlib.pyplot import plot, title, legend, show, clf
            xgrid = np.arange(emin, emax, float(emax-emin)/1000.0)
            ygrid = [self.value(x) for x in xgrid/Hartree]            
            plot(xgrid, np.real(ygrid)/_eps0_au, label='real part')
            title(figtitle)
            legend()
            show()
            clf()
            plot(xgrid, np.imag(ygrid)/_eps0_au, label='imag part')
            title(figtitle)
            legend()
            show()
            if not fname == None:
                datafile = file(fname, 'w')
                if datafile.tell() == 0:
                    for omega in xgrid:
                        datafile.write(' %22.12e %22.12e %22.12e\n' % (omega, np.real(self.value(omega/Hartree)), np.imag(self.value(omega/Hartree))))
                    datafile.close()
        except:
            parprint('Permittivity.plot: Plotting with matplotlib failed!')



# Dieletric function that renormalizes the static permittivity to wanted value (usually epsZero) 
class PermittivityPlus(Permittivity):
    def __init__(self, fname=None, epsInfty = _eps0_au, epsZero = _eps0_au, newBarOmega = 0.01, newAlpha = 0.10 ):
        Permittivity.__init__(self, fname, epsInfty)
        parprint("Original Nj=%i and eps(0) = %12.6f + i*%12.6f" % (self.Nj, self.value(0.0).real, self.value(0.0).imag))
        
        # convert given values from eVs to Hartrees
        _newBarOmega = newBarOmega / Hartree
        _newAlpha    = newAlpha / Hartree
        
        # evaluate the new value    
        _newBeta = ((epsZero - self.value(0.0))*_newBarOmega**2.0/_eps0_au).real
        self.oscillators.append(LorentzOscillator(_newBarOmega, _newAlpha, _newBeta))
        self.Nj = len(self.oscillators)
        parprint("Added following oscillator: (baromega, alpha, beta) = (%12.6f, %12.6g, %12.6f)" % (_newBarOmega*Hartree, _newAlpha*Hartree, _newBeta*Hartree*Hartree))
        parprint("New Nj=%i and eps(0) = %12.6f + i*%12.6f" % (self.Nj, self.value(0.0).real, self.value(0.0).imag))
            
