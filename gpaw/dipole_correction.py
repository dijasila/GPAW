import numpy as np
from ase.utils import basestring

from gpaw.utilities import erf


class DipoleCorrection:
    """Dipole-correcting wrapper around another PoissonSolver."""
    def __init__(self, poissonsolver, direction):
        """Construct dipole correction object.

        poissonsolver:
            Poisson solver.
        direction: int or str
            Specification of layer: 0, 1, 2, 'xy', 'xz' or 'yz'.
        """
        self.c = direction
        self.poissonsolver = poissonsolver
        
        self.correction = None
        
    def write(self, writer):
        self.poissonsolver.write(writer)
        writer.write(direction=self.c)

    def get_stencil(self):
        return self.poissonsolver.get_stencil()

    def set_grid_descriptor(self, gd):
        if isinstance(self.c, basestring):
            axes = ['xyz'.index(d) for d in self.c]
            for c in range(3):
                if abs(gd.cell_cv[c, axes]).max() < 1e-12:
                    break
            else:
                raise ValueError('No axis perpendicular to {0}-plane!'
                                 .format(self.c))
            self.c = c
            
        if gd.pbc_c[self.c]:
            raise ValueError('System must be non-periodic perpendicular '
                             'to dipole-layer.')
            
        # Right now the dipole correction must be along one coordinate
        # axis and orthogonal to the two others.  The two others need not
        # be orthogonal to each other.
        for c1 in range(3):
            if c1 != self.c:
                if abs(np.dot(gd.cell_cv[self.c], gd.cell_cv[c1])) > 1e-12:
                    raise ValueError('Dipole correction axis must be '
                                     'orthogonal to the two other axes.')

        self.poissonsolver.set_grid_descriptor(gd)

    def get_description(self):
        poissondesc = self.poissonsolver.get_description()
        desc = 'Dipole correction along %s-axis' % 'xyz'[self.c]
        return '\n'.join([poissondesc, desc])

    def initialize(self):
        self.poissonsolver.initialize()

    def solve(self, phi, rho, **kwargs):
        gd = self.poissonsolver.gd
        drho, dphi, self.correction = dipole_correction(self.c, gd, rho)
        phi -= dphi
        iters = self.poissonsolver.solve(phi, rho + drho, **kwargs)
        phi += dphi
        return iters

    def estimate_memory(self, mem):
        self.poissonsolver.estimate_memory(mem)


def dipole_correction(c, gd, rhot_g):
    """Get dipole corrections to charge and potential.

    Returns arrays drhot_g and dphit_g such that if rhot_g has the
    potential phit_g, then rhot_g + drhot_g has the potential
    phit_g + dphit_g, where dphit_g is an error function.

    The error function is chosen so as to be largely constant at the
    cell boundaries and beyond.
    """
    # This implementation is not particularly economical memory-wise

    moment = gd.calculate_dipole_moment(rhot_g)[c]
    if abs(moment) < 1e-12:
        return gd.zeros(), gd.zeros(), 0.0

    r_g = gd.get_grid_point_coordinates()[c]
    cellsize = abs(gd.cell_cv[c, c])
    sr_g = 2.0 / cellsize * r_g - 1.0  # sr ~ 'scaled r'
    alpha = 12.0  # should perhaps be variable
    drho_g = sr_g * np.exp(-alpha * sr_g**2)
    moment2 = gd.calculate_dipole_moment(drho_g)[c]
    factor = -moment / moment2
    drho_g *= factor
    phifactor = factor * (np.pi / alpha)**1.5 * cellsize**2 / 4.0
    dphi_g = -phifactor * erf(sr_g * np.sqrt(alpha))
    return drho_g, dphi_g, phifactor

def get_pw_dipole_correction(c,gd,p):
    """Dipole correction following Bengtsson,PRB,59,12301

	Returns the array vdc_g.pckl, which corresponds to a linear
	potential, which is added to vHt_q after Fourier transformation.
	The potential counters the potential in vacuum and allows to
	determine the two work functions in assymetric slab calculations.
    """
	
    ng = gd.N_c[c]    #number of grid points in the corrected direction
    cs = abs(gd.cell_cv[c,c])     #cell size in the corrected direction
    nz0 = 0     			#XXX: should not be hard coded!
    vdc_g = gd.zeros()
    z0 = (nz0)/(ng)*cs     #z0=reference grid point in realspace

    #Introduction of a linear function, which counters the potential introduced by the dipole
    #Im sure this could be implemented in a nicer way
    v = 4 * np.pi * p[c] * cs/np.linalg.det(gd.cell_cv)
    for nz in range(nz0,nz0+ng):
         z = (float(nz%ng)/ng*cs)
	 vdc_i = -v * ((z-z0)/cs-0.5)
         if c == 0: vdc_g[nz%ng,:,:] = vdc_i
         elif c == 1: vdc_g[:,nz%ng,:] = vdc_i
         else: vdc_g[:,:,nz%ng] = vdc_i
    
    #Introduction of an error function in order to counter the discontinuity at the boundary
    for i in range(-5,5):
	erfun = (-erf(0.5*float(-i))+1)
	if c == 0:
	   vdc_g[(i-nz0)%ng,:,:] = vdc_g[(-6-nz0)%ng,:,:]-erfun*(vdc_g[(-6-nz0)%ng,:,:]-vdc_g[(5-nz0)%ng,:,:])/2
	elif c == 1:
	   vdc_g[:,(i-nz0)%ng,:] = vdc_g[:,(-6-nz0)%ng,:]-erfun*(vdc_g[:,(-6-nz0)%ng,:]-vdc_g[:,(5-nz0)%ng,:])/2
	else:
	   vdc_g[:,:,(i-nz0)%ng] = vdc_g[:,:,(-6-nz0)%ng]-erfun*(vdc_g[:,:,(-6-nz0)%ng]-vdc_g[:,:,(5-nz0)%ng])/2
    return vdc_g,-v*0.5
