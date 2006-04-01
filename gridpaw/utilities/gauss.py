import Numeric as num
from Numeric import sqrt, pi, exp

# computer generated code:
Y_L = ['0.282094791774', '0.488602511903 * y', '0.488602511903 * z', '0.488602511903 * x', '1.09254843059 * x*y', '1.09254843059 * y*z', '0.315391565253 * (3*z*z-r2)', '1.09254843059 * x*z', '0.546274215296 * (x*x-y*y)', ]
gauss_L = ['sqrt(a0**3*4)/pi * exp(-a0*r2)', 'sqrt(a0**5*5.33333333333)/pi * y * exp(-a0*r2)', 'sqrt(a0**5*5.33333333333)/pi * z * exp(-a0*r2)', 'sqrt(a0**5*5.33333333333)/pi * x * exp(-a0*r2)', 'sqrt(a0**7*4.26666666667)/pi * x*y * exp(-a0*r2)', 'sqrt(a0**7*4.26666666667)/pi * y*z * exp(-a0*r2)', 'sqrt(a0**7*0.355555555556)/pi * (3*z*z-r2) * exp(-a0*r2)', 'sqrt(a0**7*4.26666666667)/pi * x*z * exp(-a0*r2)', 'sqrt(a0**7*1.06666666667)/pi * (x*x-y*y) * exp(-a0*r2)', ]
gausspot_L = ['2*sqrt(pi)*erf3D(sqrt(a0)*r)/r', '', '', '', '', '', '', '', '', ]

def L_to_lm(L):
    """convert L index to (l, m) index"""
    l = 0
    while L / (l+1.)**2 >= 1:  l += 1
    return l, L - l**2 - l

def lm_to_L(l,m):
    """convert (l, m) index to L index"""
    return l**2 + l + m

def coordinates(gd):
    """Constructs and returns matrices containing cartesian coordinates,
       and the square of the distance from the origin.
       The origin is placed in the center of the box described by the given
       grid-descriptor 'gd'.
    """    
    I  = num.indices(gd.n_c)
    dr = num.reshape(gd.h_c, (3, 1, 1, 1))
    r0 = num.reshape(gd.h_c * gd.beg0_c - .5 * gd.domain.cell_c, (3,1,1,1))
    r0 = num.ones(I.shape)*r0
    xyz = r0 + I * dr
    r2 = num.sum(xyz**2)

    # remove singularity at origin and replace with small number
    middle = gd.N_c / 2.
    # check that middle is a gridpoint and that it is on this CPU
    if num.alltrue(middle == num.floor(middle)) and \
           num.alltrue(gd.beg0_c <= middle < gd.end_c):
        m = (middle - gd.beg0_c).astype(int)
        r2[m[0], m[1], m[2]] = 1e-12

    # return r^2 matrix
    return xyz, r2

def erf3D(M):
    """Return matrix with the value of the error function evaluated for
       each element in input matrix 'M'.
    """
    from gridpaw.utilities import erf

    dim = M.shape
    res = num.zeros(dim,num.Float)
    for k in range(dim[0]):
        for l in range(dim[1]):
            for m in range(dim[2]):
                res[k, l, m] = erf(M[k, l, m])
    return res

class Gaussian:
    """Class offering several utilities related to the generalized gaussians:
                       _____                             2  
                      /  1       l!          l+3/2  -a0 r    l  m
       g (x,y,z) =   / ----- --------- (4 a0)      e        r  Y (x,y,z)
        L          \/  4 pi  (2l + 1)!                          l

       where a0 is the width of the gaussian, and Y_l^m is a real spherical
       harmonic.
       The gaussians are centered in the middle of input grid-descriptor.
    """
    def __init__(self, gd, a0=21.):
        self.gd = gd
        self.xyz, self.r2 = coordinates(gd)
        self.set_width(a0)

    def set_width(self, a0=21.):
        self.a0 = a0 / min(self.gd.domain.cell_c)
        
    def get_gauss(self, L):
        a0 = self.a0
        x  = self.xyz[0]
        y  = self.xyz[1]
        z  = self.xyz[2]
        r2 = self.r2
        return eval(gauss_L[L])

    def get_gauss_pot(self, L):
        a0 = self. a0
        r2 = self.r2
        r  = num.sqrt(r2)
        if L == 0:
            return eval(gausspot_L[L])
        else:
            raise NotImplementedError

    def get_moment(self ,n, L):
        r2 = self.r2
        x = self.xyz[0]
        y = self.xyz[1]
        z = self.xyz[2]
        return self.gd.integrate(n * eval(Y_L[L]))

    def remove_moment(self, n ,L, q=None):
        # determine multipole moment
        if q == None:
            q = self.get_moment(n, L)

        # Don't do anything if moment is less than the tolerance
        if abs(q) < 1e-7:
            return 0.

        # remove moment from input density
        n -= q * self.get_gauss(L)

        # return correction
        return q * self.get_gauss_pot(L)        

    def plot_gauss(self, L):
        from ASE.Visualization.VTK import VTKPlotArray
        cell = num.identity(3, num.Float)
        VTKPlotArray(self.get_gauss(L), cell)

## def construct_gauss(gd, a0=25.):
##     """Construct gaussian density and potential"""
##     # determine r^2 and r matrices
##     r2 = rSquared(gd)
##     r  = num.sqrt(r2)
##
##     # 'width' of gaussian distribution
##     # ng ~ exp(-a0 / 4) on the boundary of the domain
##     a = a0 / min(gd.domain.cell_c)**2
##
##     # gaussian density
##     ng = num.exp(-a * r2) * (a / num.pi)**(1.5)
##     
##     # gaussian potential
##     vg = erf3D(num.sqrt(a) * r) / r
##
##     # gaussian self energy
##     #Eg = -num.sqrt(0.5 * a / num.pi)
##
##     return ng, vg#, Eg

if __name__ == '__main__':
    from gridpaw.domain import Domain
    from gridpaw.grid_descriptor import GridDescriptor
    ## d  = Domain((2,2,2))   # domain object
    ## N  = 2**5              # number of grid points
    ## Nc = (N,N,N)           # tuple with number of grid point along each axis
    ## gd = GridDescriptor(d,Nc) # grid-descriptor object
    ## gauss = Gaussian(gd)
    ## gauss.plot_gauss(1)

    # test if multipole works
    d  = Domain((12,15,20))   # domain object
    N  = 2**5                 # number of grid points
    Nc = (N,N,N)              # tuple with number of grid point along each axis
    gd = GridDescriptor(d,Nc) # grid-descriptor object
    xyz, r2 = coordinates(gd) # matrix with the square of the radial coordinate
    r  = num.sqrt(r2)         # matrix with the values of the radial coordinate
    nH = num.exp(-2*r)/num.pi # density of the hydrogen atom
    gauss = Gaussian(gd)
    g = gauss.get_gauss(0)
    print gd.integrate(g), 2*sqrt(pi)
    print gauss.get_moment(g, 0), 1
    v = gauss.remove_moment(g, 0)
    print gd.integrate(g)
    print gauss.get_moment(g, 0)
    
    print ''
    print gauss.get_moment(nH, 0) 
    print gauss.get_moment(nH, 1) 
    print gauss.get_moment(nH, 2) 
    print gauss.get_moment(nH, 3) 
