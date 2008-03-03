"""Used to generate polarization functions for atomic basis sets."""

import sys
import math
import traceback

import numpy as npy
from ase import Atom, Atoms
try:
    import pylab
except ImportError:
    print 'Pylab not imported'

from gpaw import Calculator
from gpaw.kpoint import KPoint
from gpaw.domain import Domain
from gpaw.grid_descriptor import GridDescriptor
from gpaw.spline import Spline
from gpaw.localized_functions import create_localized_functions
from gpaw.atom.all_electron import AllElectron
from gpaw.atom.configurations import configurations
from gpaw.testing import g2
from gpaw.testing.amoeba import Amoeba

class QuasiGaussian:
    """Gaussian-like functions for expansion of orbitals.

    Implements f(r) = A [G(r) - P(r)] where::

      G(r) = exp{- alpha r^2}
      P(r) = a - b r^2

    with (a, b) such that f(rcut) == f'(rcut) == 0.
    """
    def __init__(self, alpha, rcut, A=1.):
        self.alpha = alpha
        self.rcut = rcut
        a, b = get_polynomial_coefficients(alpha, rcut)
        self.a = a
        self.b = b
        self.A = A
        
    def __call__(self, r):
        """Evaluate function values at r, which is a numpy array."""
        condition = (r < self.rcut) & (self.alpha * r**2 < 700.)
        r2 = npy.where(condition, r**2., 0.) # prevent overflow
        g = npy.exp(-self.alpha * r2)
        p = (self.a - self.b * r2)
        y = npy.where(condition, g - p, 0.)
        return self.A * y

    def renormalize(self, norm):
        """Divide function by norm."""
        self.A /= norm

class LinearCombination:
    """Represents a linear combination of 1D functions."""
    def __init__(self, coefs, functions):
        self.coefs = coefs
        self.functions = functions

    def __call__(self, r):
        """Evaluate function values at r, which is a numpy array."""
        return sum([coef * function(r) for coef, function
                    in zip(self.coefs, self.functions)])

    def renormalize(self, norm):
        """Divide coefficients by norm."""
        self.coefs = [coef/norm for coef in self.coefs]

def get_polynomial_coefficients(alpha, rcut):
    """Determine polynomial used to truncate Gaussians.

    Returns the coefficients (a, b) of the polynomial p(r) = a - b r^2,
    such that the polynomial joins exp(-alpha r**2) differentiably at r=rcut.
    """
    expmar2 = math.exp(-alpha * rcut**2)
    a = (1 + alpha * rcut**2) * expmar2
    b = alpha * expmar2
    return a, b

def gramschmidt(gd, psit_k):
    """Orthonormalize functions on grid using the Gram-Schmidt method.

    Modifies the elements of psit_k such that each scalar product
    < psit_k[i] | psit_k[j] > = delta[ij], where psit_k are on the grid gd"""
    for k in range(len(psit_k)):
        psi = psit_k[k]
        for l in range(k):
            phi = psit_k[l]
            psit_k[k] = psi - gd.integrate(psi*phi) * phi
        psi = psit_k[k]
        psit_k[k] = psi / gd.integrate(psi*psi)**.5

def make_dimer_reference_calculation(formula, a):
    # this function has not been tested since npy migration
    system = g2.get_g2(formula, (a, a, a,))
    calc = Calculator(h=.25)
    system.set_calculator(calc)
    system.get_potential_energy()
    psit = calc.kpt_u[0].psit_nG
    return calc.gd, psit, (system.positions/system.get_cell().diagonal())[0]

def rotation_test():
    molecule = 'NH3'
    a = 7.
    rcut = 5.
    l = 1

    from gpaw.output import plot

    rotationvector = npy.array([1.,1.,1.])
    angle_increment = .3
    
    system = g2.get_g2(molecule, (a,a,a))
    calc = Calculator(h=.27, txt=None)
    system.set_calculator(calc)

    pog = PolarizationOrbitalGenerator(rcut)

    r = npy.linspace(0., rcut, 300)

    maxvalues = []

    for i in range(0, int(6.28/angle_increment)):
        ascii = plot(system.positions,
                     system.get_atomic_numbers(),
                     system.get_cell().diagonal())

        print ascii
        
        print 'angle=%.03f' % (angle_increment * i)
        energy = system.get_potential_energy()
        center = (system.positions / system.get_cell().diagonal())[0]
        orbital = pog.generate(l, calc.gd, calc.kpt_u[0].psit_nG, center)
        y = orbital(r)
        pylab.plot(r, y, label='%.02f' % (i * angle_increment))
        maxvalues.append(max(y))
        print 'Quality by orbital', pretty(pog.optimizer.lastterms)
        system.rotate(rotationvector, angle_increment)
        system.center()

    print max(maxvalues) - min(maxvalues)

    pylab.legend()
    pylab.show()

    #orig = system.copy()
    #orig.center(vacuum=a/2.)
    #
    #orig.set_calculator(calc)
    #print 'Calculating energy in g2 system'
    #orig.get_potential_energy()

    #center = (orig.positions / orig.get_cell().diagonal())[0]
    #gd, psit_k, center = calc.gd, calc.kpt_u[0].psit_nG, orig.positions[0]


    #r = npy.linspace(0., rcut, 300)
    #pylab.plot(r, orbital(r))

    #x = system.positions[0][2] / (3.**.5)
    #positions = npy.array([[x,x,x],
    #                       [-x,-x,-x]])
    #system.set_positions(positions)
    #system.center(vacuum=a/2.)

    #system.set_calculator(calc)
    #system.get_potential_energy()

    #center = (system.positions / system.get_cell().diagonal())[0]
    #orbital = pog.generate(l, calc.gd, calc.kpt_u[0].psit_nG, center)
    #print 'Quality by orbital', pretty(pog.optimizer.lastterms)

def make_dummy_calculation(l, function=None, rcut=6., a=12., n=60,
                           dtype=float):
    """Make a mock reference wave function using a made-up radial function
    as reference"""
    #print 'Dummy reference: l=%d, rcut=%.02f, alpha=%.02f' % (l, rcut, alpha)
    r = npy.arange(0., rcut, .01)

    if function is None:
        function = QuasiGaussian(4., rcut)

    norm = get_norm(r, function(r), l)
    function.renormalize(norm)
    #g = QuasiGaussian(alpha, rcut)
    
    mcount = 2*l + 1
    fcount = 1
    domain = Domain((a, a, a), (False, False, False))
    gd = GridDescriptor(domain, (n, n, n))
    spline = Spline(l, r[-1], function(r), points=50)
    center = (.5, .5, .5)
    lf = create_localized_functions([spline], gd, center, dtype=dtype)
    psit_k = gd.zeros(mcount, dtype=dtype)
    coef_xi = npy.identity(mcount * fcount, dtype=dtype)
    lf.add(psit_k, coef_xi)
    return gd, psit_k, center, function

class CoefficientOptimizer:
    """Class for optimizing Gaussian/reference overlap.

    Given matrices of scalar products s and S as returned by get_matrices,
    finds the optimal set of coefficients resulting in the largest overlap.

    ccount is the number of coefficients.
    if fix is True, the first coefficient will be set to 1. and only the
    remaining coefficients will be subject to optimization.
    """
    def __init__(self, s_kmii, S_kmii, ccount, fix=False):
        self.s_kmii = s_kmii
        self.S_kmii = S_kmii
        self.fix = fix
        function = self.evaluate
        self.lastterms = None
        if fix:
            function = self.evaluate_fixed
            ccount -= 1
        ones = npy.ones((ccount, ccount))
        diag = npy.identity(ccount)
        simplex = npy.concatenate((npy.ones((ccount,1)),
                                   ones + .5 * diag), axis=1)
        simplex = npy.transpose(simplex)
        self.amoeba = Amoeba(simplex, function=function, tolerance=1e-10)
        
    def find_coefficients(self):
        self.amoeba.optimize()
        coefficients = self.amoeba.simplex[0]
        if self.fix:
            coefficients = [1.] + list(coefficients)
        return coefficients

    def evaluate_fixed(self, coef):
        return self.evaluate([1.] + list(coef))

    def evaluate(self, coef):
        ncoef = len(coef)

        coef_trans = npy.array([coef]) # complex coefficients?
        coef = coef_trans.transpose()

        terms_km = npy.zeros((ncoef, ncoef))

        for i, (s_mii, S_mii) in enumerate(zip(self.s_kmii, self.S_kmii)):
            for j, (s_ii, S_ii) in enumerate(zip(s_mii, S_mii)):
                numerator = npy.dot(coef_trans, npy.dot(S_ii, coef))
                denominator = npy.dot(coef_trans, npy.dot(s_ii, coef))
                assert numerator.shape == (1,1)
                assert denominator.shape == (1,1)                
                terms_km[i, j] = numerator[0,0] / denominator[0,0]
        
        self.lastterms = terms_km
        quality = terms_km.sum()
        badness = - quality
        return badness

    def old_evaluate(self, coef):
        coef_trans = npy.array([coef]) # complex coefficients?
        coef = coef_trans.transpose()

        numerator = npy.array([npy.dot(coef_trans, npy.dot(S, coef))
                               for S in self.S_kmii[0]])
        denominator = npy.array([npy.dot(coef_trans, npy.dot(s, coef))
                                 for s in self.s_kmii[0]])
        
        terms = numerator / denominator
        assert terms.shape == (len(self.S_kmii[0]), 1, 1)
        self.lastterms = terms[:, 0, 0]
        quality = terms.sum()
        badness = - quality
        return badness

def pretty(floatlist):
    return ' '.join(['%.02f' % f for f in floatlist])

def norm_squared(r, f, l):
    dr = r[1]
    frl = f * r**l
    assert abs(r[1] - (r[-1] - r[-2])) < 1e-10 # error if not equidistant
    return sum(frl * frl * r * r * dr)

def get_norm(r, f, l=0):
    return norm_squared(r, f, l) ** .5

class PolarizationOrbitalGenerator:
    """Convenience class which generates polarization functions."""
    def __init__(self, rcut, gaussians=None):
        self.rcut = rcut
        if gaussians is None:
            gaussians = int(rcut / .3) # lots!
        self.r_alphas = npy.linspace(1., .6 * rcut, gaussians)
        self.alphas = 1. / self.r_alphas**2
        self.s = None
        self.S = None
        self.optimizer = None

    def generate(self, l, gd, kpt_u, center, dtype=None):
        """Generate polarization orbital."""

        rcut = self.rcut
        phi_i = [QuasiGaussian(alpha, rcut) for alpha in self.alphas]
        r = npy.arange(0, rcut, .01)
        dr = r[1] # equidistant
        integration_multiplier = r**(2*(l+1))
        for phi in phi_i:
            y = phi(r)
            norm = (dr * sum(y * y * integration_multiplier))**.5
            phi.renormalize(norm)
        splines = [Spline(l, r[-1], phi(r), points=50) for phi in phi_i]

        if dtype is None:
            if len(kpt_u) == 1:
                dtype = float
            else:
                dtype = complex

        if  dtype == complex:
            self.s, self.S = multkpts_matrices(l, gd, splines, kpt_u, center)
        elif dtype == float:
            self.s, self.S = get_matrices(l, gd, splines, kpt_u, center)
        else:
            raise ValueError('Bad dtype')
        
        self.optimizer = CoefficientOptimizer(self.s, self.S, len(phi_i))
        coefs = self.optimizer.find_coefficients()
        self.quality = - self.optimizer.amoeba.y[0]
        self.qualities = self.optimizer.lastterms
        orbital = LinearCombination(coefs, phi_i)
        orbital.renormalize(get_norm(r, orbital(r), l))
        return orbital

def multkpts_matrices(l, gd, splines, kpt_u, center=(.5, .5, .5)):
    """Get scalar products of basis functions and references.

    Returns the triple-indexed matrices s and S, where::

        s    = < phi   | phi   > ,
         mij        mi      mj

              -----
               \     /        |  ~   \   /  ~   |        \
        S    =  )   (  phi    | psi   ) (  psi  | phi     )
         mij   /     \    mi  |    k /   \    k |     mj /
              -----
                k

    The functions phi are taken from the given splines, whereas psit
    must be on the grid represented by the GridDescriptor gd.
    Integrals are evaluated at the relative location given by center.
    """
    dtype = complex
    lvalues = [spline.get_angular_momentum_number() for spline in splines]
    assert lvalues.count(l) == len(lvalues) # all must be equal
    mcount = 2*l + 1
    fcount = len(splines)
    lf = create_localized_functions(splines, gd, center, dtype=dtype)
    k_kc = [kpt.k_c for kpt in kpt_u]
    lf.set_phase_factors(k_kc)

    kcount = len(kpt_u)
    bcount = kpt_u[0].psit_nG.shape[0]
    for kpt in kpt_u:
        assert kpt.psit_nG.shape[0] == bcount

    # First we have to calculate the scalar products between
    # pairs of basis functions < phi_kmi | phi_kmj >.

    s_kmii = npy.zeros((kcount, mcount, fcount, fcount), dtype=dtype)
    coef_xi = npy.identity(mcount * fcount, dtype=dtype)
    #phi_miG = gd.zeros(mcount * fcount, dtype=dtype)
    for kpt in kpt_u:
        phi_nG = gd.zeros(mcount * fcount, dtype=dtype)
        lf.add(phi_nG, coef_xi, k=kpt.k)
        phi_overlaps_ii = npy.zeros((fcount * mcount,
                                     fcount * mcount), dtype=dtype)
        lf.integrate(phi_nG, phi_overlaps_ii, k=kpt.k)
        phi_overlaps_ii.shape = (fcount, mcount, fcount, mcount)
        for m in range(mcount):
            for i in range(fcount):
                for j in range(fcount):
                    s_kmii[kpt.k, m, i, j] = phi_overlaps_ii[i, m, j, m]
    
    # Now calculate scalar products between basis functions and
    # reference functions < phi_kmi | psi_kn >.
    
    overlaps_knmi = npy.zeros((kcount, bcount, mcount, fcount), dtype=dtype)
    for kpt in kpt_u:
        overlaps_mii = npy.zeros((bcount, mcount * fcount), dtype=dtype)
        lf.integrate(kpt.psit_nG, overlaps_mii, k=kpt.k)
        overlaps_mii.shape = (bcount, fcount, mcount)
        overlaps_knmi[kpt.k, :, :, :] = overlaps_mii.swapaxes(1, 2)
        
    S_kmii = npy.zeros((kcount, mcount, fcount, fcount), dtype=dtype)
    conj_overlaps_knmi = overlaps_knmi.conjugate()

    for k in range(kcount):
        for m in range(mcount):
            for i in range(fcount):
                for j in range(fcount):
                    x1 = conj_overlaps_knmi[k, :, m, i]
                    x2 = overlaps_knmi[k, :, m, j]
                    S_kmii[k, m, i, j] = (x1 * x2).sum()

    #print 's_kmii.shape, diags', s_kmii.shape
    #print s_kmii[1,0].diagonal()
    #print s_kmii[1,1].diagonal()
    #print s_kmii[1,2].diagonal()

    #print 'S_kmii diags'
    #print S_kmii[1,0].diagonal()
    #print S_kmii[1,1].diagonal()
    #print S_kmii[1,2].diagonal()

    assert s_kmii.shape == S_kmii.shape

    return s_kmii, S_kmii

def get_matrices(l, gd, splines, kpt_u, center=(.5, .5, .5)):
    """Get scalar products of basis functions and references

    Returns the triple-indexed matrices s and S, where::

        s    = < phi   | phi   > ,
         mij        mi      mj

              -----
               \     /        |  ~   \   /  ~   |        \
        S    =  )   (  phi    | psi   ) (  psi  | phi     )
         mij   /     \    mi  |    k /   \    k |     mj /
              -----
                k

    The functions phi are taken from the given splines, whereas psit
    must be on the grid represented by the GridDescriptor gd.
    Integrals are evaluated at the relative location given by center.
    """
    assert len(kpt_u) == 1, 'This method only works for one k-point'
    kpt = kpt_u[0]
    psit_k = kpt.psit_nG
    
    mcounts = [spline.get_angular_momentum_number() for spline in splines]
    mcount = mcounts[0]
    for mcount_i in mcounts:
        assert mcount == mcount_i
    mcount = 2*l + 1
    fcount = len(splines)
    phi_lf = create_localized_functions(splines, gd, center)
    phi_mi = gd.zeros(fcount*mcount) # one set for each phi
    coef_xi = npy.identity(fcount*mcount)
    phi_lf.add(phi_mi, coef_xi)
    integrals = npy.zeros((fcount*mcount, fcount*mcount))
    phi_lf.integrate(phi_mi, integrals)
    """Integral matrix contents (assuming l==1 so there are three m-values)

                --phi1--  --phi2--  --phi3-- ...
                m1 m2 m3  m1 m2 m3  m1 m2 m3 ...
               +---------------------------------
               |
         |   m1| x 0 0     x 0 0
        phi1 m2| 0 x 0     0 x 0   ...
         |   m3| 0 0 x     0 0 x 
               |
         |   m1|   .
        phi2 m2|   .
         |   m3|   .
             . |
             .

    We want < phi_mi | phi_mj >, and thus only the diagonal elements of
    each submatrix!  For l=1 the diagonal elements are all equal, but this
    is not true in general"""

    # phiproducts: for each m, < phi_mi | phi_mj >
    phiproducts_mij = npy.zeros((mcount, fcount, fcount))
    for i in range(fcount):
        for j in range(fcount):
            ioff = mcount*i
            joff = mcount*j
            submatrix_ij = integrals[ioff:ioff+mcount,joff:joff+mcount]
            phiproducts_mij[:, i, j] = submatrix_ij.diagonal()
    # This should be ones in submatrix diagonals and zero elsewhere

    # Now calculate scalar products < phi_mi | psit_k >, where psit_k are
    # solutions from reference calculation
    psitcount = len(psit_k)
    psi_phi_integrals = npy.zeros((psitcount, fcount*mcount))
    phi_lf.integrate(psit_k, psi_phi_integrals)

    # Now psiproducts[k] is a flat list, but we want it to be a matrix with
    # dimensions corresponding to f and m.
    # The first three elements correspond to the same localized function
    # and so on.
    # What we want is one whole matrix for each m-value.
    psiproducts_mik = npy.zeros((mcount, fcount, psitcount))
    for m in range(mcount):
        for i in range(fcount):
            for k in range(psitcount):
                psiproducts_mik[m, i, k] = psi_phi_integrals[k, mcount * i + m]

    # s[mij] = < phi_mi | phi_mj >
    s = npy.array([phiproducts_mij])

    # S[mij] = sum over k: < phi_mi | psit_k > < psit_k | phi_mj >
    S = npy.array([[npy.dot(psiproducts_ik, npy.transpose(psiproducts_ik))
                    for psiproducts_ik in psiproducts_mik]])


    #assert mcount == 3
    #print 's_kmii.shape, diags', s.shape
    #print s[0,0].diagonal()
    #print s[0,1].diagonal()
    #print s[0,2].diagonal()

    #print 'S_kmii diags'
    #print S[0,0].diagonal()
    #print S[0,1].diagonal()
    #print S[0,2].diagonal()

    return s, S

def main():
    """Testing."""
    args = sys.argv[1:]
    if len(args) == 0:
        args = g2.atoms.keys()
    rcut = 6.
    generator = PolarizationOrbitalGenerator(rcut)
    for symbol in args:
        gd, psit_k, center = Reference(symbol, txt=None).get_reference_data()
        psitcount = len(psit_k)
        gramschmidt(gd, psit_k)
        print 'Wave function count', psitcount
        psit_k_norms = gd.integrate(psit_k * psit_k)

        Z, states = configurations[symbol]
        highest_state = states[-1]
        n, l_atom, occupation, energy = highest_state
        l = l_atom + 1
        
        phi = generator.generate(l, gd, psit_k, center, dtype=dtype)
        #print 'Quality by orbital', pretty(generator.optimizer.lastterms)
        
        r = npy.arange(0., rcut, .01)
        norm = get_norm(r, phi(r), l)

        quality = generator.quality
        orbital = 'spdf'[l]
        style = ['-.', '--','-',':'][l]
        pylab.plot(r, phi(r) * r**l, style,
                   label='%s [type=%s][q=%.03f]' % (symbol, orbital, quality))
    pylab.legend()
    pylab.show()

def dummy_test(lmax=4, rcut=6., lmin=0): # fix args
    """Run a test using a Gaussian reference function."""
    dtype = complex
    generator = PolarizationOrbitalGenerator(rcut, gaussians=4)
    r = npy.arange(0., rcut, .01)
    alpha_ref = 1. / (rcut/4.) ** 2.
    for l in range(lmin, lmax + 1):
        g = QuasiGaussian(alpha_ref, rcut)
        norm = get_norm(r, g(r), l)
        g.renormalize(norm)
        gd, psit_k, center, ref = make_dummy_calculation(l, g, rcut,
                                                         dtype=dtype)
        k_kc = ((0.,0.,0.), (.5,.5,.5))
        kpt_u = [KPoint(gd, 1., 0, i, i, k_c, dtype=dtype)
                 for i, k_c in enumerate(k_kc)]
        for kpt in kpt_u:
            kpt.psit_nG = psit_k
        
        phi = generator.generate(l, gd, kpt_u, center, dtype=dtype)
        
        pylab.figure(l)
        #pylab.plot(r, ref(r)*r**l, 'g', label='ref')
        pylab.plot(r, g(r)*r**l, 'b', label='g')
        pylab.plot(r, phi(r)*r**l, 'r--', label='pol')
        pylab.title('l=%d' % l)
        pylab.legend()
    pylab.show()

restart_filename = 'ref.%s.gpw'
output_filename = 'ref.%s.txt'

# Systems for non-dimer-forming or troublesome atoms
# 'symbol' : (g2 key, index of desired atom)

special_systems = {'H' : ('HCl', 1), # Better results with more states
                   'Li' : ('LiF', 0), # More states
                   'Na' : ('NaCl', 0), # More states
                   'B' : ('BCl3', 0), # No boron dimer
                   'C' : ('CH4', 0), # No carbon dimer
                   'N' : ('NH3', 0), # double/triple bonds tend to be bad
                   'O' : ('H2O', 0), # O2 requires spin polarization
                   'Al' : ('AlCl3', 0),
                   'Si' : ('SiO', 0), # No reason really.
                   'S' : ('SO2', 0), # S2 requires spin polarization
                   'P' : ('PH3', 0)}

# Calculator parameters for particularly troublesome systems
#special_parameters = {('Be2', 'h') : .2, # malloc thing?
#                      ('Na2', 'h') : .25,
#                      ('Na2', 'a') : 20., # Na is RIDICULOUSLY large
#                      ('LiF', 'h') : .22, # this is probably bad for F
#                      ('LiF', 'a') : 18.} # also large


# Errors with standard parameters
#
# Be2, AlCl3:
# python: c/extensions.h:29: gpaw_malloc: Assertion `p != ((void *)0)' failed.
#
# NaCl, Li:
# Box too close to boundary

def check_magmoms():
    systems = get_systems()
    for formula, index in systems:
        atoms = g2.get_g2(formula, (0, 0, 0))
        try:
            magmom = atoms.get_magnetic_moments()
            if magmom is None:
                raise KeyError
        except KeyError:
            pass
        else:
            if magmom.any():
                print formula, 'has nonzero magnetic moment!!'
    
def get_system(symbol):
    """Get default reference formula or atomic index."""
    system = special_systems.get(symbol)
    if system is None:
        system = (symbol + '2', 0)
    return system

def get_systems(symbols=None):
    if symbols is None:
        symbols = g2.atoms.keys()
    systems = []
    for symbol in symbols:
        systems.append(get_system(symbol))
    return systems

def multiple_calculations(systems=None, a=None, h=None):
    if systems is None:
        systems = zip(*get_systems())[0]
    formulas = [] # We want a list of unique formulas in case some elements use
    # the same system
    for system in systems:
        if not system in formulas:
            formulas.append(system)
        
    print 'All:', formulas
    for formula in formulas:
        try:
            print formula,
            
            #if h is None:
            #    h = special_parameters.get((formula, 'h'), .17)
            #if a is None:
            #    a = special_parameters.get((formula, 'a'), 14.)

            print '[a=%.03f, h=%.03f] ... ' % (a, h),

            sys.stdout.flush()
                
            make_reference_calculation(formula, a, h)
            print 'OK!'
        except KeyboardInterrupt:
            raise
        except:
            print 'FAILED!'
            traceback.print_exc()

def make_reference_calculation(formula, a, h):
    calc = Calculator(h=h, xc='PBE', txt=output_filename % formula)
    system = g2.get_g2(formula, (a, a, a))
    assert system.get_magnetic_moments() is None
    system.set_calculator(calc)
    energy = system.get_potential_energy()
    calc.write(restart_filename % formula, mode='all')

class Reference:
    """Represents a reference function loaded from a file."""
    def __init__(self, symbol, filename=None, index=None, txt=None):
        if filename is None:
            formula, index = get_system(symbol)
            filename = restart_filename % formula
        calc = Calculator(filename, txt=txt)
        print 'Loaded reference data'
        print 'kpts', calc.kpt_u

        
        atoms = calc.get_atoms()
        symbols = atoms.get_chemical_symbols()
        if index is None:
            index = symbols.index(symbol)
        else:
            if not symbols[index] == symbol:
                raise ValueError(('Atom (%s) at specified index (%d) not of '+
                                  'requested type (%s)') % (symbols[index],
                                                            index, symbol))
        self.calc = calc
        self.filename = filename
        self.atoms = atoms
        self.symbol = symbol
        self.symbols = symbols
        self.index = index
        self.center = atoms.positions[index]

        self.cell = atoms.get_cell().diagonal() # cubic cell
        self.gpts = calc.gd.N_c
        if calc.kpt_u[0].psit_nG is None:
            raise RuntimeError('No wave functions found in .gpw file')

    def get_reference_data(self):
        c = self.calc
        for kpt in c.kpt_u:
            kpt.psit_nG = kpt.psit_nG[:] # load wave functions from the file
            # this is an ugly way to do it, by the way, but it probably works
        return c.gd, c.kpt_u, self.center / self.cell

if __name__ == '__main__':
    dummy_test(lmin=1, lmax=1)
    #rotation_test()

#def load_reference(symbol, filename=None, index=None, txt=None):
    #print 'Loading reference for %s from disk.' % symbol
