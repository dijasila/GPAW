import os
import sys

from math import pi, cos, sin
import numpy as npy
from numpy.linalg import solve
from gpaw.atom.generator import Generator, parameters
from gpaw.atom import polarization
from gpaw.utilities import devnull

AMPLITUDE = 100. # default confinement potential modifier

hartree = 27.2113845

class BasisFunction:
    """Wrapper for the information one usually wants to accompany
    a basis function"""
    def __init__(self, psi, rc, n, l, j, zeta=0, pol=0):
        self.psi = psi
        self.rc = rc
        self.n = n
        self.l = l
        self.j = j # this is the index for the lists u_j, l_j in AllElectron
        self.zeta = zeta
        self.pol = pol

class BasisMaker:
    """
    This class is meant for generating basis vectors that can be used to
    represent an element in LCAO-calculations.
    """
    def __init__(self, generator, run=True):
        if isinstance(generator, str): # treat 'generator' as symbol
            generator = Generator(generator, scalarrel=True)
        self.generator = generator
        if run:
            generator.run(**parameters[generator.symbol])

    def get_unsmoothed_projector_coefficients(self, psi, l):
        """
        Returns a matrix with (i,j)'th element equal to

        < p  | psi  > ,
           i      j

        where the argument psi is the coefficient matrix for a system of
        vectors, and where p_i are determined by

                        -----
           ~             \      ~
         < p  | psi  > =  )   < p  | phi  > < p  | psi  > ,
            i      j     /       i      k      k      j
                        -----
                          k

        where p_i-tilde are the projectors and phi_k the AE partial waves.
        """
        if npy.rank(psi) == 1:
            # vector/matrix polymorphism hack
            return self.get_unsmoothed_projector_coefficients([psi], l)[0]

        g = self.generator
        u = g.u_ln[l]
        q = g.q_ln[l]

        m = len(q)
        n = len(u)

        A = npy.zeros((m, n))
        b = npy.zeros((m, len(psi)))

        for i in range(m):
            for j in range(n):
                A[i, j] = npy.dot(g.dr, q[i] * u[j])

        for i in range(m):
            for j in range(len(psi)):
                b[i, j] = npy.dot(g.dr, q[i] * psi[j])

        p = solve(A, b)
        return p

    def unsmoothify(self, psi_t, l):
        """
        Converts each column of psi_tilde, interpreted as a pseudo wave
        function, to original wave functions using the formula

                              -----
                      ~        \    /             ~    \    ~     ~
        | psi  > = | psi  > +   )  ( | phi > - | phi  > ) < p  | psi  >
             i          i      /    \     j         j  /     j      i
                              -----
                                j
        """
        if npy.rank(psi_t) == 1:
            # vector/matrix polymorphism hack
            return self.unsmoothify([psi_t], l)[0]
        g = self.generator
        (q, u, s) = (g.q_ln[l], g.u_ln[l], g.s_ln[l])
        
        psi = [psi_t[j] + sum([(u[i]-s[i])*q[i,j]
                               for i in range(len(s))])
               for j in range(len(psi_t))]
        return psi

    def smoothify(self, psi, l):
        """
        Converts each column of psi_tilde, interpreted as a pseudo wave
        function, to original wave functions using the formula

                              -----
           ~                   \    /   ~              \           
        | psi  > = | psi  > +   )  ( | phi > - | phi  > ) < p  | psi  >
             i          i      /    \     j         j  /     j      i
                              -----
                                j

        """
        if npy.rank(psi) == 1:
            # vector/matrix polymorphism hack
            return self.smoothify([psi], l)[0]
        
        p = self.get_unsmoothed_projector_coefficients(psi, l)
        g = self.generator
        (q, u, s) = (g.q_ln[l], g.u_ln[l], g.s_ln[l])
        
        psi_tilde = [psi[j] + sum([(s[i]-u[i])*p[i,j]
                                   for i in range(len(s))])
                     for j in range(len(psi))]
        return psi_tilde

    def make_orbital_vector(self, j, rcut, vconf=None):
        l = self.generator.l_j[j]
        print 'j =',j,', l =',l, 'rc',rcut
        psi, e = self.generator.solve_confined(j, rcut, vconf)
        psi_t = self.smoothify(psi, l)
        return psi_t

    def make_split_valence_vector(self, psi, l, rcut):
        """Returns an array of function values f(r) * r, where
                l           2
        f(r) = r  * (a - b r ),  r < rcut
        f(r) = psi(r),           r >= rcut

        where a and b are determined such that f(r) is continuous and
        differentiable at rcut.  The parameter psi should be an atomic
        orbital.
        """
        g = self.generator
        icut = g.r2g(rcut)
        r1 = g.r[icut] # ensure that rcut is moved to a grid point
        r2 = g.r[icut + 1]
        y1 = psi[icut] / g.r[icut]
        y2 = psi[icut + 1] / g.r[icut + 1]
        b = - (y2 / r2**l - y1 / r1**l)/(r2**2 - r1**2)
        a = (y1 / r1**l + b * r1**2)
        psi2 = g.r**(l + 1) * (a - b * g.r**2)
        psi2[icut:] = psi[icut:]
        return psi2

    def make_polarization_function(self, rcut, l, referencefile=None, 
                                   index=None, txt=devnull):
        symbol = self.generator.symbol
        ref = polarization.Reference(symbol, referencefile, index)
        gd, psit_k, center = ref.get_reference_data()
        symbols = ref.atoms.get_chemical_symbols()
        symbols[ref.index] = '[%s]' % symbols[ref.index] # mark relevant atom
        print >> txt, 'Reference system <%s>:' % ref.filename,
        print >> txt, ' '.join(['%s' % sym for sym in symbols])
        cell = ' x '.join(['%.02f' % a for a in ref.cell])
        print >> txt, 'Cell = %s :: gpts = %s' % (cell, ref.gpts)
        generator = polarization.PolarizationOrbitalGenerator(rcut)
        y = generator.generate(l, gd, psit_k, center)
        print >> txt, 'Quasi Gaussians: %d' % len(generator.alphas)
        print >> txt, 'Reference states: %d' % len(psit_k)
        qualities = ', '.join(['%.03f' % q for q in generator.qualities])
        print >> txt, 'Quality: %.03f [%s]' % (generator.quality, qualities)
        r = self.generator.r
        psi = r**l * y(r)
        return psi * r # Recall that wave functions are represented as psi*r

    def make_mock_vector(self, rcut, l):
        r = self.generator.r
        x = r / rcut
        y = (1 - 3 * x**2 + 2 * x**3) * x**l
        icut = self.generator.r2g(rcut)
        y[icut:] *= 0
        return y * r # Recall that wave functions are represented as psi*r

    def writexml(self, basis, name=None):
        """
        Writes all basis functions in the given list of basis functions
        to the file "<symbol>.<name>.basis".
        """
        # NOTE: rcs should perhaps be different for orig. and split waves!
        # I.e. have the same shape as basis_lm
        # but right now we just have one rc for each l
        if name is None:
            filename = '%s.basis' % self.generator.symbol
        else:
            filename = '%s.%s.basis' % (self.generator.symbol, name)
        write = open(filename, 'w').write
        write('<paw_basis version="0.1">\n')

        R = self.generator.r
        ng = len(R)
        write(('  <radial_grid eq="r=a*i/(n-i)" a="%f" n="%d" ' +
              'istart="0" iend="%d" id="g1"/>\n') % (self.generator.beta, ng,
                                                     ng-1))
        rc = max([bf.rc for bf in basis])
        # hack since elsewhere multiple rcs are not supported

        for basisfunction in basis:
            psi = basisfunction.psi    
            l = basisfunction.l
            write('  <basis_function l="%d" rc="%f">\n' % (l, rc))
            write('   ')
            for i in range(ng):
                if i == 0: # HACK. Change this.
                    if l == 0:
                        value = psi[1]/R[1]
                    else:
                        value = 0.0
                else:
                    value = psi[i]/R[i]
                write(' %16.12e' % value)
            write('\n')
            write('  </basis_function>\n')
        write('</paw_basis>\n')

    def find_cutoff_by_energy(self, j, esplit=.1, tolerance=.1, rguess=6.):
        """Creates a confinement potential for the orbital given by j,
        such that the confined-orbital energy is (emin to emax) eV larger
        than the free-orbital energy."""
        g = self.generator
        e_base = g.e_j[j]
        rc = rguess
        ri = rc * .6
        vconf = g.get_confinement_potential(AMPLITUDE, ri, rc)

        psi, e = g.solve_confined(j, rc, vconf)
        de_min, de_max = esplit/hartree, (esplit+tolerance)/hartree

        rmin = 0.
        rmax = g.r[-1]
        i_left = g.r2g(rmin)
        i_right = g.r2g(rmax)

        de = e - e_base
        #print '--------'
        #print 'Start bisection'
        #print'e_base =',e_base
        #print 'e =',e
        #print '--------'
        while de < de_min or de > de_max:
            if de < de_min: # Move rc left -> smaller cutoff, higher energy
                rmax = rc
                rc = (rc + rmin) / 2.
            else: # Move rc right
                rmin = rc
                rc = (rc + rmax) / 2.
            ri = rc * .6
            vconf = g.get_confinement_potential(AMPLITUDE, ri, rc)
            psi, e = g.solve_confined(j, rc, vconf)
            de = e - e_base
            #print 'rc = %.03f :: e = %.03f :: de = %.03f' % (rc, e*hartree,
            #                                                 de*hartree)
        #print 'Done!'
        return psi, e, de, vconf, ri, rc

    def generate(self, zetacount=2, polarizationcount=1, tailnorm=.15, 
                 energysplit=.2, tolerance=1.0e-3, referencefile=None, 
                 referenceindex=None, rcutpol_rel=1., txt='-'):
        if txt == '-':
            txt = sys.stdout
        elif txt is None:
            txt = devnull
        # Find out all relevant orbitals
        # We'll probably need: s, p and d.
        # The orbitals we want are stored in u_j.
        # Thus we must find the j corresponding to the highest energy of
        # each orbital-type.
        #
        # ASSUMPTION: The last index of a given value in l_j corresponds
        # exactly to the orbital we want.
        g = self.generator
        print >> txt, 'Basis functions for %s' % g.symbol
        print >> txt, '====================' + '='*len(g.symbol)
        lmax = max(g.l_j)
        lvalues = range(lmax + 1)
        
        reversed_l_j = list(reversed(g.l_j))
        j_l = []
        for l in lvalues:
            j = len(reversed_l_j) - reversed_l_j.index(l) - 1
            j_l.append(j) # index j by l rather than the other way around
        
        singlezetas = []
        doublezetas = []
        other_multizetas = [[] for i in range(zetacount - 2)]
        polarization_functions = []

        for l in lvalues:
            # Get one unmodified pseudo-orbital basis vector for each l
            j = j_l[l]
            n = g.n_j[j]
            print >> txt
            msg = 'Basis functions for l=%d, n=%d' % (l, n)
            print >> txt, msg + '\n', '-'*len(msg)
            print >> txt, 'Zeta 1: softly confined pseudo wave,',
            print >> txt, 'fixed energy shift'
            u, e, de, vconf, ri, rc = self.find_cutoff_by_energy(j,
                                                                 energysplit,
                                                                 tolerance)
            print >> txt, 'DE=%.03f eV :: rc=%.02f Bohr' % (de * hartree, rc)
            s = self.smoothify(u, l)
            bf = BasisFunction(s, rc, n, l, j, 1, None)
            singlezetas.append(bf)

            if zetacount > 1:
                # add one split-valence vector using fixed-energy-shift-scheme
                print >> txt, '\nZeta 2: split-valence wave, fixed tail norm'
                norm = npy.dot(g.dr, u*u)
                partial_norm = 0.
                i = len(u) - 1
                while partial_norm / norm < tailnorm:
                    # Integrate backwards.  This is important since the pseudo
                    # wave functions have strange behaviour near the core.
                    partial_norm += g.dr[i] * u[i]**2
                    i -= 1
                rsplit = g.r[i+1]
                msg = 'Tail norm %.03f :: rsplit=%.02f Bohr' % (partial_norm,
                                                                rsplit)
                print >> txt, msg
                splitwave = self.make_split_valence_vector(s, l, rsplit)
                bf_dz = BasisFunction(s - splitwave, rsplit, g.n_j[j], l, j,
                                      2, None)
                doublezetas.append(bf_dz)

                # If there are even more zetas, make new, smaller split radii
                # We'll just distribute them evenly between 0 and rsplit
                extra_split_radii = npy.linspace(rsplit, 0., zetacount)[1:-1]
                for i, rsplit in enumerate(extra_split_radii):
                    print >> txt, '\nZeta %d: extra split-valence wave' % (3+i)
                    print >> txt, 'rsplit=%.02f Bohr' % rsplit
                    splitwave = self.make_split_valence_vector(s, l, rsplit)
                    bf_multizeta = BasisFunction(s - splitwave, rsplit, 
                                                 g.n_j[j], l, j, 2+i, None)
                    other_multizetas[i].append(bf_multizeta)
                    
        if polarizationcount > 0:
            # Now make up some properties for the polarization orbital
            # We just use the cutoffs from the previous one times a factor
            rcut = singlezetas[-1].rc * rcutpol_rel
            l_pol = lmax + 1
            msg = 'Polarization function: l=%d, rc=%.02f' % (l_pol, rcut)
            print >> txt, '\n' + msg
            print >> txt, '-' * len(msg)
            psi_pol = self.make_polarization_function(rcut, l_pol, 
                                                      referencefile,
                                                      referenceindex,
                                                      txt)
            bf_pol = BasisFunction(psi_pol, rcut, None, l_pol, None, None, 1)
            polarization_functions.append(bf_pol)
            if polarizationcount > 1:
                msg = 'Warning: generating multiple polarization functions'
                msg += ', this doesn\'t work properly yet'
                raise NotImplementedError(msg)
                # make evenly distributed split-radii for remaining functions
                rsplits = npy.linspace(rcut, 0., polarizationcount+1)[1:-1]
                for i, rsplit in enumerate(rsplits):
                    splitwave = self.make_split_valence_vector(psi_pol, l, 
                                                               rsplit)
                    bf_pol_split = BasisFunction(psi_pol - splitwave, rsplit,
                                                 None, l_pol, None, i)
                    polarization_functions.append(bf_pol_split)
        print >> txt
        basis = []
        basis.extend(singlezetas)
        basis.extend(doublezetas)
        for multizetas in other_multizetas:
            basis.extend(multizetas)
        basis.extend(polarization_functions)
        return basis

    def plot(self, basis, figure=None, show=False, title=None):
        import pylab
        g = self.generator
        if figure is not None:
            pylab.figure(figure)
        else:
            pylab.figure() # not very elegant
        if title is None:
            title = g.symbol
        pylab.title(title)
        for bf in basis:
            label = 'n=%s, l=%s, rc=%.02f' % (str(bf.n), str(bf.l), bf.rc)
            pylab.plot(g.r, bf.psi, label=label)
        pylab.legend()
