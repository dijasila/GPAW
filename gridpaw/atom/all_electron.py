# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Atomic Density Functional Theory
"""

from math import pi, sqrt, log
import pickle
import sys

import Numeric as num
import LinearAlgebra as linalg
from ASE.ChemicalElements.name import names

from gridpaw.atom.configurations import configurations
from gridpaw.grid_descriptor import RadialGridDescriptor
from gridpaw.xc_functional import XCOperator, XCFunctional
from gridpaw.utilities import hartree

# fine-structure constant
alpha = 1 / 137.036

class AllElectron:
    """Object for doing an atomic DFT calculation."""

    def __init__(self, symbol, xcname='LDA', scalarrel=False):
        """Do an atomic DFT calculation.
        
        Example:

        a = AllElectron('Fe')
        a.Run()
        """

        self.symbol = symbol
        self.xcname = xcname
        self.scalarrel = scalarrel

        # Get reference state:
        self.Z, nlfe_j = configurations[symbol]

        # Collect principal quantum numbers, angular momentum quantum
        # numbers, occupation numbers and eigenvalues (j is a combined
        # index for n and l):        
        self.n_j = num.array([n for n, l, f, e in nlfe_j])
        self.l_j = num.array([l for n, l, f, e in nlfe_j])
        self.f_j = num.array([f for n, l, f, e in nlfe_j])
        self.e_j = num.array([e for n, l, f, e in nlfe_j])

        print
        if scalarrel:
            print 'Scalar-relativistic atomic',
        else:
            print 'Atomic',
        print '%s calculation for %s (%s, Z=%d)' % (
            xcname, symbol, names[self.Z], self.Z)

        # Number of orbitals:
        nj = len(nlfe_j)
        
        #     beta g
        # r = ------, g = 0, 1, ..., N - 1
        #     N - g
        #
        #        rN
        # g = --------
        #     beta + r
        maxnodes = max(self.n_j - self.l_j - 1)
        N = (maxnodes + 1) * 150
        print N, 'radial gridpoints.'
        beta = 0.4
        g = num.arange(N, typecode=num.Float)
        self.r = beta * g / (N - g)
        self.dr = beta * N / (N - g)**2
        self.rgd = RadialGridDescriptor(self.r, self.dr)
        self.d2gdr2 = -2 * N * beta / (beta + self.r)**3
        self.N = N
        self.beta = beta

        # Radial wave functions multiplied by radius:
        self.u_j = num.zeros((nj, N), num.Float)

        # Effective potential multiplied by radius:
        self.vr = num.zeros(N, num.Float)

        # Electron density:
        self.n = num.zeros(N, num.Float)

        self.xc = XCOperator(XCFunctional(xcname, scalarrel), self.rgd)

    def intialize_wave_functions(self):
        r = self.r
        dr = self.dr
        # Initialize with Slater function:
        for l, e, u in zip(self.l_j, self.e_j, self.u_j):
            a = sqrt(-2.0 * e)

            # This one: "u[:] = r**(1 + l) * num.exp(-a * r)" gives
            # OverflowError: math range error XXX
            u[:] = r**(1 + l)
            rmax = 350.0 / a
            gmax = int(rmax * self.N / (self.beta + rmax))
            u[:gmax] *= num.exp(-a * r[:gmax])
            u[gmax:] = 0.0

            norm = num.dot(u**2, dr)
            u *= 1.0 / sqrt(norm)
        
    def run(self):
        Z = self.Z
        r = self.r
        dr = self.dr
        n = self.n
        vr = self.vr

        vHr = num.zeros(self.N, num.Float)
        vXC = num.zeros(self.N, num.Float)

        n_j = self.n_j
        l_j = self.l_j
        f_j = self.f_j
        e_j = self.e_j

        try:
            f = open(self.symbol + '.restart', 'r')
        except IOError:
            self.intialize_wave_functions()
            n[:] = self.calculate_density()
        else:
            print 'Using old density for initial guess.'
            n[:] = pickle.load(f)
            n *= Z / (num.dot(n * r**2, dr) * 4 * pi)

        bar = '|------------------------------------------------|'
        print bar
        niter = 0
        qOK = log(1e-10)
        while True:

	    if 1:#JUSSI ###DEBUGG
#      	    #KIRJOITETAAN grid TIEDOSTOON ###DEBUGG
                file = open('./RADIALgrid', 'w') ###DEBUGG
                file.write(str(self.r)) ###DEBUGG
                file.close() ###DEBUGG

            # calculate hartree potential
            hartree(0, n * r * dr, self.beta, self.N, vHr)

            # add potential from nuclear point charge (v = -Z / r)
            vHr -= Z

            if 1:#JUSSI ###DEBUGG
#           #KIRJOITETAAN HARTREE-POTENTIAALI TIEDOSTOON ###DEBUGG
                file = open('./HARTREEpotential', 'w') ###DEBUGG
                file.write(str(vHr)) ###DEBUGG
                file.close() ###DEBUGG

            # calculated exchange correlation potential and energy
            vXC[:] = 0.0
            Exc = self.xc.get_energy_and_potential(n, vXC)

            if 1:#JUSSI ###DEBUGG
#           #KIRJOITETAAN XC-POTENTIAALI TIEDOSTOON ###DEBUGG
                file = open('./XCpotential', 'w') ###DEBUGG
                file.write(str(vXC)) ###DEBUGG
                file.close() ###DEBUGG

	    vXC_KLI = calculate_1D_KLI_potential(self)#JUSSI
            if 1:#JUSSI ###DEBUGG
#           #KIRJOITETAAN KLI-POTENTIAALI TIEDOSTOON ###DEBUGG
                file = open('./KLIpotential', 'w') ###DEBUGG
                file.write(str(vXC_KLI)) ###DEBUGG
                file.close() ###DEBUGG

            if 0:#JUSSI ###DEBUGG
#           #KIRJOITETAAN tutkailtava TIEDOSTOON ###DEBUGG
                file = open('./tutkailtavaDEBUGG', 'w') ###DEBUGG
                file.write(str(vHr+vXC_KLI)) ###DEBUGG
#                file.write(str(vHr/vXC_KLI)) ###DEBUGG
                file.close() ###DEBUGG

            # calculate new total Kohn-Sham effective potential and
            # admix with old version
            vr[:] = vHr + vXC * r           #non-self-consistent KLI #(JUSSI)
#            vr[:] = vHr + 1 * vXC_KLI * r   #self-consistent KLI #JUSSI
            if niter > 0:
		feedback = 0.4 #JUSSI
                vr[:] = feedback * vr + (1-feedback) * vrold #JUSSI
#                vr[:] = 0.4 * vr + 0.6 * vrold
            vrold = vr.copy()

            # solve Kohn-Sham equation and determine the density change
            self.solve()
            dn = self.calculate_density() - n
            n += dn

            # estimate error from the square of the density change integrated
            q = log(num.sum((r * dn)**2))

            # print progress bar
            if niter == 0:
                q0 = q
                b0 = 0
            else:
                b = int((q0 - q) / (q0 - qOK) * 50)
                if b > b0:
                    sys.stdout.write(bar[b0:min(b, 50)])
                    sys.stdout.flush()
                    b0 = b

            # check if converged and break loop if so
            if q < qOK:
                sys.stdout.write(bar[b0:])
                sys.stdout.flush()
                break
            
            niter += 1
            if niter > 117:
                raise RuntimeError, 'Did not converge!'
            
        print
        print 'Converged in %d iteration%s.' % (niter, 's'[:niter != 1])
        
        pickle.dump(n, open(self.symbol + '.restart', 'w'))

        Epot = 2 * pi * num.dot(n * r * (vHr - Z), dr)
        Ekin = num.dot(f_j, e_j) - 4 * pi * num.dot(n * vr * r, dr)

        print
        print 'Energy contributions:'
        print '-------------------------'
        print 'Kinetic:   %+13.6f' % Ekin
        print 'XC:        %+13.6f' % Exc
        print 'Potential: %+13.6f' % Epot
        print '-------------------------'
        print 'Total:     %+13.6f' % (Ekin + Exc + Epot)
        print

        print 'state    eigenvalue         ekin         rmax'
        print '---------------------------------------------'
        for m, l, f, e, u in zip(n_j, l_j, f_j, e_j, self.u_j):
            # Find kinetic energy:
            k = e - num.sum((num.where(abs(u) < 1e-160, 0, u)**2 * #XXXNumeric!
                             vr * dr)[1:] / r[1:])
            
            # Find outermost maximum:
            g = self.N - 4
            while u[g - 1] > u[g]:
                g -= 1
            x = r[g - 1:g + 2]
            y = u[g - 1:g + 2]
            A = num.transpose(num.array([x**i for i in range(3)]))
            c, b, a = linalg.solve_linear_equations(A, y)
            assert a < 0.0
            rmax = -0.5 * b / a
            
            t = 'spdf'[l]
            print '%d%s^%-2d: %12.6f %12.6f %12.3f' % (m, t, f, e, k, rmax)
        print '---------------------------------------------'
        print '(units: Bohr and Hartree)'
        
        for m, l, u in zip(n_j, l_j, self.u_j):
            self.write(u, 'ae', n=m, l=l)
            
        self.write(n, 'n')
        self.write(vr, 'vr')
        self.write(vHr, 'vHr')
        self.write(vXC, 'vXC')
        
        self.Ekin = Ekin
        self.Epot = Epot
        self.Exc = Exc

    def write(self, array, name=None, n=None, l=None):
        if name:
            name = self.symbol + '.' + name
        else:
            name = self.symbol
            
        if l is not None:
            assert n is not None
            name += '.%d%s' % (n, 'spdf'[l])
                
        f = open(name, 'w')
        for r, a in zip(self.r, array):
            print >> f, r, a
    
    def calculate_density(self):
        """Return the electron charge density divided by 4 pi"""
        n = num.dot(self.f_j,
                    num.where(abs(self.u_j) < 1e-160, 0,
                              self.u_j)**2) / (4 * pi)
        n[1:] /= self.r[1:]**2
        n[0] = n[1]
        return n
    
    def solve(self):
        """
        Solve the Schrodinger equation::
        
             2 
            d u     1  dv  du   u     l(l + 1)
          - --- - ---- -- (-- - -) + [-------- + 2M(v - e)] u = 0
              2      2 dr  dr   r         2
            dr    2Mc                    r

        
        where the relativistic mass::

                   1
          M = 1 - --- (v - e)
                    2
                  2c
        
        and the fine-structure constant alpha = 1/c = 1/137.036
        is set to zero for non-scalar-relativistic calculations.

        On the logaritmic radial grids defined by::

              beta g
          r = ------,  g = 0, 1, ..., N - 1
              N - g

                 rN
          g = --------, r = [0; oo[
              beta + r
  
        the Schrodinger equation becomes::
        
           2 
          d u      du  
          --- c  + -- c  + u c  = 0
            2  2   dg  1      0
          dg

        with the vectors c , c , and c  defined by::
                          0   1       2
                 2 dg 2
          c  = -r (--)
           2       dr
        
                  2         2
                 d g  2    r   dg dv
          c  = - --- r  - ---- -- --
           1       2         2 dr dr
                 dr       2Mc
          
                                    2    r   dv
          c  = l(l + 1) + 2M(v - e)r  + ---- --
           0                               2 dr
                                        2Mc
        """
        r = self.r
        dr = self.dr
        vr = self.vr
        
        c2 = -(r / dr)**2
        c10 = -self.d2gdr2 * r**2 # first part of c1 vector
        
        if self.scalarrel:
            self.r2dvdr = num.zeros(self.N, num.Float)
            self.rgd.derivative(vr, self.r2dvdr)
            self.r2dvdr *= r
            self.r2dvdr -= vr
        else:
            self.r2dvdr = None
            
        # solve for each quantum state separately
        for j, (n, l, e, u) in enumerate(zip(self.n_j, self.l_j,
                                             self.e_j, self.u_j)):
            nodes = n - l - 1 # analytically expected number of nodes
            delta = -0.2 * e
            nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                          self.scalarrel)
            # adjust eigenenergy until u has the correct number of nodes
            while nn != nodes:
                diff = cmp(nn, nodes)
                while diff == cmp(nn, nodes):
                    e -= diff * delta
                    nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                                  self.scalarrel)
                delta /= 2

            # adjust eigenenergy until u is smooth at the turning point
            de = 1.0
            while abs(de) > 1e-9:
                norm = num.dot(num.where(abs(u) < 1e-160, 0, u)**2, dr)
                u *= 1.0 / sqrt(norm)
                de = 0.5 * A / norm
                x = abs(de / e)
                if x > 0.1:
                    de *= 0.1 / x
                e -= de
                assert e < 0.0
                nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                              self.scalarrel)
            self.e_j[j] = e
            u *= 1.0 / sqrt(num.dot(num.where(abs(u) < 1e-160, 0, u)**2, dr))

    def kin(self, l, u, e=None): # XXX move to Generator
        r = self.r[1:]
        dr = self.dr[1:]
        
        c0 = 0.5 * l * (l + 1) / r**2
        c1 = -0.5 * self.d2gdr2[1:]
        c2 = -0.5 * dr**-2
        
        if e is not None and self.scalarrel:
            x = 0.5 * alpha**2
            Mr = r * (1.0 + x * e) - x * self.vr[1:]
            c0 += ((Mr - r) * (self.vr[1:] - e * r) +
                   0.5 * x * self.r2dvdr[1:] / Mr) / r**2
            c1 -= 0.5 * x * self.r2dvdr[1:] / (Mr * dr * r)

        fp = c2 + 0.5 * c1
        fm = c2 - 0.5 * c1
        f0 = c0 - 2 * c2
        kr = num.zeros(self.N, num.Float)
        kr[1:] = f0 * u[1:] + fm * u[:-1]
        kr[1:-1] += fp[:-1] * u[2:]
        kr[0] = 0.0
        return kr    

def shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel=False, gmax=None):
    """n, A = shoot(u, l, vr, e, ...)
       For guessed trial eigenenergy e, integrate the radial Schrodinger
       equation::
          2 
         d u      du  
         --- c  + -- c  + u c  = 0
           2  2   dg  1      0
         dg
        
               2 dg 2
        c  = -r (--)
         2       dr
        
                2         2
               d g  2    r   dg dv
        c  = - --- r  - ---- -- --
         1       2         2 dr dr
               dr       2Mc
        
                                  2    r   dv
        c  = l(l + 1) + 2M(v - e)r  + ---- --
         0                               2 dr
                                      2Mc
       The resulting wavefunction is returned in input vector u.
       The number of nodes of u is returned in attribute n.
       Returned attribute A, is a measure of the size of the derivative
       discontinuity at the classical turning point.
       The trial energy e is correct if A is zero and n is the correct number
       of nodes.
    """
    if scalarrel:
        x = 0.5 * alpha**2 # x = 1 / (2c^2)
        Mr = r * (1.0 + x * e) - x * vr
    else:
        Mr = r
    c0 = l * (l + 1) + 2 * Mr * (vr - e * r)
    if gmax is None and num.alltrue(c0 > 0):
        print """
Problem with initial electron density guess!  Try to run the program
with the '-n' option (non-scalar-relativistic calculation) and then
try again without the '-n' option (this will generate a good initial
guess for the density).
"""
        raise SystemExit
    c1 = c10
    if scalarrel:
        c0 += x * r2dvdr / Mr
        c1 = c10 - x * r * r2dvdr / (Mr * dr)

    # vectors needed for numeric integration of diff. equation
    fm = 0.5 * c1 - c2
    fp = 0.5 * c1 + c2
    f0 = c0 - 2 * c2
    
    if gmax is None:
        # set boundary conditions at r -> oo (u(oo) = 0 is implicit)
        u[-1] = 1.0
        
        # perform backwards integration from infinity to the turning point
        g = len(u) - 2
        u[-2] = u[-1] * f0[-1] / fm[-1]
        while c0[g] > 0.0: # this defines the classical turning point
            u[g - 1] = (f0[g] * u[g] + fp[g] * u[g + 1]) / fm[g]
            if u[g - 1] < 0.0:
                # There should't be a node here!  Use a more negative
                # eigenvalue:
                print '!!!!!!',
                return 100, None
            if u[g - 1] > 1e100:
                u *= 1e-100
            g -= 1

        # stored values of the wavefunction and the first derivative
        # at the turning point
        gtp = g + 1 
        utp = u[gtp]
        dudrplus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]
    else:
        gtp = gmax

    # set boundary conditions at r -> 0
    u[0] = 0.0
    u[1] = 1.0
    
    # perform forward integration from zero to the turning point
    g = 1
    nodes = 0
    while g <= gtp: # integrate one step further than gtp
                    # (such that dudr is defined in gtp)
        u[g + 1] = (fm[g] * u[g - 1] - f0[g] * u[g]) / fp[g]
        if u[g + 1] * u[g] < 0:
            nodes += 1
        g += 1
    if gmax is not None:
        return

    # scale first part of wavefunction, such that it is continuous at gtp
    u[:gtp + 2] *= utp / u[gtp]

    # determine size of the derivative discontinuity at gtp
    dudrminus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]
    A = (dudrplus - dudrminus) * utp
    
    return nodes, A
            
if __name__ == '__main__':
    a = AllElectron('Cu', scalarrel=True)
    a.run()










def calculate_1D_KLI_potential(self): #JUSSI

	pii=3.141592654
        sopivan_pieni_luku = 1e-30 # avoidZeroDivision

        print '---------- ---------- ---------- ---------- ---------- vvvvvvvvvv'
        #Grid

        r = self.r

        #Occupied state radial distributions : ru_j.
        ru_j = self.u_j

        #Occupied state wf:s : u_j.
#        u_j = ru_j / (r+sopivan_pieni_luku) #first try, most incorrect at zero
        u_j = num.zeros((ru_j.shape[0], ru_j.shape[1]), num.Float)
	for j in range(ru_j.shape[0]):
            u_j[j,1:] = ru_j[j,1:] / r[1:]
	    #linear extrapolation to zero
	    k = (u_j[j,2]-u_j[j,1]) / (r[2]-r[1])
	    u_j[j,0] = u_j[j,1] + k * (r[0]-r[1])



	#Occupations : f_j
        f_j = self.f_j /2 #one spin olnly
	print 'f_j = ', f_j # DEBUGG



        #Square occupied state wf:s : |u_j|^2.
        u_j_abs2 = num.conjugate(u_j) * u_j

        #Number of occupied orbitals.
        HOMO = u_j.shape[0]
        print 'HOMO=', HOMO ###DEBUGG



        #Calculate matrix elements A_ji. (riveittain)
        #
        #                     (Dimensions must not include 
        #      i=0,1..HOMO-2   each degenerate state separately!!!)
        #      _____________   
        #     |             |    0
        #     |             |    1
        # A = |             | j= .
        #     |             |    .
        #     |_____________|    HOMO-2
        #
        A = num.zeros((HOMO-1,HOMO-1),num.Float)

        #Check for and avoid zeros in the denominator.
        avoidZeroDevision_n = num.zeros(self.N,num.Float)
        for i in range(self.N):
                if self.n[i] == 0.0:
                        #Replace a zero in the denominator by a small number
                        #(of which sign? does it matter?).
                        avoidZeroDevision_n[i] = sopivan_pieni_luku
                        #MIKA ON SOPIVA 'pieni' LUKU??? eps@python? ###DEBUGG
                else:
                        avoidZeroDevision_n[i] = self.n[i]

        #Do the actual matrix element calculating.
        for j in range(HOMO-1):
                for i in range(HOMO-1):
                        if i == j:
                                A[j,i] = 1.0 - \
                                integroi_pallokoordinaatistossa( f_j[j]*u_j_abs2[j,:]/(4*pii)*\
								 f_j[i]*u_j_abs2[i,:]/(4*pii) \
								 / avoidZeroDevision_n/2, r)
                        else:
                                A[j,i] = 0.0 - \
                                integroi_pallokoordinaatistossa( f_j[j]*u_j_abs2[j,:]/(4*pii)*\
								 f_j[i]*u_j_abs2[i,:]/(4*pii) \
								 / avoidZeroDevision_n/2, r)
        print 'A = ' ###DEBUGG
        print A ###DEBUGG



        #ALA VAPAUTA MUUTTUJALLE avoidZeroDevision_n VARATTUA MUISTIA! ###DEBUGG
        #SITA TARVITAAN VIELA! ###DEBUGG





        #Use Poisson solver to calculate 
        #int(dr')[{ num.conjugate(u_j[i,:,:,:]) * u_j[k,:,:,:] } / { |r-r'| }] 
        #for each pair (i,k).
        #Allocate memory to store the results.
        poisson_ik = num.zeros((HOMO,HOMO,
                                self.N), 
                               num.Float)
#       #POISSON-RATKAISIJAA KAYTETAAN NAIN! ###DEBUGG
        #vHr = self.calculate_hartree_potential(n) - Z	(GPAW4, reference? vHr(infty)=0)
        #vHr = self.calculate_hartree_potential(n)	(GPAW3, reference? vHr(0)=0)
        #hartree(0, n * r * dr, self.beta, self.N, vHr)	(GPAW5, 1/2)
        #vHr -= Z					(GPAW5, 2/2)

        #SHOULD BE IMPROVED !!!!!
        #       by calculating only poisson_ik[i,k,:] and
        #       conjugating to get poisson_ik[k,i,:]=num.conjugate(poisson_ik[i,k,:])

	Z=self.Z
        print 'Z = ', Z ###DEBUGG

        for i in range(HOMO):
                for k in range(HOMO):
			peitto=integroi_pallokoordinaatistossa(num.conjugate(u_j[i,:])*u_j[k,:]\
							       /(4*pii), r)
#			print 'peitto = ', peitto # DEBUGG
			POISSONINPUT = 	sqrt(f_j[i]) * sqrt(f_j[k]) *\
					num.conjugate(u_j[i,:]) * u_j[k,:]\
					/(4*pii)

#                        poisson_ik[i,k,:] = \
#                           (\
#                           self.calculate_hartree_potential(POISSONINPUT) \
#                                - sqrt(f_j[i])*sqrt(f_j[k])*peitto \
#                           )#/(-4*pii) # -1/4pii HATUSTA!!!???

			hartree(0, POISSONINPUT * self.r * self.dr, self.beta, self.N, poisson_ik[i,k,:])
			poisson_ik[i,k,:] -= sqrt(f_j[i])*sqrt(f_j[k])*peitto


        #Calculate u_xi(r) for occupied orbitals i.
        #Allocate memory to store the u_xi(r).
        u_xi = num.zeros((HOMO,
                          self.N), 
                         num.Float)
        poissonsum = num.zeros(self.N, 
                               num.Float)

        #But first, again, check for and avoid zeros in the denominator.
        avoidZeroDevision_u_j = num.zeros((u_j.shape[0],
                                           self.N),
                                          num.Float)
        for l in range(u_j.shape[0]):
                for i in range(self.N):
                        if u_j[l,i] == 0.0:
                                #Replace a zero in the denominator by a small number
                                #(of which sign? does it matter?).
                                avoidZeroDevision_u_j[l,i] = sopivan_pieni_luku 
                                #MIKA ON SOPIVA 'pieni' LUKU??? eps@python? ###DEBUGG
                        else:
                                avoidZeroDevision_u_j[l,i] = u_j[l,i]

        #Then, do the actual calculating of u_xi.
        for i in range(HOMO):
                for k in range(HOMO):
                        poissonsum += sqrt(f_j[k])*u_j[k,:] * poisson_ik[i,k,:]
                u_xi[i,:] = - poissonsum / \
			    ( num.conjugate(sqrt(f_j[i])*avoidZeroDevision_u_j[i,:]) )

        #PITAISI VAPAUTTAA MUUTTUJALLE avoidZeroDevision_u_j VARATTU MUISTI! ###DEBUGG

        #Calculate ubar_xi(r) for occupied orbitals i.
        #Allocate memory to store the ubar_xi.
        ubar_xi = num.zeros((HOMO),num.Float)
        for i in range(HOMO):
                ubar_xi[i] = integroi_pallokoordinaatistossa(f_j[i]*u_j_abs2[i,:]/(4*pii) * \
							     u_xi[i,:], r)



        #Calculate vector elements b for orbitals i=0,1,....,HOMO-2.
        #Allocate memory to store b.
        b = num.zeros((HOMO-1),num.Float)

        for j in range(HOMO-1):
                tmpsum = num.zeros(self.N,num.Float)
                for k in range(HOMO):
                        tmp = f_j[k]*u_j_abs2[k,:] * \
                              (1./2) * ( u_xi[k,:] + num.conjugate(u_xi[k,:]) )
                        tmpsum += tmp
#                       print 'tmpsum+ = ', tmp ###DEBUGG
#               print 'tmpsum = ', tmpsum ###DEBUGG

                b[j] = \
		       integroi_pallokoordinaatistossa(f_j[j]*u_j_abs2[j,:]/(4*pii) * tmpsum \
						       / avoidZeroDevision_n/2, r)



#               print 'b[', j, '] = ', b[j] ###DEBUGG
	print 'b = ', b ###DEBUGG
        print '---------- ---------- ---------- ---------- ---------- ^^^^^^^^^^'



        #Solve A * x = b for x.
        Ainv = linalg.inverse(A)
        x = num.zeros((HOMO-1),num.Float)
        #Calculate x = Ainv * b.
        for k in range(HOMO-1):
                x[k] = num.vdot(Ainv[k,:],b)



        #Solve Vbar_xi for orbitals i=0,1,...,HOMO-2 and 
        #calculate vXC_KLI.
        Vbar_xi = num.zeros((HOMO),num.Float)
        for i in range(HOMO-1):
                Vbar_xi[i] = x[i] + (1./2) * ( ubar_xi[i] + num.conjugate(ubar_xi[i]) )
	Vbar_xi[HOMO-1] = ubar_xi[HOMO-1]#In order to guarantee correct long range behaviour
					 #KLI:PRA,45,101,Eq.(47)



        #Allocate memory for vXC_KLI.
        vXC_KLI = num.zeros(self.N,
                            num.Float)
        tmpsum = num.zeros(self.N,
                           num.Float)
        for i in range(HOMO):
                tmpsum += f_j[i]*u_j_abs2[i,:] * \
                          ( u_xi[i,:] + ( Vbar_xi[i] - ubar_xi[i] ) + \
                            num.conjugate( u_xi[i,:] + ( Vbar_xi[i] - ubar_xi[i] ) ) )

        vXC_KLI = tmpsum / ( 2 * avoidZeroDevision_n/2 ) /(-4*pii) # -1/4pii HATUSTA!!!



	if 0:
		#Testataan poisson-ratkaisijaa.
		print 'TESTING POISSON starts!!!'
		syote = num.zeros((self.N), num.Float)
		r_cut = 0.02 * r[len(r)-1]
		r_cut_index = 0
		for i in range(syote.shape[0]):
		    if r[i]<r_cut:
		        syote[i] = 0.01
			r_cut_index += 1
		    else:
			syote[i] = 0

		print 'r_cut (viimeinen   ~= 0) = ', r[r_cut_index-1]
		print 'r_cut (ensimmainen == 0) = ', r[r_cut_index]
		nollakohtaARVAUS = integroi_pallokoordinaatistossa(syote, r)
		print 'nollakohtaARVAUS = ', nollakohtaARVAUS

		kertyma = num.zeros((len(r)), num.Float)
		for i in range(len(r)):
			kertyma[i] = integroi_pallokoordinaatistossa(syote[0:i+1], r[0:i+1])
		kertymaPERr2 = num.zeros((len(r)), num.Float)
		for i in range(len(r)-1):
			kertymaPERr2[i+1] = kertyma[i+1]/r[i+1]/r[i+1]
		#linear extrapolation to zero
		k = (kertymaPERr2[2]-kertymaPERr2[1]) / (r[2]-r[1])
		kertymaPERr2[0] = kertymaPERr2[1] + k * (r[0]-r[1])
		nollakohta = integroi_radiaalisesti(kertymaPERr2,r)
		print 'nollakohta = ', nollakohta
		
		poisson_test = num.zeros((self.N), num.Float)
		poisson_test[:] = self.calculate_hartree_potential(syote) - nollakohtaARVAUS

		print 'vakiotiheys = ', syote[0]
		print 'pallon sade = ', r[r_cut_index-1], '...', r[r_cut_index]
		print 'kokonaissyote = ', \
			4*pii*r[r_cut_index-1]*r[r_cut_index-1]*r[r_cut_index-1]/3*syote[0], \
		      '...', \
			4*pii*r[r_cut_index]*r[r_cut_index]*r[r_cut_index]/3*syote[0]
		print 'kokonaissyote (INT) = ', integroi_pallokoordinaatistossa(syote,r)
		print 'nollakohdan siirto = ', poisson_test[len(poisson_test)-1]-poisson_test[0]
		print 'ehdotus siirroksi = ', 2*pii*syote[0]*r[r_cut_index-1]*r[r_cut_index-1], \
		      '...', 2*pii*syote[0]*r[r_cut_index]*r[r_cut_index]
		print 'TESTING POISSON ends!!!!!'

        if 1:#JUSSI ###DEBUGG
	    nro=0
#       #KIRJOITETAAN tutkailtava TIEDOSTOON ###DEBUGG
            file = open('./tutkailtavaDEBUGG', 'w') ###DEBUGG
#            file.write(str(ru_j[nro,:])) ###DEBUGG
#            file.write(str(u_j[nro,:])) ###DEBUGG
#            file.write(str(u_j_abs2[nro,:])) ###DEBUGG
            file.write(str(self.n[:])) ###DEBUGG
#            file.write(str(poisson_ik[0,0,:])) ###DEBUGG
#            file.write(str(u_xi[nro,:])) ###DEBUGG
#            file.write(str(u_j_abs2[nro,:]*u_xi[nro,:])) ###DEBUGG
#            file.write(str(u_j_abs2[nro,:]*(u_xi[nro,:]+(Vbar_xi[nro]-ubar_xi[nro])))) ###DEBUGG
#            file.write(str(syote[:])) ###DEBUGG
#            file.write(str(kertyma[:])) ###DEBUGG
#            file.write(str(kertymaPERr2[:])) ###DEBUGG
#            file.write(str(poisson_test[:])) ###DEBUGG
            file.close() ###DEBUGG

	print 'A = ', A
	print 'ubar_xi = ', ubar_xi
	print 'b = ', b
	print 'x = ', x
	print 'Vbar_xi = ', Vbar_xi
	print 'Vbar_xi-ubar_xi = ', Vbar_xi-ubar_xi
	print 'N = integroi_pallokoordinaatistossa(self.n,r) = ', integroi_pallokoordinaatistossa(self.n,r)

        return vXC_KLI





def integroi_pallokoordinaatistossa(integrandi,radiaalinen_gridi): #JUSSI

	pii=3.141592654

	delta_r = num.zeros(len(radiaalinen_gridi), num.Float) #allocate

        delta_r[0:len(delta_r)-1] = \
		radiaalinen_gridi[1:len(radiaalinen_gridi)] \
		- radiaalinen_gridi[0:len(radiaalinen_gridi)-1] #Mika on 
								#viimeinen alkio?
        delta_r[len(delta_r)-1]=delta_r[len(delta_r)-2] #Olkoon se viimeinen alkio 
							#paremman puutteessa 
                                                        #sama kuin 2. viimeinen.
	integraali = 4 * pii * \
                     num.vdot(radiaalinen_gridi*radiaalinen_gridi, integrandi * delta_r)
	return integraali





def integroi_radiaalisesti(integrandi,radiaalinen_gridi): #JUSSI

	delta_r = num.zeros(len(radiaalinen_gridi), num.Float) #allocate

        delta_r[0:len(delta_r)-1] = \
		radiaalinen_gridi[1:len(radiaalinen_gridi)] \
		- radiaalinen_gridi[0:len(radiaalinen_gridi)-1] #Mika on 
								#viimeinen alkio?
        delta_r[len(delta_r)-1]=delta_r[len(delta_r)-2] #Olkoon se viimeinen alkio 
							#paremman puutteessa 
                                                        #sama kuin 2. viimeinen.
	integraali = num.vdot(integrandi, delta_r)
	return integraali
