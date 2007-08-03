
import Numeric as num
import gpaw.mpi as mpi
import gpaw.tddft.BiCGStab as BiCGStab
from gpaw.utilities.cg import CG
import math

class RecursionMethod:
    """ this class sets up the Haydock recursion method for the S-1H case

    class attributes:
        self.u_init     - the initial wave function
        self.u_lat_l    - the latest left vector in the lanczos basis
        self.u_lat_r    - the latest right vector in the lanczos basis
        self.u_nlat_l   - the second latest left vector in the lanczos basis
        self.u_nlat_r   - the second latest right vector in the lanczos basis
        self.b          - array of recursion coefficients, b 
        self.a          - array of recursion coefficients, a
        
        self.max_previous_iter  - a counter that keeps track of the number of lanczos vectors 
        self.paw                - ingoing paw object
        self.kpt                - ingoing k-point that is to be calculated 
        self.max_iter           - the maximum allowed iterations 
        self.solver             - solver for Sx=y equation

    """


    def __init__(self, phi_t_in, calc, kpoint = 0, tol = 1e-10, max_iter = 1000):
        """
        =========== ===================================
        Parameters:
        =========== ===================================
        phi_t_in    the starting vector
        paw         the paw object
        kpoint      the k-point that is to be calculated
        tol         tolerance for terminating the continued fraction
        max_iter    maximum number of iterations
        =========== ===================================
        """

        #self.max_previous_iter = 1
        self.calc = calc
        self.gd = calc.gd
        self.kpt = calc.kpt_u[kpoint]
        self.nuclei = calc.nuclei
        self.max_iter = max_iter
        self.typecode = calc.typecode
        self.u_init =  phi_t_in
        self.tol = tol 
        #print num.sum(self.u_init.flat)
        self.u_lat_l = calc.gd.zeros(3) 
        self.u_lat_r = calc.gd.zeros(3)
        self.x_tmp = calc.gd.zeros(3) #num.zeros(phi_t_in.shape, self.typecode)

        #normalize initial vector for method 1
        if 0:
            for i, u_i in enumerate(self.u_init):
                self.u_lat_l[i] = u_i / math.sqrt(self.gd.integrate(u_i**2))
                self.u_lat_r[i] = u_i / math.sqrt(self.gd.integrate(u_i**2))
                # should be complex conjugate for complex wf
        else:
            u_tmp = self.gd.zeros(3)
            self.solve(u_tmp, self.u_init)
            i=0
            for u_i,u_t in zip(self.u_init, u_tmp):
                self.u_lat_r[i] = u_i / math.sqrt(self.gd.integrate(u_i * u_t))
                i += 1
                
        self.u_nlat_l =  calc.gd.zeros(3) #num.zeros(phi_t_in.shape, self.typecode)
        self.u_nlat_r =  calc.gd.zeros(3) #num.zeros(phi_t_in.shape, self.typecode)
        self.b = []
        self.b.append([0,0,0])
        self.a =[]
        self.y_lat = calc.gd.zeros(3)


        
    
        
        # for i in range(5):
        #     self.compute_lanczos_vectors2(i)
        
    def compute_spectrum(self, e_start, n_e, e_step, broadening, max_iter):

        self.max_iter = max_iter
        e = num.zeros(n_e, num.Float)
        e_b = num.zeros(n_e, num.Complex)
        for  i in range(n_e):
            e[i] =  e_start + e_step * i
            e_b[i] =  complex(e[i], broadening)

        print e
        print e_b
        c_frac_x = self.cont_frac(e_b,0,1)
        c_frac_y = self.cont_frac(e_b,1,1)
        c_frac_z = self.cont_frac(e_b,2,1)

        print c_frac_x
        print c_frac_y
        print c_frac_z
        
#        c_frac = [c_frac_x, c_frac_y, c_frac_z]
        return e,  [1/(-c_frac_x.imag/math.pi), 1/(-c_frac_y.imag/math.pi), 1/(-c_frac_z.imag/math.pi)]
        
        
    def cont_frac(self, shift, xyz, i):
        """  the continued fraction
        
        =========== ===================================
        Parameters:
        =========== ===================================
        shift       a num.array with
        the complex shift, e + i*gamma
        i           current iteration
        =========== ===================================
        """
        
        if i > len(self.a) -1:
            asdasdasd
            #self.compute_lanczos_vectors(i)
            
        # get new a,b from table
        a =  num.array(self.a[i - 1])[xyz]
        b =  num.array(self.a[i - 1])[xyz]
        a_new = num.array(self.a[i])[xyz]
        b_new = num.array(self.b[i])[xyz]
        
        # check if converged, otherwise continue recursion
        if self.stopping_citerium(a, a_new, b, b_new, i):
            return self.terminator(a_new, b_new, shift)
        else:
#            print "going into level", i+1, b_new, b_new - b
            return a_new - shift - abs(b_new)**2 / self.cont_frac(shift, xyz, i + 1)

    def solve(self, x_out, x_in):
        # either go via the hasnip formula or via BiCGStab
        # BiCGStab version 
        # return self.solver.solve(self,x_in, x_out,debug=1)
        #sdas
        # CG version
        #def A(self,x_in,x_out):
        #    """Function that is called by CG. It returns S~-1Sx_in in x_out
        #    """
        #    kpt.apply_overlap(self.nuclei, x_in, self.x_tmp)
        #    kpt.apply_inverse_overlap(self.nuclei, self.x_tmp, x_out)
        #
        #print "length", x_in.shape

        x_tmp =  self.gd.zeros(3)
        self.kpt.apply_inverse_overlap(self.nuclei, x_in, x_tmp)
   
        CG(self.A, x_out, x_tmp, tolerance=1.0e-25)
        
    def A(self, x_in, x_out):
        """Function that is called by CG. It returns S~-1Sx_in in x_out
        """
        
        self.kpt.apply_overlap(self.nuclei, x_in, self.x_tmp)
        self.kpt.apply_inverse_overlap(self.nuclei, self.x_tmp, x_out)

#    def dot(self, x_in, x_out):
#        """Applies S to vector, used in the BiCGStab algorithm
#        to solve Sx_out = x_in"""
#        self.apply_s(x_in, x_out)

    def apply_s(self, x_in, x_out):
        self.kpt.apply_overlap( self.calc.pt_nuclei, x_in, x_out )

    def apply_h(self, x_in, x_out):
        self.kpt.apply_hamiltonian( self.calc.hamiltonian, 
                                    x_in, x_out )

    def terminator(self, a, b, shift):
        """ Analytic formula to terminate the continued fraction from
        [R Haydock, V Heine, and M J Kelly, J Phys. C: Solid State Physics, Vol 8, (1975), 2591-2605]
        """
        print (shift.real - a)**2
        print 0.5 * ( shift.real - a - num.sqrt( (shift.real - a)**2 - 4 * abs(b)**2 ) / ( abs(b)**2 ))
        return  0.5 * ( shift.real - a - num.sqrt( (shift.real - a)**2 - 4 * abs(b)**2 ) / ( abs(b)**2 ))


    def stopping_citerium(self, a, a_new, b, b_new, i):
        """ checks if the maximum number of iterations is exceeded
        and if abs(a - a_new) < self.tol, and abs(b - b_new) < self.tol
        the continued fraction is converged
        """

        if i > self.max_iter:
            #print "Error, the continued fraction did not converge!"
            return True
        #elif abs(a - a_new) < self.tol and abs(b - b_new) < self.tol:    
        #    print "Continued fraction converged in", i, "iterations" 
        #    return True
        else:
            return False
            
    def compute_lanczos_vectors(self, it):
        """for iteration number it, compute the lanczos vectors and recursion coefficients
        updates self.max_previous_iter
        """
        
        assert len(self.a) == it  
        assert len(self.b) == it +1 
        
        
        # right eigenvector |u_new_r> = S-1H |u_i_r> - a_i|u_i_r> - b_i|u_i-1_r> 
        # first apply H on  |u_i_r>  so that H|u_i_r> = |v_r> then solve S|z_r> = |v_r>
        # now |z_r> = S-1H |u_i_r>
        
        
        v_r = self.gd.zeros(3) #num.zeros(phi_t_in.shape, num.Float)
        self.apply_h(self.u_lat_r, v_r)
        z_r = self.gd.zeros(3) #num.zeros(phi_t_in.shape, num.Float) 
        self.solve(z_r, v_r)
        
        a_tmp=[]
        for u_l, u_r, z in zip(self.u_lat_l,self.u_lat_r, z_r):
            a_tmp.append(self.gd.integrate(u_l * z)) #/ self.gd.integrate(u_l * u_r))
        self.a.append(a_tmp)
        
        
        u_new_r = self.gd.zeros(3)
        i=0
        for  z, u_r, u_n_r, a, b in zip(z_r, self.u_lat_r,
                                        self.u_nlat_r, self.a[it], self.b[it]) :
            u_new_r[i] = z - a * u_r - b * u_n_r
            i +=1
            
            
        # left eigenvector  <u_new_l| = <u_i_l| S-1H - <u_i_l| a_i - <u_i-1_l| b_i 
        # <u_i_l| S-1H = (HS-1|u_i_l>)*
        # so we solve S v_l = u_i_l 
        # then H v_l = z_l
        # <z_l| = (|z_l>)* = (HS-1|u_i_l>)*
        
        v_l = self.gd.zeros(3)
        self.solve(v_l, self.u_lat_l)
        z_l = self.gd.zeros(3) 
        self.apply_h(v_l, z_l)
        
        u_new_l = self.gd.zeros(3)
        i=0
        for z, u_l, u_n_l, a, b in  zip(z_l, self.u_lat_l,
                                        self.u_nlat_l, self.a[it], self.b[it]) :
            u_new_l[i] = z - a * u_l - b * u_n_l
            i +=1
            
        # calculate next b
        b_tmp=[]
        for u_l, u_r in zip(u_new_l, u_new_r): 
            b_tmp.append( math.sqrt( self.gd.integrate(u_l * u_r))) 
        self.b.append(b_tmp)
        
        #normalize and set the 4 new saved vectors 
        self.u_nlat_r = self.u_lat_r.copy()
        self.u_nlat_l = self.u_lat_l.copy()
        i=0
        for u_r, b in zip(u_new_r, self.b[it +1]): 
            self.u_lat_r[i] = u_r / b
            i += 1
        i=0
        for u_l, b in zip(u_new_l, self.b[it +1]): 
            self.u_lat_l[i] = u_l / b
            i += 1 


        #test
        for i in range(3):
            print "<u_0|u0>",i, self.gd.integrate(self.u_nlat_r[i] * self.u_nlat_l[i])
            print "<u_1|u1>",i, self.gd.integrate(self.u_lat_r[i] * self.u_lat_l[i])
            print "<u_1|u0>",i, self.gd.integrate(self.u_lat_r[i] * self.u_nlat_l[i])
       
    def compute_lanczos_vectors2(self, it):
        """for iteration number it, compute the lanczos vectors and recursion coefficients using JJ's method
        """
        
        assert len(self.a) == it  
        assert len(self.b) == it +1 

    
        # right eigenvector |u_new_r> = S-1H |u_i_r> - a_i|u_i_r> - b_i|u_i-1_r> 
        # first apply H on  |u_i_r>  so that H|u_i_r> = |v_r> then solve S|z_r> = |v_r>
        # now |z_r> = S-1H |u_i_r>

      
            
        z = self.gd.zeros(3) #num.zeros(phi_t_in.shape, num.Float)
        self.solve(z, self.u_lat_r)
        y = self.gd.zeros(3) #num.zeros(phi_t_in.shape, num.Float) 
        self.apply_h(z,y)
        
        a_tmp=[]
        for y1, z1 in zip(y, z):
            a_tmp.append(self.gd.integrate(y1 * z1)) #/ self.gd.integrate(u_l * u_r))
        #print a_tmp
        self.a.append(a_tmp)
        

        u_new_r = self.gd.zeros(3)
        i=0
        for  y1, u_r, u_n_r, a, b in zip(y, self.u_lat_r,
                                        self.u_nlat_r, self.a[it], self.b[it]) :
            u_new_r[i] = y1 - a * u_r - b * u_n_r
            i +=1


        # calculate next b
        z_lat = self.gd.zeros(3)
        self.solve(z_lat,u_new_r)

        z_nlat = self.gd.zeros(3)
        self.solve(z_nlat, self.u_lat_r)


        
        b_tmp=[]
        for z1, u_r in zip(z_lat, u_new_r): 
            b_tmp.append( math.sqrt( self.gd.integrate(z1 * u_r))) 
        self.b.append(b_tmp)
        
        #normalize and set the 4 new saved vectors 
        self.u_nlat_r = self.u_lat_r.copy()
        #self.u_nlat_l = self.u_lat_l.copy()
        i=0
        for u_r, b in zip(u_new_r, self.b[it +1]): 
            self.u_lat_r[i] = u_r / b
            i += 1
     
        self.y_lat = y.copy()
     
        #test
        for i in range(3):
            print "<u_0|u0>",i, self.gd.integrate(self.u_lat_r[i] * z_lat[i]/self.b[it +1][i])
            print "<u_0|u0>",i, self.gd.integrate(self.u_nlat_r[i] * z_nlat[i])
            print "<u_1|u1>",i, self.gd.integrate(self.u_lat_r[i] * z_nlat[i] )
           
       
