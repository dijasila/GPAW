""" Class to handle constraint DFT - where a region 
    defined by atomic densities is constraint to contain
    a particular charge value N_c. 

    This is achieved by defining a new functional

                      |  /                 |
    F[n,V] = E[n] + Vc|  | w(r)n(r)dr - Nc |
                      |  /                 |

    To define the charge region one needs to consider 
    a weight, w(r), which applies to the molecular space.

    To that end the Hirshfeld definitions are used

                 sum_(i in mol)   n_i(r)
        w_i(r) = -----------------------
                     sum_tot n_j(r)

    where ...

"""

from math import pi, sqrt
import numpy as np
from ase.units import Bohr, Hartree
from ase.parallel import rank
from WeightFunction import WeightFunc
import pickle
import gpaw.mpi as mpi
#from constraint_potential import CDFTexternal

class CDFT:

    def __init__(self, N, calc, atoms, atom_ind, iterations = 50,
                 C = 1.e-2, method='gauss', Vc=None, forces = None,
                 V_update = 'factor', alpha_max=1.,alpha_min = 1.e-5,
                 step_size_control = 'factor_step',factor_values = [10.,5.,0.1],
                 force_calculation ='analytical',outfile = 'cdft.out', restart = False):

        # N : charge constraint value, positive means
        # less electron(s) within the molecular region
        # calc : GPAW calculator object
        # atom_ind : array indexing atoms which comprise
        # a molecule (e.g. np.arange(i,i+n,1), or similar)
        # C : threshold value for charge constraint -
        # number of electrons in molecular weight space
        # V_update = method to update V constraint: 
        #          factor, newton or CG
        # step_size_contol = how to choose step size for CG and newton
        #                  armijo, factor_step, goldstein
        # alpha_max and alpha_min: maximum and minimum step sizes
        # force_calculation = 'fd' or 'analytical' --> how cdft forces
        # are computed
        """ Regarding N: if calculator has the charge variable
            set, and the goal is to trap that charge on a 
            molecule, then set: N = charge to be consistent. 

        """

        ###############################################
        # method 'gauss':                             #
        # n_i(r-R_i) is defined as a std norm gauss   #
        # scaled with the charge of atom i            #
        #                                             #
        #                 N_el,i          |-(r-R_i)^2|#
        #   n_i(r-R_i) = ------------- exp|----------|#
        #                s(2*pi)^(1/2)    |  2*s^2   |#
        #                                             #
        # a cut-off, Rc, is defined as well, see      #
        # WeightFunc ...                              #
        ###############################################
        self.outfile = outfile
        self.restart = restart
        self.N     = np.array(N) # Charge 
        self.calc  = calc
        self.force_calculation = force_calculation
        self.atoms = atoms
        self.I      = 0     # Outer loop index
        #if self.atoms.get_calculator() is None:
        self.atoms.set_calculator(calc)
        
        self.index = atom_ind
        self.V_update_method = V_update
        self.ls_mode =  step_size_control
      
        self.n_const = len(N) # number of constrained regions
        self.alpha0 = alpha_max
        self.step_size = 0.
        self.alpha_min = alpha_min
        self.factor_values = factor_values
        self.iter = iterations
        self.forces = forces
        
        # Neutral charge of molecule
        
        Zn = np.zeros(self.n_const)
        for j in range(self.n_const):
            for atom in self.atoms[self.index[j]]:  
                    Zn[j] += atom.number

        self.Nc = Zn - self.N  # Charge constraint

        # Hold on to for later:
        self.weight   = None # The weight function --> array
        self.dens0 = None # Density at i-1, may loaded from a previous calculation
        self.dens     = None # Density at i
        self.n_c      = np.zeros(self.n_const)  # Values of dens in w(r) region
        self.n_c0     = np.zeros(self.n_const)    # Values of dens in w(r) region at i-1
        self.F_c      = np.zeros(self.n_const)    # Constraint integral values
        self.factor   = np.zeros(self.n_const)    # acceleration factors for climb/descent
        
        self.C = C
            
        self.iter = iterations
        
        if Vc:
            self.Vc = np.array(Vc)
        else:
            self.Vc = np.zeros(self.n_const)
            for i in range(self.n_const):
                self.Vc[i] = np.sign(self.N[i])*0.1   
            # Scalar applied to weight -> V_ext
                            # As a Vext it has units [H/e] ...
        
        # Updated variables
        self.Nct = np.zeros(self.n_const)  # pseudo comp charges
        self.Vc0 = np.zeros(self.n_const)  # Vc at i-1     
        
        ### Initialize calc ###
        
        if self.restart:
            self.dens0 = self.calc.get_all_electron_density(
                                       gridrefinement=2,
                                       collect=False,
                                       broadcast = False,
                                       pad = False)
            self.gd = self.calc.density.finegd
            self.comm = self.gd.comm
            self.vol = self.gd.dv
            # Initialize weight
            self.get_weight_function()
            self.n_c0 = self.get_integral(self.dens0)
            # do cDFT calculation
            self.run()
            
        else:
            self.energy = self.atoms.get_potential_energy() # DFT Energy
            self.dens0 = self.calc.get_all_electron_density(
                                       gridrefinement=2,
                                       collect=False,
                                       broadcast = False,
                                       pad = False)
            self.gd = self.calc.density.finegd
            self.comm = self.gd.comm
            self.vol = self.gd.dv
            # Initialize weight
            self.get_weight_function()
            self.n_c0 = self.get_integral(self.dens0)
            self.run()
            
        # Grab a total size of the grid 
        self.grid_size = self.dens0.size
        self.grid_shape = self.dens0.shape       
        
    def get_weight_function(self):
        w = []
        for i in range(self.n_const):
            wf = WeightFunc(self.gd,
                        self.atoms,
                        self.index[i])

            w.append(wf.construct_weight_function())

        self.weight = np.array(w)

        # units: None (just a scale) (Per cpu)
       
    def get_integral(self, dens):
        ''' returns 
            /                 
            | w(r)n(r)dr 
            /                 
        '''
        
        n_c = []

        for i in range(self.n_const):
 
            w = self.weight[i]
                
            nc = self.gd.integrate(w * dens * Bohr**3, global_integral = True)
            n_c.append(nc)
                
        nc = np.array(n_c)
        return nc

    def update_energy(self):
        # The potential energy due to v_ext is included in
        # the output from GPAWs E_KS. Missing the Nc component!
        # Since v_ext only acts on the pseudo density
        # Nc ~> Nc_t = sum_a Z_at - n_c 
        # where t is tilde.
   
        # Collect compensation charges:
        comp_ch = np.zeros(len(self.atoms))
        for a, D_sp in self.calc.density.D_asp.items():
            comp_ch[a] += (np.dot(D_sp[:self.calc.wfs.nspins].sum(0),
                                  self.calc.wfs.setups[a].Delta_pL)[0]\
                                + self.calc.wfs.setups[a].Delta0)

        self.comm.sum(comp_ch)
        comp_ch *= (-1) * sqrt(4 * pi)
        for i in range(self.n_const):
            self.Nct[i] = comp_ch[self.index[i]].sum() - self.N[i]
            self.correction = self.Vc[i] * self.Nct[i] * Hartree
            # The energy is -(Vc * Nct)
            self.energy -= self.Vc[i] * self.Nct[i] * Hartree

    def cdft_print(self):
        # something something logfile etc.
        # print >> logfile, stuff

        # Print header first? somewhere... maybe in init
        if rank == 0:
        
            with open(self.outfile, 'a') as out:
                print >> out, 'Iter:%d' %self.I, \
                'Energy:%5.4f' %self.energy, \
                'Vc:%4.3f'%self.Vc, \
                'Dif. from goal:%4.3f'%self.F_c, \
                'Conv:',self.check_criteria(), \
                'Step size:%4.3f'%self.step_size,\
                'Correction:%5.4f'%self.correction

    def update_Vc(self, method = 'factor'):
        # Update Vc and I
        ###################################
        # given d0 for Vc = 0, and d1 for #
        # Vc1 = num - grab a:             #
        #     d1 - d0                     #
        # a = --------                    #
        #     Vc1 - Vc                    #
        #                  d1 - Nc        #
        # set: VcNc -= b * --------       #
        #                     a           #
        #                                 #
        # method = 'factor' or 'newton'   #
        ###################################
       
        if self.V_update_method == 'factor':
            for i in range(self.n_const):
                a = (self.n_c[i] - self.n_c0[i]) /\
                (self.Vc[i] - self.Vc0[i])
 
                self.n_c0[i]  = self.n_c[i]
                self.Vc0[i]   = self.Vc[i]
                self.dens0 = self.dens
                
                if self.factor is None:
                    self.factor = self.get_acceleration()
                    
                # Update Vc
                self.Vc[i] -= self.factor[i] * self.F_c[i] / a
                #self.Vc[i] -= 0.01 * self.F_c[i] 
                # Make b a evolving variable! 
                # (larger as difference is smaller - up to unity)
       
        elif self.V_update_method == 'CG':
            self.get_cg_update()
        elif self.V_update_method == 'newton':
            self.get_newton_update()

        self.I += 1   
    
    def check_criteria(self):
        """ The convergence criteria is:
                     | /                 |
            C <= abs | | w(r)n(r)dr - Nc |
                     | /                 |
        """
        return all( fc <= self.C for fc in abs(self.F_c) )
    


    def get_acceleration(self):
        # acceleration factor for the climb/descent

        acceleration_temp = np.zeros(self.n_const)
        for i in range(self.n_const):
            acceleration_temp[i] = \
            0.4 + 0.6 /\
                    (1.0 + np.exp(30.0 * (self.F_c[i])**2))
        return acceleration_temp

    def get_factor_step(self):
        ''' smooth transition between
        alpha_min and alpha0 (ie max step)
        using the gradient F_c,
        takes large step when density changes
        a lot and small near convergence'''
        fac = self.factor_values
        alpha = -(self.alpha0 - self.alpha_min)/\
                 np.exp(fac[0]*(np.sum(np.abs(self.F_c))**2)+\
                 fac[1]*np.sum(np.abs(self.F_c))+\
                 fac[2]*np.square(np.sum(np.abs(self.F_c)))) +\
                  self.alpha0

        return np.abs(alpha)
    
    def get_cg_update(self, cg_type ='DaiYuan'):
        '''type = DaiYuan or PolakRibiere
        specifies the new CG direction '''
        self.type = cg_type
        #current -gradient
        self.dx = -self.F_c
        # initial direction
        if self.I == 0:
            self.dir_1 = self.dx
            #make a unit vector
            self.dir_1 = self.dir_1/np.linalg.norm(self.dir_1)
        #previous -gradient
        self.dx_1 = -(self.n_c0 - self.Nc)
        self.Vc0 = self.Vc
        
        self.update_hessian()
        
        if self.type == 'DaiYuan':
            self.beta = np.dot(np.transpose(self.dx), self.dx)/ \
                      np.dot( np.transpose(self.dir_1), (self.dx-self.dx_1))
         
        elif self.type == 'PolakRibiere':

            self.beta = np.dot(np.transpose(self.dx), (self.dx-self.dx_1))/ \
                      np.dot( np.transpose(self.dx_1), self.dx_1)
        
        # conjugate direction
        # first check that self.beta > 0 or reset
        self.beta = max(self.beta,0.)
        
        self.dir = self.dx + self.beta * self.dir_1
        #make into a unit vector
        self.dir = self.dir/np.linalg.norm(self.dir)
        
        #step size from line search
        
        self.step_size = self.cdft_linesearch()
        
        # substract because maximizing
        
        self.Vc = self.Vc0 - self.step_size * self.dir
        
        #self.beta0 = self.beta
        self.dir_1 = self.dir
        #update
        self.n_c0  = self.n_c
        self.dens0 = self.dens
        
    def get_newton_update(self):
        ''' V_i+1 = V_i + E'({V_i})/E''({V_i})
         E'({V_i}) ~ E'(V_i) i.e. update 
        not the Hessian
        
        E'(V_i) = self.F_c'''
        
        self.update_hessian()
        self.dir = self.F_c*np.linalg.inv(self.Hk)
        # use the unit vector
        self.dir = self.dir/np.linalg.norm(self.dir)

        self.step_size = self.cdft_linesearch()
        
        self.Vc = self.Vc + self.step_size * \
                   self.dir
        
        self.Vc0 = self.Vc
        self.n_c0  = self.n_c
        self.dens0 = self.dens
       
    def cdft_linesearch(self):
        '''returns step size for cg or newton using 
        a line search'''
        if self.ls_mode == 'constant':
            return self.alpha0
        
        elif self.ls_mode == 'armijo':
            ''' The back-tracking line
            search satisfying the Armijo
            condition:
            f(xk+ak*pk) =< f(xk) + c*ak*g(xk)^T*pk
            where pk is the search direction
            and g is the gradient f'
            '''
            return self.armijo_line_search()
        elif self.ls_mode == 'factor_step':
            return self.get_factor_step()
            
    def armijo_line_search(self, c1 = 1.e-4):
        ''' phi(alpha) = E(Vx+alpha*pk)
        pk = direction
        search min [phi(alpha) ; alpha>0]''' 

        ''' Taylor expansion for predecting
        the energy at new alpha -->
        need to build Hessian using a BFGS
        type procedure'''

        def phi(alpha):
            # approximate energy at new V
            # from alpha (stepsize) using 2nd order Taylor 
            return self.energy + alpha*np.dot(self.dir,self.F_c)+ \
                     0.5*alpha**2*np.dot(np.transpose(self.dir),np.dot(self.Hk,self.dir))

        self.derphi0 = np.dot(self.F_c,self.dir)
        
        self.phi0 = phi(0.)
        self.phi_a0 = phi(self.alpha0)

        if self.phi_a0 <= self.phi0 + c1*self.alpha0*self.derphi0:
            
            return self.alpha0
        
        # quadratic interpolation
        
        self.alpha1 = -(self.derphi0) * self.alpha0**2 / 2.0 / (self.phi_a0 - self.phi0 - self.derphi0 * self.alpha0)
        self.phi_a1 = phi(self.alpha1)

        if (self.phi_a1 <= self.phi0 + c1*self.alpha1*self.derphi0) and self.alpha1 >= 0.:
            return self.alpha1
        
        while self.alpha1 > 0.: #alpha>0 is a descent direction
            #quadratic interplation
            factor = self.alpha0**2 * self.alpha1**2 * (self.alpha1-self.alpha0)
            
            a = self.alpha0**2 * (self.phi_a1 - self.phi0 - self.derphi0*self.alpha1) - \
                self.alpha1**2 * (self.phi_a0 - self.phi0 - self.derphi0*self.alpha0)
            a = a / factor
            
            b = -self.alpha0**3 * (self.phi_a1 - self.phi0 - self.derphi0*self.alpha1) + \
                self.alpha1**3 * (self.phi_a0 - self.phi0 - self.derphi0*self.alpha0)
            
            b = b / factor
            
            self.alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * self.derphi0))) / (3.0*a)
            self.phi_a2 = phi(self.alpha2)
            
            if (self.phi_a2 <= self.phi0 + c1*self.alpha2*self.derphi0):
                return self.alpha2
            
                
            if (self.alpha1 - self.alpha2) > self.alpha1 / 2.0 or (1 - self.alpha2/self.alpha1) < 0.96:
                self.alpha2 = self.alpha1 / 2.0
            self.alpha0 = self.alpha1
            self.alpha1 = self.alpha2
            self.phi_a0 = self.phi_a1
            self.phi_a1 = self.phi_a2
        
        return self.alpha_min 
       
    def update_hessian(self):
        if self.I == 0:
            #Initialize Hessian as identity
            #scaled with gradients
            self.Hk = np.abs(self.F_c)* np.identity(self.n_const)
        else:       
            self.Hk0 = self.Hk
            # Form new Hessian using BFGS
            # difference of Vs = s
            self.s = self.step_size * self.dir

            # difference of gradients = y
            # self.F_c0 = self.n_c0 - self.Nc
            self.y = self.F_c - (self.n_c0 - self.Nc)

            #BFGS step
            #Hk = Hk0 + y*yT/(yT*s) - Hk0*s*sT*Hk0/(sT*Hk0*s)
            #form each term
            self.first_num = np.dot(self.y, np.transpose(self.y))
            self.first_den = np.dot(np.transpose(self.y),self.s)
            
            self.second_num = np.dot(self.Hk0 ,np.dot(self.s, np.dot( np.transpose(self.s),self.Hk0) ) )
            self.second_den = (np.dot(np.transpose(self.s), np.dot(self.Hk0, self.s)))
            
            self.Hk = self.Hk0 + \
                   self.first_num/self.first_den - \
                   self.second_num/self.second_den
            
        #make sure Hk is pos. def.eigs = np.linalg.eigvals(self.Hk)
        G = self.Hk.copy()
        eigs = np.linalg.eigvals(G)
        if not all( eig > 0. for eig in eigs):
            G = self.Hk.copy()
            while not all( eig > 0. for eig in eigs):
                #round down smallest eigenvalue with 2 decimals 
                mineig = np.floor(min(eigs)*100.)/100.
                G = G - mineig*np.identity(self.n_const)
                eigs = np.linalg.eigvals(G)
        self.Hk = G          

    def get_cdft_forces(self):
        '''Computes the cDFT contribution to force 
        that is the second term in
        f_cDFT_a = f_KS_a -\sum_i {V_i \nabla_a \int dr w_i(r) n(r)}
        actual calculation performed in weight function'''

        if self.atoms.calc is None:
            self.atoms.set_calculator(self.calc)

        f_KS = self.atoms.get_forces()
        
        f = WeightFunc(self.calc.density.finegd,
                        self.atoms,
                        self.index)
        
        f_cdft = f.get_cdft_forces(self.calc.get_all_electron_density(
                                       gridrefinement=2,
                                       collect=False,
                                       pad=False),
                                   self.atoms,
                                   self.Vc, self.vol,self.force_calculation)
        
        self.calc.wfs.world.broadcast(f_cdft,0) 
        f_KS += f_cdft
        self.forces = f_KS.reshape((-1, 3))
        self.calc.wfs.world.broadcast(self.forces,0)
        #mpi.synchronize_atoms(atoms, self.calc.wfs.world)  
        return self.forces
    
    def run(self, single=False,
            factor=None):
        # Determine weight function
        if self.atoms.calc is None:
            self.atoms.set_calculator(self.calc)
        self.get_weight_function()
        
        if factor:
            self.factor = factor
        else:
            self.factor = None

        # CLEAN!!!
        if single:
            self.I = self.iter -1
        else:
            self.I = 0

        while self.I < self.iter:

            # Construct external potential and add to calc

            self.calc.set(external=CDFTexternal(self.weight,
                                                self.Vc, self.n_const))
            self.atoms.set_calculator(self.calc)

            # Get and record potential energy
            self.energy = self.atoms.get_potential_energy()
            self.update_energy() # Add -(Vc*Nc) component to energy          

            # Grab density
            self.dens = self.calc.get_all_electron_density(
                                           gridrefinement=2,
                                           collect= False,
                                           pad = False)
          
            self.n_c = self.get_integral(self.dens)

            self.F_c = self.n_c - self.Nc
            
            if self.check_criteria():
                #converged
                self.cdft_print()
                self.I += self.iter + 1 # Breaks loop            
                continue
            else:
                self.cdft_print()
                self.update_Vc()
                continue
    
    def get_forces(self,atoms):

        if self.calculation_required(atoms, ['force']):
           self.update_for_optimization(atoms,quantity = 'force')
        return self.forces
        
    def get_stress(self, atoms):
        return np.zeros((3,3))

    def get_potential_energy(self,atoms):
        
        if self.calculation_required(atoms, ['energy']):
            self.update_for_optimization(atoms,quantity = 'energy')
        return self.energy

    def update_for_optimization(self, atoms,quantity = None):
        # do all the things needed for 
        # geometry optimization

        # KS calculation
        if self.calculation_required(atoms, ['energy','force']):
            self.atoms = atoms.copy()
            
            if quantity == 'energy':
                self.run()
            #compute forces
            elif quantity == 'force':
                self.get_cdft_forces()
            else:
                mpi.exit()

    def calculation_required(self, atoms, quantities):
        if len(quantities) == 0:
            return False
        if ((self.atoms != atoms) or
        (self.energy is None) or 
        (self.forces is None) or 
        (self.I == 0 )):
            return True
        if self.atoms is None:
            return True
        elif (len(atoms) != len(self.atoms) or
              (atoms.get_atomic_numbers() !=
               self.atoms.get_atomic_numbers()).any() or
              (atoms.get_initial_magnetic_moments() !=
               self.atoms.get_initial_magnetic_moments()).any() or
              (atoms.get_cell() != self.atoms.get_cell()).any() or
              (atoms.get_pbc() != self.atoms.get_pbc()).any()):
            return True
        elif (atoms.get_positions() !=
              self.atoms.get_positions()).any():
            return True
        else:
             return True
            
    def get_weight(self):
        # collect weight from all
        # CPUs and return
        w = self.gd.collect(self.weight,broadcast = True)
        return w

    def get_lagrange_multipliers(self):
        return self.Vc
        
    def cdft_energy(self):
        return self.energy
        
    def get_grid(self):
        return self.gd
    
    def get_targets(self):
        return self.N
     
    def get_calc(self):
       return self.calc

#### EXTERNAL POTENTIAL ####

class CDFTexternal:

    def __init__(self, weight, Vc, n_const):
        self.weight = weight
        self.Vc     = np.array(Vc)
        self.n_const = n_const
    
    def get_potential(self, gd=None):
        ###############################
        # Apply:                      #
        #                             #
        # dF[n,Vc]                    #
        # -------  =  v_eff + Vc w(r) #
        #   dn                        #
        #                             #
        # as v'_eff in KS.            #
        ###############################
        gd = None

        # Variables already determined in CDFT main
        pot = None
        for i in range(self.n_const):
            # sum over all constraints
            if pot is None:
                pot = self.weight[i] * self.Vc[i]
            else:
                pot += self.weight[i] * self.Vc[i]
        return pot

    def get_taylor(self, position=None, spos_c=None):
        """ No expansion, potential only interacts
            with the pseudo density atm.

        """
        # Test... weight function at core is 1.
        # Vc is potential value.
        return [[0]]
    
    def write(self, writer):
        if hasattr(self, 'todict'):
            from ase.io.jsonio import encode
            writer['ExternalPotential'] = encode(self).replace('"', "'")            




