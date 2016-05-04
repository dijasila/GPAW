from __future__ import print_function
import functools

from ase.calculators.calculator import Calculator
from ase.data import covalent_radii, atomic_numbers
from ase.units import Bohr, Hartree
from ase.utils import convert_string_to_fd
import numpy as np
from math import pi, sqrt
from scipy.optimize import minimize

from gpaw.external import ExternalPotential
import gpaw.mpi as mpi           
        
class CDFT(Calculator):
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, calc, atoms, regions, charges, coefs=None, txt='-',
                method='BFGS', forces = 'analytical',
                 minimizer_options={'gtol':0.01, 'ftol': 1e-8, 'xtol':1e-8,
                 'max_trust_radius':1.,'initial_trust_radius':1.e-2}):
        
        """Constrained DFT calculator.
        
        calc: GPAW instance
            DFT calculator object to be constrained.
        regions: list of list of int
            Atom indices of atoms in the different regions.
        charges: list of float
            constrained charges in the different regions.
        coefs: list of float
            Initial values for constraint coefficients (eV).
        txt: None or str or file descriptor
            Log file.  Default id '-' meaning standard out.  Use None for
            no output.
        minimizer_options: dict
            options for scipy optimizers, see:scipy.optimize.minimize
        method: str
            One of scipy optimizers, e.g., BFGS, CG
        forces: str
            cDFT weight function contribution to forces
            'fd' for finite difference or 'analytical'
        """
        
        Calculator.__init__(self)
        self.calc = calc
        self.charge_i = np.array(charges, dtype=float)
        self.options = minimizer_options
        self.regions = regions
        self.forces = forces

        if coefs is None: # to Hartree
            self.v_i = 0.1 * np.sign(self.charge_i)
        else:
            self.v_i = np.array(coefs) / Hartree
        
        self.regions = regions
        
        # The objective is to constrain the number of electrons (nel)
        # in a certain region --> convert charge to nel
        self.regions = regions
        Zn = np.zeros(len(self.charge_i))
        for j in range(len(Zn)):
            for atom in atoms[self.regions[j]]:  
                    Zn[j] += atom.number
        
        self.constraints = Zn - self.charge_i

        self.log = convert_string_to_fd(txt)
        self.method = method
        
        # initialise without v_ext
        atoms.set_calculator(self.calc)
        atoms.get_potential_energy()
        self.cdft_initialised = False
        
        self.atoms = atoms
        self.gd = self.calc.density.finegd
       
        # construct cdft potential
        self.ext = CDFTPotential(regions = regions,
                    gd = self.gd, 
                    atoms = self.atoms, 
                    constraints = self.constraints,
                    txt=self.log)
        
        self.calc.set(external=self.ext)
            
    def calculate(self, atoms, properties, system_changes):
        # check we're dealing with same atoms
        if atoms != self.atoms:
            self.atoms = atoms
        
        Calculator.calculate(self, self.atoms)
        
        # update positions and weight functions
        if 'positions' in system_changes or not self.cdft_initialised:
            self.ext.set_positions_and_atomic_numbers(
                self.atoms.positions / Bohr, self.atoms.numbers)
            self.cdft_initialised = True
            
        self.atoms.calc = self.calc
        
        p = functools.partial(print, file=self.log)
        
        self.iteration = 0
        
        def f(v_i):
            #nonlocal iteration
            self.ext.set_levels(v_i)
            self.v_i = v_i
            Edft = self.atoms.get_potential_energy() # in eV

            self.ae_dens = self.calc.get_all_electron_density(gridrefinement=2,
                            collect=False,
                            pad=False) * Bohr**3
            
            # get the cDFT gradient 
            self.dn_i = (self.gd.integrate(self.ext.w_ig*self.ae_dens,global_integral=True) -
                    self.constraints)
            self.w = self.ext.w_ig
            
            if self.iteration == 0:
                n = 7 * len(self.v_i)
                p('Optimizer setups:{}'.format(self.options))
                p('iter {0:{1}} energy     errors'.format('coefs', n))
                p('     {0:{1}} [eV]       [e]'.format('[eV]', n))
            p('{0:4} {1} {2:10.3f} {3}'
              .format(self.iteration,
                      ''.join('{0:7.3f}'.format(v) for v in self.v_i * Hartree),
                      Edft,
                      ''.join('{0:6.4f}'.format(dn) for dn in self.dn_i)))
            
            self.iteration += 1

            return -Edft, -self.dn_i # return negative because maximising wrt v_i
        
        def hessian(v_i):
            # Hessian approximated with BFGS
            self.hess = self.update_hessian(v_i)
            return self.hess

        if self.method == 'trust-ncg' or self.method == 'dogleg':
            # these methods need hessian
            m = minimize(f, self.v_i, jac=True, method = self.method,
                     hess = hessian,options=self.options)
        else:
            m = minimize(f, self.v_i, jac=True, method = self.method,
                     options=self.options)
        assert m.success, m
        
        p(m.message + '\n')
        
        self.v_i = m.x
        
        try:
            self.dn_i = -m.jac
        except AttributeError:
             print ('{} does not have a Jacobian'.format(self.method))

        self.results['energy'] = -m.fun
        self.Edft = -m.fun 
        # Free energy <A|H^KS + V_a w_a|A> = Edft + <A|Vw|A>
        try:
            # remove external from Edft
            self.Edft -= np.dot(self.v_i, self.dn_i)
        except:
            pass
        
        self.Ecdft = self.Edft + np.dot(self.v_i, 
                                self.gd.integrate(self.w, self.ae_dens, 
                                global_integral=True)) 
        #forces
        
        f = WeightFunc(self.gd,
                    self.atoms,
                    self.regions)
        
        f_cdft = f.get_cdft_forces(self.ae_dens / (Bohr**3), self.v_i, self.forces)

        self.calc.wfs.world.broadcast(f_cdft,0) 

        self.results['forces'] = self.atoms.get_forces() + f_cdft

    def get_weight(self):
        return self.w

    def cdft_energy(self):
        return self.Ecdft

    def dft_energy(self):
        return self.Edft

    def get_lagrangians(self):
        return self.v_i

    def get_constraints(self):
        return self.constraints

    def get_grid(self):
        return self.gd
    
    def update_hessian(self,v_i):
        '''Computation of a BFGS Hessian to be 
        used with trust-ncg and dogleg optimizers

        returns a pos.def. hessian
        '''
        iteration = self.iteration - 1
        n_regions = len(self.regions)

        if iteration == 0:
            # Initialize Hessian as identity
            # scaled with gradients
            Hk = np.abs(self.dn_i)*np.identity(n_regions)

        else:       
            Hk0 = self.hess
            # Form new Hessian using BFGS
            s = v_i - self.old_v_i
            # difference of gradients = y
            y = self.dn_i - self.old_gradient
            #BFGS step
            #Hk = Hk0 + y*yT/(yT*s) - Hk0*s*sT*Hk0/(sT*Hk0*s)
            #form each term
            first_num = np.dot(y, np.transpose(y))
            first_den = np.dot(np.transpose(y),s)
                
            second_num = np.dot(Hk0 ,np.dot(s, np.dot( np.transpose(s),Hk0) ) )
            second_den = (np.dot(np.transpose(s), np.dot(Hk0, s)))
                
            Hk = Hk0 + \
                    first_num/first_den - \
                    second_num/second_den
                
        #make sure Hk is pos. def.eigs = np.linalg.eigvals(self.Hk)
        hess = Hk.copy()
        eigs = np.linalg.eigvals(hess)
        if not all( eig > 0. for eig in eigs):    
            hess = Hk.copy()
            while not all( eig > 0. for eig in eigs):
                #round down smallest eigenvalue with 2 decimals 
                mineig = np.floor(min(eigs)*100.)/100.
                hess = hess - mineig*np.identity(n_regions)
                eigs = np.linalg.eigvals(hess)
            
        self.old_gradient = self.dn_i
        self.old_v_i = self.v_i
        self.old_hessian = hess

        return hess 

def gaussians(gd, positions, numbers):
    r_Rv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
    radii = covalent_radii[numbers]
    cutoffs = radii + 3.0
    sigmas = radii * min(covalent_radii) + 0.5
    result_R = gd.zeros()
    for pos, Z, rc, sigma in zip(positions, numbers, cutoffs, sigmas):
        d2_R = ((r_Rv - pos)**2).sum(3)
        a_R = Z / (sigma**3 * (2 * np.pi)**1.5) * np.exp(-d2_R / (2 * sigma**2))
        a_R[d2_R > rc] = 0.0
        result_R += a_R
    return result_R
    
    
class CDFTPotential(ExternalPotential):
    def __init__(self, regions, gd, atoms, constraints, txt='-'):
        self.indices_i = regions
        self.gd = gd
        self.log = convert_string_to_fd(txt)
        self.atoms = atoms
        self.v_i = None
        self.pos_av = None
        self.Z_a = None
        self.w_ig = None
        self.constraints = constraints
        self.name = 'CDFT'

    def __str__(self):
        self.name = 'CDFT'
        return 'CDFTPotential'
        
    def get_name(self):
        return self.name
        
    def update_ae_density(self,ae_dens):
        self.ae_dens = ae_dens

    def get_atoms(self):
        return self.atoms

    def get_ae_density(self):
        return self.ae_dens
    
    def get_vi(self):
        return self.v_i
    
    def get_constraints(self):
        return self.constraints
    
    def set_levels(self, v_i):
        self.v_i = np.array(v_i, dtype=float)
        self.vext_g = None

    def get_w(self):
        return self.w_ig
        
    def set_positions_and_atomic_numbers(self, pos_av, Z_a):
        self.pos_av = pos_av
        self.Z_a = Z_a
        self.w_ig = None
        self.vext_g = None
        
    def initialize_partitioning(self, gd):
        self.w_ig = gd.empty(len(self.indices_i))
        ntot_g = gd.zeros()
        missing = list(range(len(self.Z_a)))

        N_i = []
        
        '''for i, indices in enumerate(self.indices_i):
            n_g = gaussians(gd, self.pos_av[indices], self.Z_a[indices])
            N_i.append(gd.integrate(n_g))
            ntot_g += n_g
            self.w_ig[i] = n_g
            for a in indices:
                missing.remove(a)
        
        ntot_g += gaussians(gd, self.pos_av[missing], self.Z_a[missing])
        ntot_g[ntot_g == 0] = 1.0
        self.w_ig /= ntot_g
        '''

        w = []
        # make weight functions
        for i in range(len(self.indices_i)):
            wf = WeightFunc(self.gd,
                        self.atoms,
                        self.indices_i[i])
            weig = wf.construct_weight_function()
            self.mu = wf.mu
            self.Rc = wf.Rc
            w.append(weig)
        
        self.w_ig = np.array(w)
        
        volume_i = self.gd.integrate(self.w_ig)

        p = functools.partial(print, file=self.log)
        p('Electrons:',
              ', '.join('{0}: {1:.3f} ???'.format(indices, N)
                        for indices, N in zip(self.indices_i, N_i)),
              file=self.log)
        p('Volumes:',
              ', '.join('{0}: {1:.3f} Ang^3'.format(indices, volume * Bohr**3)
                        for indices, volume in zip(self.indices_i, volume_i)),
              file=self.log)
        p('Parameters')
        p('Atom      Width[A]      Rc[A]')
        for a in self.mu:
            p('  {atom}     {width}   {Rc}'.format(atom=a, width =round(self.mu[a],3),
                   Rc =round(self.Rc[a],3)))
        print(file=self.log)

    def calculate_potential(self, gd):
        if self.w_ig is None:
            self.initialize_partitioning(self.gd)
        self.vext_g = np.einsum('i,ijkl', self.v_i, self.w_ig)



# Cut-off dict:
Rc = {
     }

# mu dict
mu = {
     }

class WeightFunc:
    """ Class which builds a weight function around atoms or molecules
    given the atom index - using normalized Gaussian with cut-off!

    The weight function is constructed on the coarse or fine grid and
    can be used to do charge constraint DFT.

    """
    def __init__(self, gd, atoms, indexes, Rc=Rc, mu=mu):
        """ Given a grid-descriptor, atoms object and an index list
            construct a weight function defined by:
                     n_i(r-R_i)
            w(r) = ---------------
                   sum_a n_a(r-R_a)

            where a runs over all atoms, and i can index
            an atom or a list of atoms comprising a molecule, etc.

            The n_i are construced with atom centered gaussians
            using a pre-defined cut-off Rc_i.

        """
        self.gd    = gd
        self.atoms = atoms
        self.ind   = indexes # Indices of constrained regions
         
        # Weight function parameters in Bohr 
        # Cutoffs
        new = {}
        for a in self.atoms:
            if a.symbol in Rc:
                new[a.symbol] = Rc[a.symbol] / Bohr
            else:
                elemement_number = atomic_numbers[a.symbol]
                cr = covalent_radii[elemement_number]
                #Rc to roughly between 3.5 and 5.
                new[a.symbol] = (cr + 3.) / Bohr

        self.Rc = new

        # Construct mu (width) dict
        # mu only sets the width and height so it's in angstrom
        new_mu = {}
        for a in self.atoms:
            if a.symbol in mu:
                new_mu[a.symbol] = mu[a.symbol] #/ Bohr
            else:
                elemement_number = atomic_numbers[a.symbol]
                cr = covalent_radii[elemement_number]
                # mu to be roughly between 0.5 and 1.0 AA
                cr = (cr * min(covalent_radii) + 0.5) 
                new_mu[a.symbol] = cr #/ Bohr

        # "Larger" atoms may need a bit more width
        self.mu = new_mu

    def normalized_gaussian(self, dis, mu, Rc):
        # Given mu - width, and Rc
        # a normalized gaussian is constructed
        # around some atom. This is
        # placed on the gd (grid) - and truncated 
        # at a given cut-off value Rc. dis
        # are the distances from atom to grid points.

        """ Normalized gaussian is:
                      1          
        g(r) = ---------------  e^{-(r-Ra)^2/(2mu^2)}
               mu^3*(2pi)^1.5 

        for |r-Ra| <= Rc, 0 elsewhere

        """
        # Check function
        check = abs(dis) <= Rc
        
        # Make gaussian 3D Guassian 
        gauss = 1.0 / (mu**3 * (2.0*pi)**(3./2.)) *\
               np.exp(-dis**2 / (2.0 * mu**2))
        
        # apply cut-off and return
        return np.array((gauss * check))

    def get_distance_vectors(self, pos,distance = True):
        # Given atom position [Bohr], grab distances to all
        # grid points - employ MIC when appropriate.

        # Scaled position of gpts on some cpu, relative to all gpts
        s_G = (np.indices(self.gd.n_c, float).T +\
               self.gd.beg_c) / self.gd.N_c
        # Subtract scaled distance from atom to box boundaries
        s_G -= np.linalg.solve(self.gd.cell_cv.T, pos)
        ## MIC
        s_G -= self.gd.pbc_c * (2 * s_G).astype(int)
        # Apparently doing this check twice works better ...
        s_G -= self.gd.pbc_c * (2 * s_G).astype(int)
        # x,y,z distances
        xyz = np.dot(s_G, self.gd.cell_cv).T.copy()
        if distance:
            # returns vector norm
            return np.sqrt((xyz**2).sum(axis=0))
        else:
            # gives raw vector 
            return xyz
    
    def construct_total_density(self, atoms):
        # Add to empty grid
        dens = self.gd.zeros()

        for atom in atoms:
            charge = atom.number
            symbol = atom.symbol
            pos = atom.position / Bohr

            dis = self.get_distance_vectors(pos)

            dens += charge *\
                     self.normalized_gaussian(dis,
                                              self.mu[symbol],      
                                              self.Rc[symbol])
        return dens

    def construct_weight_function(self):
        # Grab atomic / molecular density
        dens_n = self.construct_total_density(
                          self.atoms[self.ind])
        # Grab total density
        dens = self.construct_total_density(self.atoms)
        # Check zero elements
        check = dens == 0
        # Add value to zeros ...
        dens += check * 1.0
        # make weight function
        return (dens_n / dens)


    def get_cdft_forces(self, dens, Vc, method = 'fd'):
        ''' Calculate cDFT force as a sum
        dF/dRi = Fi(inside) + Fs(surf)
        due to cutoff (Rc) in gauss
                  / dw(r)
        Fi_a = -V | ------ n(r) dr
                  /  dR_a
        dw(r)
        ----  = sum of Gaussian functions...
        dR_a

        this is computed in get_dw_dRa
        dens = density
        Vc = cDFT constraint value
        method = 'fd' or 'analytical' for
              finite difference or analytical
              dw/dR
        '''
        
        cdft_forces = np.zeros((len(self.atoms),3))
        f_xa, f_ya,f_za = self.gd.zeros(),self.gd.zeros(),self.gd.zeros()
        
        for a,atom in enumerate(self.atoms):
            if method == 'analytical':

                a_pos = atom.position / Bohr # 
                #radial derivative 
                dw_dRa = self.get_analytical_derivative(atom,Vc)
                # from the chain rule
                # first dx = -(r-X_ai), i=x,y,z ...
                delta_x = -self.get_distance_vectors(a_pos, distance = False)
                
                # then dr = |r-Ra|
                delta_r = self.get_distance_vectors(a_pos, distance = True)
                check = delta_r == 0
                # Add value to zeros ...
                delta_r += check * 1.0                
                
                f_xa = dw_dRa * delta_x[0]/delta_r
                f_ya = dw_dRa * delta_x[1]/delta_r
                f_za = dw_dRa * delta_x[2]/delta_r
            
            elif method == 'fd':
                f_xa, f_ya, f_za = self.get_fd_gradient(Vc, atom)
            
            # multiply by n(r) and integrate
            
            cdft_forces[a][0] += -self.gd.integrate(
                        f_xa * dens, global_integral=True)
             
            cdft_forces[a][1] += -self.gd.integrate(
                        f_ya * dens, global_integral=True)
             
            cdft_forces[a][2] += -self.gd.integrate(
                        f_za * dens, global_integral=True)
                        
        return cdft_forces
        
    def get_analytical_derivative(self,nucleus, Vc):
        a_index = nucleus.index
        # derivatives 
        dw_dRa = self.gd.zeros()
        
        # denominator
        dens = self.construct_total_density(self.atoms)
        # Check zero elements
        check = dens == 0.
        # Add value to zeros ...
        dens += check * 1.0
        
        # Gaussian inside and at surface
        G_ai, G_as = self.construct_gaussian_derivative(nucleus)
        
        # loop over constrained regions
        for vi, region in enumerate(self.ind):
            w_i = self.gd.zeros() 
            atoms = self.atoms[region]
            
            # make weight function w_i
            w_i = self.construct_total_density(atoms)

            w_i = w_i / dens
            
            # atom A in region?
            if a_index in region:
                w_i -= 1
                
            dw_dRa += Vc[vi] * (G_ai + G_as) * w_i # sum regions

        return dw_dRa 
  
    def construct_gaussian_derivative(self, nucleus):
        # returns Ga * (r-Ra)* sum_i G_i
        # Ga is a gaussian at Ra nucleus
        
        a_index = nucleus.index
        a_symbol = nucleus.symbol
        a_charge = nucleus.number
        a_pos = nucleus.position / Bohr
        a_dis = self.get_distance_vectors(a_pos)
        
        #denominator
        dens = self.construct_total_density(self.atoms) # sum_k n_k
        # Check zero elements
        check = dens == 0.
        # Add value to zeros ...
        dens += check * 1.0
        
        # gaussian at nucleus a

        G_a =  a_charge * \
               self.normalized_gaussian(a_dis,
                   self.mu[a_symbol],      
                   self.Rc[a_symbol])        
                   
        # within cutoff or at surface ? --> heaviside
        # inside
        check_i = abs(a_dis) < self.Rc[a_symbol]        
        rRc = check_i*a_dis
        
        #surface
        check_s = abs(abs(a_dis) - self.Rc[a_symbol] ) <= max(self.gd.get_grid_spacings())
        
        #reinforce cutoff (Heaviside(r-Rc)*Ga); inside
        G_ai = rRc * G_a / (self.mu[a_symbol])**2  # (\Theta * (r-R_a) n_A) / \sigma^2
        G_ai = G_ai / dens # / sum_k n_k

        #surface
        G_as = check_s * G_a #\ sigma_{A\in i} n_A
        G_as = G_as / dens # / sum_k n_k
        
        return G_ai, G_as


    def get_fd_gradient(self, Vc, nucleus):
      # compute the forces using finite difference
      # for atom nucleus
      
      F_cdft_x = self.gd.zeros()
      F_cdft_y = self.gd.zeros()
      F_cdft_z = self.gd.zeros()
 

      for vi, region in enumerate(self.ind):
          der_x = self.finite_difference(nucleus.index, region, direction = 'x') 
          der_y = self.finite_difference(nucleus.index, region, direction = 'y')
          der_z = self.finite_difference(nucleus.index, region, direction = 'z')
                  
          F_cdft_x += Vc[vi] * der_x
          F_cdft_y += Vc[vi] * der_y
          F_cdft_z += Vc[vi] * der_z
          
      return F_cdft_x, F_cdft_y, F_cdft_z
                           
    def finite_difference(self, nucleus,region, direction='x'):
        # take finite difference of dw(r,Ra)/dXa = 
        # w(r, Ra+h)-w(r,Ra-h)/2h
        # weight functions are shifted accordingly
        # direction = x,y,or z
        dx = 1e-4

        atoms = self.atoms.copy()
        
        # first get the normal weight function
        ############################
        dens_n = self.construct_total_density(
                          atoms[region])
        # Grab total density
        dens = self.construct_total_density(atoms)
        # Check zero elements
        check = dens == 0
        # Add value to zeros ...
        dens += check * 1.0
        # make weight function
        (dens_n / dens)
        
        w0 = (dens_n / dens)
        
        # move  weight in + direction
        ###############################
        nuc_pos = atoms[nucleus].position
        if direction == 'x':
            atoms[nucleus].position = nuc_pos + [dx,0.,0.]
        elif direction == 'y':
            atoms[nucleus].position = nuc_pos + [0.,dx,0.]
        elif direction == 'z':
            atoms[nucleus].position = nuc_pos + [0.,0.,dx]
        
        dens_n = self.construct_total_density(
                          atoms[region])
        # Grab total density
        dens = self.construct_total_density(atoms)
        # Check zero elements
        check = dens == 0
        # Add value to zeros ...
        dens += check * 1.0
        # make weight function
        (dens_n / dens)
        
        w_pos = (dens_n / dens)

        # move weight in - direction
        ###############################
        # move twice        
        if direction == 'x':
            atoms[nucleus].position = nuc_pos - [2*dx,0.,0.]
        elif direction == 'y':
            atoms[nucleus].position = nuc_pos - [0.,2*dx,0.]
        elif direction == 'z':
            atoms[nucleus].position = nuc_pos - [0.,0.,2*dx]
        
        dens_n = self.construct_total_density(
                          atoms[region])
        # Grab total density
        dens = self.construct_total_density(atoms)
        # Check zero elements
        check = dens == 0
        # Add value to zeros ...
        dens += check * 1.0
        # make weight function
        (dens_n / dens)
        
        w_neg = (dens_n / dens)
       
        ####### fd derivative
        fd_der = (w_pos-w_neg)/(2*dx)

        return fd_der
