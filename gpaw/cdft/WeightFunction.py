""" Class which builds a weight function around atoms or molecules
    given the atom index - using normalized Gaussian with cut-off!

    The weight function is constructed on the coarse or fine grid and
    can be used to do charge constraint DFT.

"""

import numpy as np
from math import pi
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
import gpaw.mpi as mpi

# Cut-off dict:
Rc = {
     }

# mu dict
mu = {
     }

class WeightFunc:

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
        self.ind   = indexes # Array of indexes

        # Construct Rc dict
        new = {}
        for a in self.atoms:
            if a.symbol in Rc:
                new[a.symbol] = Rc[a.symbol]
            else:
                elemement_number = atomic_numbers[a.symbol]
                cr = covalent_radii[elemement_number]
                #Rc to roughly between 3.5 and 5.
                new[a.symbol] = cr + 3.
            # Scale with known covalent radii?

        self.Rc = new

        # Construct mu dict
        new_mu = {}
        for a in self.atoms:
            if a.symbol in mu:
                new_mu[a.symbol] = mu[a.symbol]
            else:
                elemement_number = atomic_numbers[a.symbol]
                cr = covalent_radii[elemement_number]
                # mu to be roughly between 0.5 and 1.0
                cr = cr * min(covalent_radii) + 0.5
                new_mu[a.symbol] = cr

            # Scale with known covalent radii?

        # "Larger" atms may need a bit more width
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
        g(r) = ---------- e^{-(r-Ra)^2/(2mu^2)}
               mu(2pi)^.5 

        for |r-Ra| <= Rc, 0 elsewhere

        """
        # Check function
        check = abs(dis) <= Rc / Bohr
        # Make gaussian

        gauss = 1.0 / (mu * (2.0*pi)**(1./2.)) *\
               np.exp(-dis**2 / (2.0 * mu**2))
        # apply cut-off
        return np.array((gauss * check))

    def get_distance_vectors(self, pos):
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
 
        return np.sqrt((xyz**2).sum(axis=0))

    def construct_total_density(self, atoms):
        # Add to empty grid
        empty = self.gd.zeros()

        for atom in atoms:
            charge = atom.number
            symbol = atom.symbol
            pos = atom.position / Bohr

            dis = self.get_distance_vectors(pos)

            empty += charge *\
                     self.normalized_gaussian(dis,
                                              self.mu[symbol],      
                                              self.Rc[symbol])
        return empty

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


    def get_cdft_forces(self, dens, atoms, Vc, dv, method = 'fd'):
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
        V = cDFT constraint value
        dv = volume of grid element
        method = 'fd' or 'analytical' for
              finite difference or analytical
              dw/dR
        '''
        
        cdft_forces = np.zeros((len(self.atoms),3))
        f_xa, f_ya,f_za = self.gd.zeros(),self.gd.zeros(),self.gd.zeros()
        for a,atom in enumerate(self.atoms):
            if method == 'analytical':

                a_pos = atom.position #in Bohr
                #a_pos += np.random.rand()/1.e6 #to avoid 0 division
                d_a = self.get_analytical_derivative(atom,Vc)
                length = np.linalg.norm(a_pos)
                if length == 0.:
                    length = 1.
                
                f_xa = d_a * a_pos[0]/length
                f_ya = d_a * a_pos[1]/length
                f_za = d_a * a_pos[2]/length
            
            elif method == 'fd':
                f_xa, f_ya, f_za = self.get_fd_gradient(Vc, atom)
            
            # multiply by n(r) and integrate
            
            cdft_forces[a][0] += -self.gd.integrate(
                        f_xa * dens)#, global_integral=True)
             
            cdft_forces[a][1] += -self.gd.integrate(
                        f_ya * dens)#, global_integral=True)
             
            cdft_forces[a][2] += -self.gd.integrate(
                        f_za * dens)#, global_integral=True)
                        
        return cdft_forces
        
    def get_analytical_derivative(self,nucleus, Vc, method = 'gauss'):
        #first inside weight function
        w_i = self.gd.zeros()   
        #then at the cutoff 'surface'
        w_s = self.gd.zeros()

        # derivatives inside and at surface
        d_inside = self.gd.zeros()
        d_surface = self.gd.zeros()
        
        # special values of the nucleus a
        a_index = nucleus.index
        a_symbol = nucleus.symbol
        a_charge = nucleus.number
        a_pos = nucleus.position / Bohr
        a_dis = self.get_distance_vectors(a_pos)
        
        # denominator
        dens = self.construct_total_density(self.atoms)
        # Check zero elements
        check = dens == 0
        # Add value to zeros ...
        dens += check * 1.0
        
        # Gaussian inside and at surface
        G_ai, G_as = self.construct_gaussian_derivative(nucleus)
        
        # loop over constrained regions
        for vi, region in enumerate(self.ind):
            
            atoms = self.atoms[region]
            
            # make weight function w_i
            for atom in atoms:
                charge = atom.number
                symbol = atom.symbol
                pos = atom.position / Bohr

                dis = self.get_distance_vectors(pos)

                w_i += charge * \
                     self.normalized_gaussian(dis,
                                              self.mu[symbol],      
                                              self.Rc[symbol])
                w_s += charge * \
                     self.normalized_gaussian(dis,
                                             self.mu[symbol],      
                                              self.Rc[symbol]) 
            w_i = w_i / dens
            w_s = w_s / dens
            
            # atom A in region?
            if a_index in region:
                w_i -= 1
                w_s -= 1
                
            d_inside += G_ai * w_i
            d_surface += G_as * w_s

        return -d_inside + d_surface
  
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
        check = dens == 0
        # Add value to zeros ...
        dens += check * 1.0
        
        # gaussian at nucleus a

        G_a =  a_charge * \
               self.normalized_gaussian(a_dis,
                   self.mu[a_symbol],      
                   self.Rc[a_symbol])        
                   
        # within cutoff or at surface ? --> heaviside
        # inside
        check_i = abs(a_dis) < self.Rc[a_symbol] / Bohr        
        #rRc =  np.abs(check_i*a_dis) 
        rRc = check_i*a_dis
        
        #surface
        check_s = abs(abs(a_dis) - self.Rc[a_symbol] / Bohr) <= max(self.gd.get_grid_spacings())
        
        #reinforce cutoff (Heaviside(r-Rc)*Ga); inside
        G_ai = rRc * G_a / (self.mu[a_symbol])**2  # (\Theta * (r-R_a) n_A) / \sigma^2
        G_ai = G_ai / dens # / sum_k n_k
        
        #surface
        G_as = check_s * G_a #\ sigma_{A\in i} n_A
        G_as = G_as / dens # / sum_k n_k
        
        return -G_ai, G_as

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
        grid_spacings = self.gd.get_grid_spacings()
        #dx = dy = dz = 1
        dx = grid_spacings[1]/100.
        dy = grid_spacings[1]/100.
        dz = grid_spacings[2]/100.

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
        
        # move to weight to + direction
        ###############################
        nuc_pos = atoms[nucleus].position
        if direction == 'x':
            atoms[nucleus].position = nuc_pos + [dx,0.,0.]
        elif direction == 'y':
            atoms[nucleus].position = nuc_pos + [0.,dy,0.]
        elif direction == 'z':
            atoms[nucleus].position = nuc_pos + [0.,0.,dz]
        
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

        # move to weight to - direction
        ###############################
        # move twice        
        if direction == 'x':
            atoms[nucleus].position = nuc_pos - [2*dx,0.,0.]
        elif direction == 'y':
            atoms[nucleus].position = nuc_pos - [0.,2*dy,0.]
        elif direction == 'z':
            atoms[nucleus].position = nuc_pos - [0.,0.,2*dz]
        
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
        
        # get derivatives
        # try central
        dtest = 0
        for test in range(0,3):
            if dtest == 0:
                try:
                    if direction == 'x':
                        fd_der = (w_pos-w_neg)/(2*dx)
                    elif direction == 'y':
                        fd_der = (w_pos-w_neg)/(2*dy)
                    elif direction == 'z':
                        fd_der = (w_pos-w_neg)/(2*dz)
                    break
                except: # try forward
                    print 'central derivative failed'
            if dtest == 1:
                try:
                    if direction == 'x':
                        fd_der = (w_pos-w0)/(dx)
                    elif direction == 'y':
                        fd_der = (w_pos-w0)/(dy)
                    elif direction == 'z':
                        fd_der = (w_pos-w0)/(dz)
                    break
                except:
                    print 'forward derivative failed'
            if dtest == 2:
                try: #try backwards
                    if direction == 'x':
                        fd_der = (w0-w_pos)/(dx)
                    elif direction == 'y':
                        fd_der = (w0-w_pos)/(dy)
                    elif direction == 'z':
                        fd_der = (w0-w_pos)/(dz)
                except:
                    print 'backward derivative failed'
                    print 'fd calculation for forces failed!'
        
        return fd_der
