import numpy as np
from scipy.special import sph_harm
import itertools
from functools import reduce

def fermi_function(energy, temperature, chemical_potential):
    if temperature != 0:
        return 1/(np.exp((energy-chemical_potential)/temperature) + 1)
    else:
        return float(int(energy < chemical_potential))




class AllElectronResponse:
    def __init__(self, all_electron_atom, nspins):
        if nspins != 1:
            raise NotImplementedError
        self.all_electron_atom = all_electron_atom #Main calculation object in aeatom.py
        self.nspins = nspins
        self.initialize()

        ang_num = min(3, len(all_electron_atom.channels)-1)
        print("Testing for angular number: {}".format(ang_num))
        chempot = (all_electron_atom.channels[ang_num].e_n[1] + all_electron_atom.channels[ang_num].e_n[0])/2

        self.calculate_exact_chi_channel(ang_num, 0, chempot)


    ###TODO###

    ##What is the mass?
    ##How does kin energy look in aeatom.py?
    ##Do the projectors need to change? No, just use standard form psi(r,theta,phi)psi(r', theta', phi') but remember spherical integral measure

    #Solve Sternheimer equation via bicgstab

    #Power method algo


    #Get every wavefunction from AEA. What do we mean by every?

    ###END TODO###

    def initialize(self):
        self.theta_grid = np.linspace(0, np.pi, 200)
        self.phi_grid = np.linspace(0, 2*np.pi, 400)
        self.radial_grid = self.all_electron_atom.rgd.r_g
        #self._set_up_spherical_harmonics()
        #self._calc_valence_wavefunctions()

    def _set_up_spherical_harmonics(self):
        max_total_angular_momentum = len(self.all_electron_atom.channels)
        self.spherical_harmonics = {
            l : 
            [
                np.array([sph_harm(m, l, phi,theta) 
                          for theta,phi in itertools.product(self.theta_grid, self.phi_grid)])
                for m in range(-l, l)
            ] 
            for l in range(max_total_angular_momentum)
        }
                

    #Extract valence wavefunctions as function of (r,theta,phi)
    def _calc_valence_wavefunctions(self):
        self.valence_wfs = {
            l : 
            [
                (np.kron(n_Rfunc[1], spherical_harm), channel.e_n[n_Rfunc[0]])

                for n_Rfunc, spherical_harm in itertools.product(enumerate(channel.phi_ng), self.spherical_harmonics[l])
            ] 
            for l, channel in enumerate(self.all_electron_atom.channels)
        }   





    def calculate_exact_chi_channel(self, angular_number, omega = 0, temp = 0, chemical_potential = 0, eta = 0.001):
        '''
        This function returns chi (the response function) projected to the angular number (total angular momentum number) specified, i.e. returns chi(r,r')_l where

        chi(r, r')_l = \int d\Omega d\Omega' Y_{lm}(\Omega)^* chi(r\Omega, r'\Omega') Y_{lm}(\Omega')

        (independent of m due to spherical symmetry)

        '''
        #TODO can be made more efficient by only summing over pairs where n_1 < n_2 and then using symmetries of chi to get the rest.
        radial_wfs, energies = self._get_radial_modes(angular_number)

        wf_en_tups = zip(radial_wfs, energies)
        
        combos = itertools.product(wf_en_tups, wf_en_tups)
        print("number of wavefunctions: {}".format(len(radial_wfs)))
        def delta(pair):
            wf_en1, wf_en2 = tup
            wf1, en1 = wf_en1
            wf2, en2 = wf_en2

            delta = np.outer(wf1.conj(), wf1)
            delta += np.outer(wf2.conj(), wf2)

            delta *= fermi_function(en1, temp, chemical_potential) - fermi_function(en2, temp, chemical_potential)
            delta *= 1/(en1 - en2 + omega + 1j*eta)

            return delta
            

        chi_channel = reduce(lambda a, b : a + delta(b), combos, 0)

        return chi_channel


    def _get_radial_modes(self, angular_number):
        channel = self.all_electron_atom.channels[angular_number]

        states = channel.phi_ng
        energies = channel.e_n

        return states, energies
        


             
    
    #Calculate exact dielectric matrix via AE wfs
    def old_calculate_exact_dielectric_matrix(self, omega = 0, temp=0, chemical_potential = 0, eta = 0.0001):
        raise NotImplementedError
        wfs = self._get_wfs_for_exact() #Something
        def calc_chi_delta(tup):
            wf_tuple1, wf_tuple2 = tup
            state1, energy1 = wf_tuple1
            state2, energy2 = wf_tuple2
            if energy1 == energy2:
                return 0
            delta = np.outer(state1.conj(), state1) #psi_n(x)psi_n(x')
            delta *= np.outer(state2.conj(), state2) #psi_m(x)psi_m(x')
            delta *= 1/(energy1 - energy2 + omega + 1j*eta)
            delta *= fermi_function(energy1, temp, chemical_potential) - fermi_function(energy2, temp, chemical_potential)
            return delta
        
        all_wf_combs = itertools.product(wfs, wfs)
        
        chi = reduce(lambda a,b : a + calc_chi_delta(b), all_wf_combs, 0)

        coulomb_matrix = self._get_coulomb_matrix()
        
        return np.eye(chi.shape[0]) - self.dot(coulomb_matrix, chi)
            


    def solve_sternheimer(self, potential, omega = 0):
        assert np.real(omega) == 0

        alpha = 1

        valence_projector = self.get_valence_projector()
        conduction_projector = np.eye(valence_projector.shape[0])*self.delta_function_norm() - valence_projector

        assert np.allclose(self.dot(valence_projector, conduction_projector), np.zeros_like(valence_projector))

        ham = self.get_hamiltonian()
        
        raise NotImplementedError


    def _get_wfs_for_exact(self):
        raise NotImplementedError


    def get_valence_projector(self):
        raise NotImplementedError

    def get_hamiltonian(self):
        #Get Hamiltonian - in position basis? in gaussian basis?
        raise NotImplementedError
    
    def _get_coulomb_matrix(self):
        r_theta_phi_combos = itertools.product(self.radial_grid, self.theta_grid, self.phi_grid)
        def x_y_z(combo):
            r, theta, phi = combo
            return r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
        
        num_grid_points = len(self.radial_grid)*len(self.theta_grid)*len(self.phi_grid)
      
        coulomb_matrix = np.zeros((num_grid_points, num_grid_points))


        cutoff = 0.1
        for k1, combo1 in enumerate(r_theta_phi_combos):
            for k2, combo2 in enumerate(r_theta_phi_combos):
                x1,y1,z1 = x_y_z(combo1)
                x2,y2,z2 = x_y_z(combo2)               
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                coulomb_matrix[k1,k2] = 1/(distance + cutoff)

        return coulomb_matrix

    def dot(self, x1, x2):
        #Implements dot product with proper volume measure
        raise NotImplementedError


    def delta_function_norm(self):
        raise NotImplementedError











