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
    def __init__(self, all_electron_atom):
        self.all_electron_atom = all_electron_atom #Main calculation object in aeatom.py
        self.initialize()


    ###TODO###
    #Set up Sternheimer equation in (r,theta,phi) space
    ##Kinetic energy will be different ~ spherical laplacian
    ##We think of Sternheimer equation as integral equation so we have to remember the spherical volume measure: r^2sin(theta)
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
        self._set_up_spherical_harmonics()
        self._calc_valence_wavefunctions()

    def _set_up_spherical_harmonics(self):
        max_total_angular_momentum = len(self.all_electron_atoms.channels)
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
    
    #Calculate exact dielectric matrix via AE wfs
    def calculate_exact_dielectric_matrix(self, omega = 0, temp=0, chemical_potential = 0, eta = 0.0001):
        raise NotImplementedError
        wfs = None #Something
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
            
            
    
    def _get_coulomb_matrix(self):
        raise NotImplementedError

    def dot(self, x1, x2):
        #Implements dot product with proper volume measure
        raise NotImplementedError
