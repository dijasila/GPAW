import numpy as np
from scipy.special import sph_harm
import itertools
from functools import reduce
from scipy.stats import ortho_group 
from gpaw.utilities.clebschgordan import ClebschGordanCalculator
from scipy.sparse.linalg import bicgstab

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
        self.chemical_potential = chempot
        self.calculate_exact_chi_channel(ang_num, 0, chempot)

        self.get_valence_projector_r(all_electron_atom.channels[ang_num])

        #import matplotlib.pyplot as plt
        #plt.plot(all_electron_atom.channels[ang_num].basis.basis_bg[0])
        #plt.show()
        

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
        self.cg_calculator = ClebschGordanCalculator()
        self.bicgstab_tol = 1e-8
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
            wf_en1, wf_en2 = pair
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
        


    def sternheimer_calculation(self, omega = 0, num_eig_vecs = 1, return_only_eigenvalues = True):
        #trial_potentials = self._get_random_trial_potentials(num_eig_vecs, len(self.radial_grid))
        

        #channel_number = min(3, len(all_electron_atom.channels)-1)
        total_angular_momenta= [(0, np.ones(self.radial_grid.shape))]
        vals, vecs = self._iterative_solve(omega, total_angular_momenta)
        print("Done")
        #From output of iterative solver reconstruct epsilon/chi or just return eigvals
        if return_only_eigenvalues:
            return np.sort(np.real(vals))

        else:
            response = self._construct_response(vals, vecs)
            return response
        
        
    def _get_random_trial_potentials(self, num_eig_vecs, size_of_response_matrix):
        vecs = ortho_group.rvs(size_of_response_matrix)[:num_eig_vecs]

        vecs = [vec/np.sqrt(self.dot(vec.conj(), vec)) for vec in vecs]
        return vecs


    def _iterative_solve(self, omega, angular_momenta_trials):#total_angular_momenta): #num eigs, trial, resp ang
        '''
        Uses the power method (see wikipedia) plus the sternheimer equation to calculate num_eig_vecs of the eigenvector-eigenvalue pairs of the response matrix
        '''
        error_threshold = 1e-8
        ok_error_threshold = 1e-5
        max_iter = 100

        
        found_vals_vecs = []
        for l2, trial in angular_momenta_trials:
   #     for l2 in total_angular_momenta:
            for m in [0]:#range(-l2, l2+1):
                        
                current_trial = trial
                current_eigval = 1
                num_iter = 0

    
                while True:
                ##Main calculation
                    if (num_iter % 10) == 0 or True:
                        print("Iteration number:       {}".format(num_iter))
                    num_iter += 1
                    
                    #trial = self._orthonormalize(trial, list(map(lambda a : a[1], found_vals_vecs)))
                    
                    #Not necessary, use aeatom module methods
                    #gaussian_trial = self._get_gaussian_potential(trial, channel_number)
                    
                    new_trial = self._solve_sternheimer(omega, current_trial, (l2, m))
                    
                    #new_trial = 2*np.real(np.sum([t[0].conj()*t[1] for t in delta_wfs_wfs]))
                    
                    #Minus because eigvals of chi are negative
                    current_eigval = -np.sqrt(self.dot(new_trial, new_trial)/self.dot(current_trial, current_trial))
                
                    new_trial = -new_trial/np.sqrt(self.dot(new_trial, new_trial))


                    ##Check for convergence
                    error = np.abs(1 - self.dot(new_trial, current_trial))

                    current_trial = new_trial





                    if error < error_threshold:
                        found_vals_vecs.append((current_eigval, new_trial))
                        break
                    elif num_iter >= max_iter:
                        if error < ok_error_threshold:
                            found_vals_vecs.append((current_eigval, new_trial))
                            print("Warning: error in Sternheimer iterations was {} for eigenvalue {}".format(error, current_eigval))
                            break
                        else:
                            raise ValueError("Sternheimer iteration did not converge")
                            
        if len(found_vals_vecs) != len(angular_momenta_trials):
            raise ValueError("Not enough eigenpairs was found")

        return zip(*found_vals_vecs)


    def _construct_response(self, vals, vecs):
        response = 0
        for val, vec in zip(vals,vecs):
            response += val*np.outer(vec, vec.conj())

        return response


    def _orthonormalize(self, vector, ortho_space):
        for other_vec in ortho_space:
            vector = vector - self.dot(other_vec.conj(), vector)*other_vec

        return vector


    def _get_gaussian_potential(self, trial, channel_number):
        gaussian_basis = self.all_electron_atom.channels[channel_number].basis.basis_bg

        size = len(gaussian_basis)
        gaussian_pot = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                gaussian_pot[i,j] = self.integrate(gaussian_basis[i].conj()*trial*gaussian_basis[j])

        return gaussian_pot
             

    def _solve_sternheimer(self, omega, trial, response_angular_momentum):
        '''
        Returns delta wfs and wfs in position basis
        '''
        assert np.real(omega) == 0
        response_l2, response_m = response_angular_momentum

        for l, channel in enumerate(self.all_electron_atom.channels):

            #Need to solve for u in psi = ru. AeAtom module is set up this way.
            #Do it same way to reuse as much as possible from AeAtom
            trial_r = trial*self.all_electron_atom.rgd.r_g

#            gaussian_trial_r = np.array([[self.integrate(trial_r*g1)*g2 for g2 in channel.basis.basis_bg] for g1 in channel.basis.basis_bg])

            gaussian_trial_r = np.zeros((len(channel.basis), len(channel.basis)))
            for k1, g1 in enumerate(channel.basis.basis_bg):
                for k2, g2 in enumerate(channel.basis.basis_bg):
                    k1k2_val = 0
                    for g3 in channel.basis.basis_bg:
                        k1k2_val += self.integrate(trial_r*g3*g2*g1)
                    assert not np.allclose(k1k2_val, 0)
                    gaussian_trial_r[k1,k2] = k1k2_val
                    

            hamiltonian = channel.basis.calculate_potential_matrix(trial_r)
            atom_vr = self.all_electron_atom.vr_sg[channel.s]
            hamiltonian += channel.basis.calculate_potential_matrix(atom_vr)
            hamiltonian += channel.basis.T_bb

            valence_projector_r = self.get_valence_projector_r(channel)
            assert valence_projector_r.ndim == 2

            conduction_projector_r = np.eye(valence_projector_r.shape[0]) - valence_projector_r
            assert conduction_projector_r.ndim == 2
            assert np.allclose(np.dot(conduction_projector_r, valence_projector_r), np.zeros_like(valence_projector_r))

            alpha = 1


            wfs = channel.C_nb
            ens = channel.e_n


            sol_dict = {}
            for k, wf_en in enumerate(zip(wfs, ens)):
                wf, en = wf_en
                for combined_ang_momentum in range(np.abs(response_l2 - l), response_l2 + l+1):
                    for M in range(-combined_ang_momentum, combined_ang_momentum + 1):
                        for m in range(-l, l+1):
                            LHS = (alpha*valence_projector_r + hamiltonian - en)#*self.all_electron_atom.rgd.r_g**2

                            cg_coeff = self.cg_calculator.calculate(response_l2, 0, l, m, combined_ang_momentum, 0)
                            cg_coeff *= self.cg_calculator.calculate(response_l2, response_m, l, m, combined_ang_momentum, M)
                            assert not np.allclose(cg_coeff, 0)
                            RHS = -np.dot(conduction_projector_r, np.dot(gaussian_trial_r, wf))*cg_coeff
                            assert not np.allclose(gaussian_trial_r, np.zeros_like(gaussian_trial_r))
                            assert not np.allclose(RHS, np.zeros_like(RHS))
                            delta_wf, info = bicgstab(LHS, RHS, atol = self.bicgstab_tol, tol = self.bicgstab_tol)
                            
                            if info != 0:
                                raise ValueError("Bicgstab did not converge. Info: {}".format(info))
                            assert np.allclose(np.dot(LHS, delta_wf), RHS, atol = self.bicgstab_tol)
                            
                            sol_dict[(k, l, m, combined_ang_momentum, M)] = (delta_wf, wf)
                            
                ##For every total momentum LM that can be reached when combining channel ang mom and resp ang mom:
                ##Solve sternheimer eq
                ##See notes 4.3.18
            

        delta_n = 0
        assert len(sol_dict.keys()) != 0
        all_zero = True
        for k_l_m_L_M in sol_dict:
            k, l, m, L, M = k_l_m_L_M
            delta_wf, wf = sol_dict[k_l_m_L_M]
            wf = channel.basis.expand(wf)
            delta_wf = channel.basis.expand(delta_wf)
            
            cg_coeff = self.cg_calculator.calculate(l, -m, L, M, response_l2, response_m)
            if not np.allclose(cg_coeff, 0):
                all_zero = False

                
            delta_n += wf.conj()*delta_wf*cg_coeff
        assert not all_zero
        assert not np.allclose(self.dot(delta_n, delta_n), 0)
        return delta_n



    def get_valence_projector_r(self, channel):
        '''
        Returns the operator P_v \times r
        '''

        Pv = 0

        for vec, en in zip(channel.C_nb, channel.e_n):
            if en > self.chemical_potential:
                break
            Pv += np.outer(vec, vec.conj())

        assert np.allclose(np.dot(Pv, Pv), Pv)

        return Pv


    def get_valence_projector(self):
        raise NotImplementedError


    def dot(self, x1, x2):
        #Implements dot product with proper volume measure
        gd = self.all_electron_atom.rgd

        return gd.integrate(x1.conj()*x2, n = 0)
        
    def integrate(self, function):
        gd = self.all_electron_atom.rgd

        return gd.integrate(function, n = 0)

    def delta_function_norm(self):
        raise NotImplementedError











