import numpy as np
from scipy.special import sph_harm, genlaguerre
import itertools
from functools import reduce
from scipy.stats import ortho_group 
from gpaw.utilities.clebschgordan import ClebschGordanCalculator
from scipy.sparse.linalg import bicgstab
import math
import scipy
def fermi_function(energy, temperature, chemical_potential):
    if temperature != 0:
        raise NotImplementedError
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
 #       print("Testing for angular number: {}".format(ang_num))
        chempot = (all_electron_atom.channels[ang_num].e_n[1] + all_electron_atom.channels[ang_num].e_n[0])/2
        self.chemical_potential = chempot
#        self.calculate_exact_chi_channel(ang_num, 0, chempot)

  #      self.get_valence_projector_r(all_electron_atom.channels[ang_num])

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
        self.bicgstab_tol = 1e-5
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


    def calculate_analytical_chi_channel(self, angular_number, num_levels, omega = 0, temp = 0, chemical_potential = None, eta = 0.0001):

        if chemical_potential is None:
            chemical_potential = self.chemical_potential


        wfs_ens  = self._get_analytical_levels(num_levels, angular_number)
        
        combos = list(itertools.product(wfs_ens, wfs_ens))
        
        def delta(pair):
            (wf1, en1), (wf2, en2) = pair
            if np.allclose(en1, en2):
                return 0

            d = np.outer(wf1.conj(), wf1).astype(np.complex128)
            d *= np.outer(wf2, wf2.conj())
            #assert not np.allclose(d, np.zeros_like(d))

            d *= fermi_function(en1, temp, chemical_potential) - fermi_function(en2, temp, chemical_potential)
            d *= 1/(en1 - en2 + omega + 1j*eta)

            return d
        chi_channel = reduce(lambda a, b : a + delta(b), combos, 0)

        angular_factor = self._get_chi_angular_factor(angular_number)
        chi_channel *= angular_factor
        return chi_channel


    def _get_analytical_levels(self, num_levels, angular_number):
        wfs_ens = []
        lags = []
        rgd = self.all_electron_atom.rgd
        assert angular_number == 0
        for n in range(1, num_levels+1):
            bohr_radius = 1
            reduced_radius = rgd.r_g*2/(bohr_radius*n)
            t = (2/(n*bohr_radius))**3
            prefactor = np.sqrt(t*math.factorial(n-angular_number-1)/(2*n*math.factorial(n+angular_number)))
            exp = np.exp(-reduced_radius/2)
            rl = reduced_radius**angular_number
            lag = genlaguerre(n-angular_number-1, 2*angular_number + 1, monic=False)(reduced_radius)


            radial_wf = prefactor*exp*rl*lag
            import matplotlib.pyplot as plt
            plt.plot(rgd.r_g, radial_wf, label = str(n))
            
            norm = np.sqrt(self.integrate(radial_wf**2))
            assert norm.ndim == 0
            #radial_wf = radial_wf/norm
            b = np.allclose(self.integrate(radial_wf**2), 1, atol =1e-3)
            if not b or True:
                #print("Norm of radial: ", self.integrate(radial_wf**2))
                print("Norm: ", scipy.integrate.cumtrapz(radial_wf**2*rgd.r_g**2, rgd.r_g)[-1])
            #assert b

            energy = -1/2*1/n**2

            wfs_ens.append((radial_wf, energy))

        plt.legend()
        plt.savefig("radii.png")
        for k1, (wf1, en1) in enumerate(wfs_ens):
            for k2, (wf2, en2) in enumerate(wfs_ens):
                if k1 != k2:
                    b = np.allclose(self.integrate(wf1*wf2), 0, atol = 1e-2)
                    #if not b:# or not b2:
                    #    print(self.integrate(wf1*wf2))

                    #assert b


        return wfs_ens


    def _get_chi_angular_factor(self, angular_number):
        if angular_number != 0:
            raise NotImplementedError
        return 1/(4*np.pi)

    def calculate_exact_chi_channel(self, angular_number, omega = 0, temp = 0, chemical_potential = None, eta = 0.001):
        '''
        This function returns chi (the response function) projected to the angular number (total angular momentum number) specified, i.e. returns chi(r,r')_l where

        chi(r, r')_l = \int d\Omega d\Omega' Y_{lm}(\Omega)^* chi(r\Omega, r'\Omega') Y_{lm}(\Omega')

        (independent of m due to spherical symmetry)

        '''
        if chemical_potential is None:
            chemical_potential = self.chemical_potential
        #TODO can be made more efficient by only summing over pairs where n_1 < n_2 and then using symmetries of chi to get the rest.
        radial_wfs, energies = self._get_radial_modes(angular_number)
        wf_en_tups = list(zip(radial_wfs, energies))

        combos = list(itertools.product(wf_en_tups, wf_en_tups))
        def delta(pair):
            (wf1, en1), (wf2, en2) = pair
            if np.allclose(en1, en2):
                return 0

            d = np.outer(wf1.conj(), wf1).astype(np.complex128)
            d *= np.outer(wf2, wf2.conj())
            assert not np.allclose(d, np.zeros_like(d))

            d *= fermi_function(en1, temp, chemical_potential) - fermi_function(en2, temp, chemical_potential)
            d *= 1/(en1 - en2 + omega + 1j*eta)

            return d
            
        ##TODO Need Clebsch Gordan factors, see eq. 5.1.2 in notes
        ##Problem with this because we do not sum over all excited states.
        ##These are not found in the aeaatom code: we need for all nl pairs and aeaatom only finds for occupied ones -- For any external perturbation with angular momentum LM there will be infinitely many terms in the sum over nlm that contribute. No, no quite, because one term has to be occupied and the other has to be unoccupied. 
        ##Is there another way to test the Sternheimer result?
        chi_channel = reduce(lambda a, b : a + delta(b), combos, 0)
        return chi_channel


    def _get_radial_modes(self, angular_number):
        channel = self.all_electron_atom.channels[angular_number]

        states = [channel.basis.expand(C) for C in channel.C_nb]
        energies = channel.e_n

        return states, energies
        


    def sternheimer_calculation(self, omega = 0, angular_momenta = [0], return_only_eigenvalues = True):
        #trial_potentials = self._get_random_trial_potentials(num_eig_vecs, len(self.radial_grid))
        

        #channel_number = min(3, len(all_electron_atom.channels)-1)
        start_vec = self.all_electron_atom.channels[0].phi_ng[0]
        total_angular_momenta = [(l, start_vec) for l in angular_momenta]
        vals, vecs = self._iterative_solve(omega, total_angular_momenta)
        print("Done")
        #From output of iterative solver reconstruct epsilon/chi or just return eigvals
        if return_only_eigenvalues:
            return np.sort(np.real(vals))

        else:
            response = self._construct_response(vals, vecs)
            return response
        
        
    def _get_random_trial_potentials(self, num_eig_vecs, size_of_response_matrix):
        vecs = ortho_group.rvs(size_of_response_matrix).T[:num_eig_vecs]

        vecs = [vec/np.sqrt(self.dot(vec.conj(), vec)) for vec in vecs]
        return vecs


    def _iterative_solve(self, omega, angular_momenta_trials):#total_angular_momenta): #num eigs, trial, resp ang
        '''
        Uses the power method (see wikipedia) plus the sternheimer equation to calculate num_eig_vecs of the eigenvector-eigenvalue pairs of the response matrix
        '''
        error_threshold = 1e-12
        ok_error_threshold = 1e-5
        max_iter = 100
        mixing = 0.5
        errors = []
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
                        print("Iteration number:       {}".format(num_iter), end = "\r")

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
                    errors.append(error)
                    current_trial = (1-mixing)*current_trial + mixing*new_trial
                    current_trial = current_trial/np.sqrt(self.dot(current_trial, current_trial))
                    assert np.allclose(self.dot(current_trial, current_trial), 1)




                    if error < error_threshold:
                        print("")
                        found_vals_vecs.append((current_eigval, new_trial))
                        np.savetxt("errors.txt", errors)
                        break
                    elif num_iter >= max_iter:
                        if error < ok_error_threshold:
                            found_vals_vecs.append((current_eigval, new_trial))
                            print("")
                            print("Warning: error in Sternheimer iterations was {} for eigenvalue {}".format(error, current_eigval))
                            break
                        else:
                            print("")
                            np.savetxt("errors.txt", errors)
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
        delta_n = 0
        for l, channel in enumerate(self.all_electron_atom.channels):

            gaussian_trial = self.get_trial_matrix(channel, trial)
                    
            hamiltonian = self.get_hamiltonian(channel)
            valence_projector = self.get_valence_projector(channel)

            assert valence_projector.ndim == 2

            conduction_projector = np.eye(valence_projector.shape[0]) - valence_projector
            assert conduction_projector.ndim == 2
            assert np.allclose(np.dot(conduction_projector, valence_projector), np.zeros_like(valence_projector))
            assert not np.allclose(conduction_projector, np.zeros_like(conduction_projector))




            alpha = 1

            all_wfs = channel.C_nb ##TODO only include valence here
            all_ens = channel.e_n ##TODO only include valence here
            wfs, ens = zip(*[(wf, en) for wf, en in zip(all_wfs, all_ens) if en < self.chemical_potential])
            
            ##TODO Should Combined ang momentum be in steps of 2
            sol_dict = {}
            for k, (wf, en) in enumerate(zip(wfs, ens)):
                for combined_ang_momentum in range(np.abs(response_l2 - l), response_l2 + l+1):
                    for M in range(-combined_ang_momentum, combined_ang_momentum + 1):
                        for m in range(-l, l+1):
                            LHS = (alpha*valence_projector + hamiltonian - en*np.eye(hamiltonian.shape[0]))

                            cg_coeff = self.cg_calculator.calculate(response_l2, 0, l, m, combined_ang_momentum, 0)
                            cg_coeff *= self.cg_calculator.calculate(response_l2, response_m, l, m, combined_ang_momentum, M)
                            cg_coeff *= np.sqrt((2*l+1)*(2*response_l2+1)/(4*np.pi*(2*combined_ang_momentum + 1)))

                            
                            RHS = -np.dot(conduction_projector, np.dot(gaussian_trial, wf))*cg_coeff
                            assert not np.allclose(gaussian_trial, np.zeros_like(gaussian_trial))
                            #print("RHS mean abs / LHS mean abs size: {}".format(np.mean(np.abs(RHS))/np.mean(np.abs(LHS))))
                            #assert not np.allclose(RHS, np.zeros_like(RHS))
                            delta_wf, info = bicgstab(LHS, RHS, atol = self.bicgstab_tol, tol = self.bicgstab_tol, maxiter = 10000)


                            
                            #delta_wf = np.dot(np.linalg.inv(LHS), RHS)
                            ##Problem seems to be here. LHS^-1 it just way too big
                            #info = 0
                            
                            if info != 0:
                                print("Solution should be: {}".format(np.dot(np.linalg.inv(LHS), RHS)))
                                raise ValueError("Bicgstab did not converge. Info: {}".format(info))

                            b = np.allclose(np.dot(LHS, delta_wf), RHS, atol = 1e-2)
                            if not b:
                                print("Error: {}".format(np.max(np.abs(np.dot(LHS, delta_wf) - RHS))))
                            assert b
                            
                            sol_dict[(k, l, m, combined_ang_momentum, M)] = (delta_wf, wf)

                ##For every total momentum LM that can be reached when combining channel ang mom and resp ang mom:
                ##Solve sternheimer eq
                ##See notes 4.3.18
            

            assert len(sol_dict.keys()) != 0

            for k_l_m_L_M in sol_dict:
                k, l, m, L, M = k_l_m_L_M
                delta_wf, wf = sol_dict[k_l_m_L_M]
                wf = channel.basis.expand(wf)
                delta_wf = channel.basis.expand(delta_wf)
            
                #cg_coeff = self.cg_calculator.calculate(l, -m, L, M, response_l2, response_m)
                delta_n += 2*np.real(wf.conj()*delta_wf*cg_coeff)
            

        assert not np.allclose(self.dot(delta_n, delta_n), 0)
        return delta_n



    def get_trial_matrix(self, channel, trial):
        size = len(channel.basis)
        trial_matrix = np.zeros((size, size))

        for k1, g1 in enumerate(channel.basis.basis_bg):
            for k2, g2 in enumerate(channel.basis.basis_bg):
                trial_matrix[k1,k2] = self.integrate(g1*g2*trial)

        assert not np.allclose(trial_matrix, np.zeros_like(trial_matrix))
        return trial_matrix

    def get_hamiltonian(self, channel):
        atom_vr = self.all_electron_atom.vr_sg[channel.s]
        hamiltonian = channel.basis.calculate_potential_matrix(atom_vr)
        hamiltonian += channel.basis.T_bb
        assert np.allclose(hamiltonian.conj().T, hamiltonian)
        return hamiltonian
                

    def get_valence_projector(self, channel):
        '''
        Returns the operator P_v
        '''
        Pv = 0

        wfs = channel.C_nb
        ens = channel.e_n

        for wf, en in zip(wfs, ens):
            if en > self.chemical_potential:
                break
            Pv += np.outer(wf, wf.conj())


        assert Pv.shape == (22,22)
        assert np.allclose(np.dot(Pv, Pv), Pv)
        return Pv


    def dot(self, x1, x2, n = 0):
        #Implements dot product with proper volume measure
        gd = self.all_electron_atom.rgd
        if x1.ndim == 1 and x2.ndim == 1:
            return gd.integrate(x1.conj()*x2, n = n)/(4*np.pi)
        elif x1.ndim == 2 and x2.ndim == 1:
            return gd.integrate(x1*x2, n = n)/(4*np.pi)
        else:
            raise NotImplementedError
    def integrate(self, function, n = 0):
        gd = self.all_electron_atom.rgd

        return gd.integrate(function, n = n)/(4*np.pi)

    def delta_function_norm(self):
        raise NotImplementedError











