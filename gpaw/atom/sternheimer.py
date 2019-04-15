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
    def __init__(self, all_electron_atom, nspins, calc_epsilon=False):
        if nspins != 1:
            raise NotImplementedError
        self.calc_epsilon = calc_epsilon
        self.all_electron_atom = all_electron_atom #Main calculation object in aeatom.py
        self.nspins = nspins
        self.initialize()
        self.log = all_electron_atom.log
        ang_num = len(all_electron_atom.channels)-1
        chempot = (all_electron_atom.channels[ang_num].e_n[1] + all_electron_atom.channels[ang_num].e_n[0])/2
        self.chemical_potential = chempot


    def initialize(self):        
        self.cg_calculator = ClebschGordanCalculator()
        self.bicgstab_tol = 1e-5
        
   


    def calculate_analytical_chi_channel(self, angular_number, num_levels, omega = 0, temp = 0, chemical_potential = None, eta = 0.0001):
        '''
        Calculate chi_l using the analytical solutions for Hydrogen
        '''

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
            d *= fermi_function(en1, temp, chemical_potential) - fermi_function(en2, temp, chemical_potential)
            d *= 1/(en1 - en2 + omega + 1j*eta)

            return d
        #mapped = map(lambda x : delta(x), combos)
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
            energy = -1/2*1/n**2

            wfs_ens.append((radial_wf, energy))

        return wfs_ens


    def _get_chi_angular_factor(self, angular_number):
        if angular_number != 0:
            raise NotImplementedError
        return 1/(4*np.pi)

            

    def sternheimer_calculation(self, omega = 0, angular_momenta=[0], num_eigs=1, return_only_eigenvalues=True):
        start_vecs_ln = self._get_random_trial_vectors(num_eigs, angular_momenta)
        #start_vecs_ln = {l: [self.all_electron_atom.channels[l].phi_ng[k] for k in range(num_eigs)] for l in angular_momenta}
        found_vals_vecs_ln = self._iterative_solve(omega, start_vecs_ln, num_eigs)
        self.log("Sternheimer calculation finished")
        l = 0
        vals = [val_vec[0] for val_vec in found_vals_vecs_ln[l]]
        vecs = [val_vec[1] for val_vec in found_vals_vecs_ln[l]]

        self.chi_vals = vals
        if return_only_eigenvalues:
            return np.sort(np.real(vals))

        else:
            if self.calc_epsilon:
                chi = self._construct_response(vals, vecs)
                #Use Laplace expansion for Coulomb potential
                laplace_coulomb = self.get_laplace_coulomb()
                rgd = self.all_electron_atom.rgd
                drs = np.diag(rgd.dr_g)
                integral = laplace_coulomb.dot(drs.dot(chi))*4*np.pi
                eps = np.linalg.inv(np.eye(chi.shape[0]).dot(drs) + integral.dot(drs))
                vals, vecs = np.linalg.eig(eps)
                return np.sort(np.real(vals)), vecs.T
            return vals, vecs
            #response = self._construct_response(vals, vecs)
            #return response

    
    def _get_random_trial_vectors(self, num_eigs, angular_momenta):
        start_vecs_ln = {}

        for l in angular_momenta:
            size = len(self.all_electron_atom.channels[l].phi_ng[0])
            start_vecs_ln[l] = []
            for n in range(num_eigs):
                start = np.sin((n+1)*self.all_electron_atom.rgd.r_g*np.pi/max(self.all_electron_atom.rgd.r_g))
                start = start/np.linalg.norm(start)
                start_vecs_ln[l].append(start)

        return start_vecs_ln


    def get_laplace_coulomb(self):
        r_g = self.all_electron_atom.rgd.r_g
        nrs = len(r_g)

        cutoff = min(self.all_electron_atom.rgd.dr_g)
        lapcoul = np.array([
            [1/max(r, rp, cutoff) for rp in r_g]
            for r in r_g])

        return lapcoul
                

        
    def _get_random_trial_potentials(self, num_eig_vecs, size_of_response_matrix):
        vecs = ortho_group.rvs(size_of_response_matrix).T[:num_eig_vecs]
        vecs = [vec/np.sqrt(self.dot(vec.conj(), vec)) for vec in vecs]
        return vecs


    def _iterative_solve(self, omega, start_vecs_ln, num_eigs):#total_angular_momenta): #num eigs, trial, resp ang
        '''
        Uses the power method plus the sternheimer equation to calculate num_eig_vecs of the eigenvector-eigenvalue pairs of the response matrix
        '''
        error_threshold = 1e-12
        ok_error_threshold = 1e-5
        max_iter = 100
        mixing = 0.5
        errors = []
        found_vals_vecs_ln = {}
        for l2, trial_vectors in start_vecs_ln.items():
            found_vals_vecs = []
            for trial_vector in trial_vectors:
                current_trial = trial_vector
                current_eigval = 1
                num_iter = 0

                while True:
                    ##Main calculation
                
                    self.log("Iteration number:       {}".format(num_iter), end = "\r")

                    num_iter += 1
                    
                    new_trial = self._solve_sternheimer(omega, current_trial, l2)
                
                
                    #The eigenvalue estimate here is negative because eigvals of chi are negative
                    current_eigval = -np.sqrt(self.dot(new_trial, new_trial)/self.dot(current_trial, current_trial))
                    new_trial = -new_trial/np.sqrt(self.dot(new_trial, new_trial))
                
                
                    ##Check for convergence
                    error = np.abs(1 - self.dot(new_trial, current_trial))
                    errors.append(error)
                    current_trial = (1-mixing)*current_trial + mixing*new_trial
                    current_trial = self._orthonormalize(current_trial, [val_vec[1] for val_vec in found_vals_vecs])
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
                            raise ValueError(f"Sternheimer iteration did not converge. Final error: {error}")
            found_vals_vecs_ln[l2] = found_vals_vecs
        #if len(found_vals_vecs) != len(angular_momenta_trials):
            #raise ValueError("Not enough eigenpairs was found")

        return found_vals_vecs_ln


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
        response_l2 = response_angular_momentum
        delta_n = 0
        for l, channel in enumerate(self.all_electron_atom.channels):

            # if self.calc_epsilon:
            #     rgd = self.all_electron_atom.rgd
            #     coulomb_matrix = rgd.r_g.copy()
            #     coulomb_matrix[0] = coulomb_matrix[1]
            #     coulomb_matrix = 1/coulomb_matrix
            #     trial = trial
            #     raise NotImplementedError

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
                            cg_coeff *= self.cg_calculator.calculate(response_l2, 0, l, m, combined_ang_momentum, M)
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
            

        #assert not np.allclose(self.dot(delta_n, delta_n), 0)
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











