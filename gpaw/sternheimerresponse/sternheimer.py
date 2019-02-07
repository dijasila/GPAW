from gpaw import GPAW
from functools import partial
from gpaw.matrix import matrix_matrix_multiply as mmm
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
from functools import reduce
from gpaw.utilities import pack, unpack


class SternheimerResponse:
    def __init__(self, filename):
        self.restart_filename = filename
        self.calc = GPAW(filename, txt=None)
        #TODO Check that all info is in file
        self.wfs = self.calc.wfs
    

        npts = self.wfs.kpt_u[0].psit.array.shape[1]
        #print("psit array shape", self.wfs.kpt_u[0].psit.array.shape)
        trial_potential = np.zeros((npts,npts))

        self.solve_sternheimer(trial_potential, 0)
    
        #test_wave = self.wfs.kpt_u[0].psit_nG[0]
        #self._apply_hamiltonian(test_wave, self.wfs.kpt_u[0])
        return

    def epsilon_response(self, qvectors, num_eigs):
        '''
        Input:
        qvectors: List of qvectors
        num_eigs: Number of eigenvalue-eigenvector pairs to find at each qvector
        
        
        Returns:
        A dictionary with key: qvector -> Value: List of num_eigs eigenvalue-eigenvector pairs
        '''
        #TODO only allow certain qvectors, namely only those that are given by a difference of two k-vectors in the BZ
        start_vectors = self._get_trial_vectors(num_eigs)
        solutions = self.epsilon_iteration(start_vectors, qvectors, num_eigs)
    
        return solutions

                


    def epsilon_iteration(self, start_vectors, qvectors, num_eigs):           
        solutions = {} #Dict from qvector (array) -> list of num_eigs eigenpairs
        
        for qvec in qvectors:
            found_eigenvectors = []
            for start_vec in start_vectors:                
                eigen_pair = self._sc_iteration(start_vec,
                                                self.epsilon_potential_from_deltawfs_wfs,
                                                qvec,
                                                self._epsilon_eigenpair_extractor,
                                                found_eigenvectors)

                if qvec not in solutions:
                    solutions[qvec] = [eigen_pair]
                else:
                    solutions[qvec].append(eigen_pair)
                    found_eigenvectors = [pair[1] for pair in solutions[qvec]]

                                

        return solutions                                        
                                


    def epsilon_potential_from_vector(self, start_vector):
        #TODO Smooth part + PAW corrections
        return np.diag(start_vector)

    def epsilon_potential_from_deltawfs_wfs(self, deltawfs_wfs):
                

        #delta_tilde_n = self.get_delta_tilde_n(deltawfs_wfs)
        fine_delta_tilde_n = self.get_fine_delta_tild_n(deltawfs_wfs)

       
        #Comp charges is dict: atom -> charge
        total_delta_comp_charge = self.get_compensation_charges(deltawfs_wfs)


        poisson_term = self.solve_poisson(fine_delta_tilde_n + total_delta_comp_charge)
        soft_term_G = self.transform_to_coarse_grid_operator(poisson_term)

        soft_term_R = self.wfs.pd2.ifft(soft_term_G)

        W_ap = self.get_charge_derivative_terms(poisson_term)                



        def apply_potential(wvf_G, q_index):
            wvf_R = self.wfs.pd.ifft(wvf_G)
            pt = self.wfs.pt

            c_ai = pt.integrate(wvf_G, q=q_index)
            
            for a, c_i in c_ai.items():
                W_ii = unpack(W_ap[a])
                c_i[:] = W_ii.dot(c_i)

                V1 = self.wfs.pd.fft(soft_term_R*wvf_R)
                pt.add(V1, c_ai, q=_index)
                return V1   


        return apply_potential




    def get_compensation_charges(self, deltawfs_wfs):        
        pt = self.wfs.pt
        p_overlaps = {}
        D_aii = {}
        for q_index in deltawfs_wfs:
            deltawfs = np.array([tup[0] for tup in deltawfs_wfs[q_index]])
            c_axi = pt.integrate(deltawfs, q=q_index)
            p_overlaps[q_index] = c_axi

            kpt = self.wfs.mykpts[q_index]
            for a, c_xi in c_axi.items():
                P_ni = kpt.P[a]
                if a in D_aii:
                    D_aii[a] += np.dot(P_ni.T.conj()*kpt.f_n, c_xi)
                else:
                    D_aii[a] = np.dot(P_ni.T.conj()*kpt.f_n, c_xi)
        self.D_aii = D_aii                                
                
        setups = self.wfs.setups
        Q_aL = {}
        for a in D_aii:
            setup = setups[a]
            D_ii = D_aii[a]
            Delta_iiL = setup.Delta_iiL
            Q_L = np.einsum("ij, ij", D_ii, Delta_iiL)
            Q_aL[a] = Q_L


                
        comp_charge_G = self.calc.density.pd3.zeros()
                
        self.calc.density.ghat.add(comp_charge_G, Q_aL)

        return comp_charge_G



    def get_fine_delta_tilde_n(self, deltawfs_wfs):
        #Interpolate delta psi
        #Use wavefunctions/pw.py
        #calc.density.pd2 (fine G grid)
        
        pd = self.wfs.pd
        pd2 = self.calc.density.pd2
        pd3 = self.calc.density.pd3
        delta_n_R = pd2.gd.zeros()
        for deltawf, wf in deltawfs_wfs:
            deltawf_R = pd.ifft(deltawf)
            wf_R = pd.ifft(wf)

            delta_n_R += deltawf_R*wf_R
            
        fine_delta_n_R, fine_delta_n_G = pd2.interpolate(delta_n_R, pd3)

        return fine_delta_n_G

    def solve_poisson(self, charge_G, pd):
        #charge(q)
        G2 = pd3.G2_qG[0].copy()
        G2[0] = 1.0
        
        return charge_G * 4 * pi/G2
        
    def transform_to_coarse_grid_operator(self, fine_grid_operator):        
        dens = self.calc.density
        coarse_potential = dens.pd2.zeros()

        dens.map23.add_to1(coarse_potential, fine_grid_operator)

        return coarse_potential
        





    def get_charge_derivative_terms(self, poisson_term):
        #For each atom get partial wave, smooth partial waves and PAW projectors
        #Calculate some stuff
        #return stuff*projection_matrix
        setups = self.wfs.setups

        coulomb_factors_aL = self.calc.density.ghat.integrate(poisson_term)
        W_ap = {}
        
        for a, glm_int_L in coulomb_factors_aL.items():
            W_ap[a] = setups[a].Delta_pL.dot(glm_int_L)

            

        for a, setup in enumerate(self.wfs.setups):
            D_p = pack(self.D_aii[a])
            W_ap[a] += setup.M_pp.dot(D_p)


        return W_ap



    def _epsilon_eigenpair_extractor(self, current_density, new_density):
        eigenvalue = np.sqrt(new_density.dot(new_density)/current_density.dot(current_density))
        new_density = new_density/np.sqrt(new_density.dot(new_density))
        return (eigenvalue, new_density)


    def _sc_iteration(self, start_vector, potential_generator_function, qvector, eigenpair_extractor, found_eigenvectors):
        error_threshold = 1e-10
        iter_max = 500
        current_vector = start_vector
        current_density = -np.ones(start_vector.shape)
        for num_iter in range(iter_max):
            potential = potential_generator_function(current_vector)

            deltawfs_wfs = self.solve_sternheimer(potential, qvector)

            new_density = self._density_from_wf_pairs(deltawfs_wfs)              
            new_density = self.orthogonalize(new_density, found_eigenvectors)
            
            eigenvalue, new_density = eigenpair_extractor(current_density, new_density)

            error = np.abs(1 - new_density.dot(current_density))
            
            if error < error_threshold:
                return (eigenvalue, new_density)


        raise ValueError("Self-consistent iteration did not converge. Final error was {}".format(error))
                


    def orthogonalize(out_vector, list_of_vectors):
        new_vec = out_vector
        for vec in list_of_vectors:
            new_vec -= vec*(out_vector.dot(vec))

        return new_vec




    def _density_from_wf_pairs(self, deltawfs_wfs):
        #TODO add PAW corrections, see 1D code
        return 2*np.real(reduce(lambda a, b : a + b[1].conj()*b[0], deltawfs_wfs, 0))
            
        



    def solve_sternheimer(self, potential, qvector):
        solution_pairs = {}

        for index, kpt in enumerate(self.wfs.mykpts):            
            solution_pairs[index] = []
            k_plus_q_index = self.wfs.kd.find_k_plus_q(qvector, [index])
            assert len(k_plus_q_index) == 1
            k_plus_q_index = k_plus_q_index[0]
            k_plus_q_object = self.wfs.mykpts[k_plus_q_index]


            #Do this because shape of wavefunctions is seemingly not constant across kpts
            #This is bad and should be avoided
            potential_at_k = self._get_potential_at_kpt(potential, kpt)


            for energy, psi in zip(kpt.eps_n, kpt.psit.array):
                linop = self._get_LHS_linear_operator(k_plus_q_object, energy)
                RHS = self._get_RHS(k_plus_q_object, psi, potential_at_k)
                

                delta_wf, info = bicgstab(linop, RHS)

                if info != 0:
                    print(f"bicgstab did not converge. Info: {info}")


                solution_pairs[index].append((delta_wf, psi))
                #Solve Sternheimer for given n, q
                #
            #print("Looking at kpt")
        return solution_pairs

    def _get_potential_at_kpt(self, potential, kpt):
        #TODO fix this, how does this couple to potential higher up?
        ##shape = kpt.psit.array.shape[1] ##This seems buggy psit.array.shape is not the same as psit.myshape
        shape = kpt.psit.myshape[1]
        return np.zeros((shape,shape))


    def _kindex_to_kvector(self, kindex):
        coordinates = self.wfs.kd.bzk_kc[kindex]
        

    def _get_LHS_linear_operator(self, kpt, energy):
        def mv(v):
            return self._apply_LHS_Sternheimer(v, kpt, energy)


        shape_tuple = (kpt.psit.array[0].shape[0], kpt.psit.array[0].shape[0])
        linop = LinearOperator(shape_tuple, matvec = mv)

        return linop


    def _apply_LHS_Sternheimer(self, wavefunction, kpt, energy):
        result = np.zeros_like(wavefunction)

        
        result = self._apply_valence_projector(wavefunction, kpt)
        
        result = result + self._apply_hamiltonian(wavefunction, kpt)
        
        result = result - (np.eye(result.shape[0])*energy).dot(wavefunction)

        return result

    def _apply_valence_projector(self, wavefunction, kpt):
        result = np.zeros_like(wavefunction)
        f_n = kpt.f_n/kpt.f_n.max()

        for index, wvf in enumerate(kpt.psit.array):
            result = result + wvf*(wvf.conj().dot(wavefunction)*f_n[index])


        return result


    def _apply_hamiltonian(self, wavefunction, kpt):
        kpt.psit.read_from_file()
        H_partial = partial(self.wfs.apply_pseudo_hamiltonian, kpt, self.calc.hamiltonian)
        psit = kpt.psit
        #print(len(psit))
        #print("Len of work", len(self.wfs.work_array))
        tmp = psit.new(buf=self.wfs.work_array)
        H = self.wfs.work_matrix_nn
        P2 = kpt.P.new()

        psit.matrix_elements(operator=H_partial, result=tmp, out=H,
                     symmetric=True, cc=True) ##!! Is cc=True correct here?
        self.calc.hamiltonian.dH(kpt.P, out = P2)
        
        a = mmm(1.0, kpt.P, "N", P2, "C", 1.0, H,  symmetric=True)
        #print("Type of a", type(a))
        self.calc.hamiltonian.xc.correct_hamiltonian_matrix(kpt, a.array)
        
        result = np.zeros_like(wavefunction)

        for index1, wvf1 in enumerate(psit.array):
            for index2, wvf2, in enumerate(psit.array):
                inner_prod = wvf2.conj().dot(wavefunction)
                matrix_element = a.array[index1, index2]
                result = result + wvf1*matrix_element*inner_prod

        return result
        
    def _get_RHS(self, kpt, psi, potential):
        RHS = np.zeros_like(kpt.psit.array[0])
        RHS = -(potential.dot(psi) - self._apply_valence_projector(potential.dot(psi), kpt))

        return RHS
        





if __name__=="__main__":
    filen = "test.gpw"
    import os
    if os.path.isfile(filen):
        print("Reading file")
        respObj = SternheimerResponse(filen)
    else:
        print("Generating new test file")
        from ase import Atoms
        from ase.build import bulk
        from gpaw import PW, FermiDirac
        a = 10
        c = 5
        d = 0.74
        #atoms = Atoms("H2", positions=([c-d/2, c, c], [c+d/2, c,c]),
        #       cell = (a,a,a))
        atoms = bulk("Si", "diamond", 5.43)
        calc = GPAW(mode=PW(100),
                xc ="PBE",
                kpts=(8,8,8),
                random=True,
                #symmetry=False,
                occupations=FermiDirac(0.01),
                txt = "test.out")
        
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        calc.write(filen, mode = "all")
 
        respObj = SternheimerResponse(filen)
        

