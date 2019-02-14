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


    def calculate(self, qvectors, num_eigs):
        qvector = qvectors[0]
        
        deltapsi_qnG, eigval = self.initial_guess()

        

        error_threshold = 1e-9
        error = 100
        num_iter = 200
        def Precondition(deltapsi_qnG):
            return deltapsi_qnG
        stepsize = 0.1

        for _ in range(num_iter):
            #Calculate residual
            new_deltapsi_qnG = self.apply_K(deltapsi_qnG, qvector)
            residual_qnG = {q : new_deltapsi_qnG[q] - eigval*deltapsi_nG for q, deltapsi_nG in deltapsi_qnG.itesm()}



            error = self.calculate_error(residual_qnG)
            if error < error_threshold:
                return eigval, deltapsi_qnG




            
            #Precondition and step
            deltapsi_qnG = {q : deltapsi_nG + stepsize*Precondition(residual_qnG[q]) for q, deltapsi_nG in deltapsi_qnG.items()}
            
            
            #Orthonormalize ##Replace this with Gram-Schmidt if solving for more eigenvectors
            norm = np.sqrt(self.inner_product(deltapsi_qnG, deltapsi_qnG))
            deltapsi_qnG = {q : deltapsi_nG/norm for q, deltapsi_nG in deltapsi_qnG.items()}


            ##Replace this with subspace diagonalization if solving for more eigenvectors concurrently
            eigval = self.inner_product(deltapsi_qnG, self.apply_K(deltapsi_qnG))
        raise ValueError("Calculation did not converge")


    def initial_guess(self):
        deltapsi_qnG = {}
        
        for index, kpt in enumerate(self.wfs.mykpts):
            ##Replace this with better start guess probably
            deltapsi_qnG[index] = np.ones(kpt.psit.array.shape)

            
        ##Replace this with guess dependent on other guess/found values to speed up convergence
        eigval = 2

        return deltapsi_qnG, eigval

    def calculate_error(self, residual_qnG):
        error = 0

        for q, residual_nG in residual_qnG.items():
            for residual_G in residual_nG:
                error += np.linalg.norm(residual_G)

        return error
            


    def inner_product(self, deltapsi1_qnG, deltapsi2_qnG):
        fine_delta_tilde_n1_G = self.get_fine_delta_tilde_n(deltapsi1_qnG)
        fine_delta_tilde_n2_G = self.get_fine_delta_tilde_n(deltapsi2_qnG)

        DeltaD1_aii = self.get_density_matrix(delta_psi1_nG)
        DeltaD2_aii = self.get_density_matrix(delta_psi2_nG)


        comp_charge1_G = self.get_compensation_charges(deltapsi1_qnG, DeltaD1_aii)
        comp_charge2_G = self.get_compensation_charges(deltapsi2_qnG, DeltaD2_aii)

        poisson_G = self.solve_poisson(fine_delta_tilde_n1_G + comp_charge1_G)

        P12 = (fine_delta_tilde_n2_G + comp_charge2_G).dot(poisson_G)


        #C_app = {a : 2*setup.M_pp for a, setup in enumerate(self.wfs.setups)}

        Pd = 0
        for a, DeltaD1_ii in DeltaD1_aii.items():
            DeltaD1_p = pack(DeltaD1_ii)
            DeltaD2_p = pack(DeltaD2_aii[a])
            Pd = DeltaD1_p.dot((2*self.wfs.setups[a].M_pp).dot(DeltaD2_p))

        return P12 + Pd


    def get_density_matrix(self, deltapsi_qnG):
        D_aii = {}
        for q_index, deltapsi_nG in deltapsi_qnG.items():
            c_axi = pt.integrate(deltapsi_nG, q=q_index)
            p_overlaps[q_index] = c_axi

            kpt = self.wfs.mykpts[q_index]
            for a, c_xi in c_axi.items():
                P_ni = kpt.P[a]
                if a in D_aii:
                    D_aii[a] += np.dot(P_ni.T.conj()*kpt.f_n, c_xi)
                else:
                    D_aii[a] = np.dot(P_ni.T.conj()*kpt.f_n, c_xi)
        return D_aii
        


    def apply_K(self, deltapsi_qnG, qvector):
        potential_function = self.epsilon_potential(deltapsi_qnG)
        
        new_deltapsi_qnG = self.solve_sternheimer(potential_function, qvector)

        return {q: -val for q, vel in new_deltapsi_qnG.items()}

            

    def epsilon_potential(self, deltapsi_qnG):
        #Should add qvector to args to construct potential suitable for individual kpt?

        #delta_tilde_n = self.get_delta_tilde_n(deltawfs_wfs)
        fine_delta_tilde_n = self.get_fine_delta_tilde_n(deltapsi_qnG)


       
        #Comp charges is dict: atom -> charge
        DeltaD_aii = self.get_density_matrix(deltapsi_qnG)
        total_delta_comp_charge = self.get_compensation_charges(deltapsi_qnG, DeltaD_aii)


        poisson_term = self.solve_poisson(fine_delta_tilde_n + total_delta_comp_charge)
        soft_term_G = self.transform_to_coarse_grid_operator(poisson_term)
        soft_term_R = self.wfs.pd2.ifft(soft_term_G)



        W_ap = self.get_charge_derivative_terms(poisson_term, DeltaD_aii)



        def apply_potential(wvf_G, q_index):
            wvf_R = self.wfs.pd.ifft(wvf_G)
            pt = self.wfs.pt

            c_ai = pt.integrate(wvf_G, q=q_index)
            
            for a, c_i in c_ai.items():
                W_ii = unpack(W_ap[a])
                c_i[:] = W_ii.dot(c_i)

            V1 = self.wfs.pd.fft(soft_term_R*wvf_R)
            pt.add(V1, c_ai, q=q_index)
            return V1   


        return apply_potential




    def get_compensation_charges(self, deltapsi_qnG, DeltaD_aii):                        
        setups = self.wfs.setups
        Q_aL = {}
        for a in D_aii:
            setup = setups[a]
            DeltaD_ii = DeltaD_aii[a]
            Delta_iiL = setup.Delta_iiL
            Q_L = np.einsum("ij, ij", DeltaD_ii, Delta_iiL)
            Q_aL[a] = Q_L


                
        comp_charge_G = self.calc.density.pd3.zeros()
                
        self.calc.density.ghat.add(comp_charge_G, Q_aL)

        return comp_charge_G



    def get_fine_delta_tilde_n(self, deltapsi_qnG):
        #Interpolate delta psi
        #Use wavefunctions/pw.py
        #calc.density.pd2 (fine G grid)
        
        pd = self.wfs.pd
        pd2 = self.calc.density.pd2
        pd3 = self.calc.density.pd3
        delta_n_R = pd2.gd.zeros()
        for q_index, sol_list in deltawfs_wfs.items():
            for deltawf, wf in sol_list:
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
        





    def get_charge_derivative_terms(self, poisson_term, DeltaD_aii):
        #For each atom get partial wave, smooth partial waves and PAW projectors
        #Calculate some stuff
        #return stuff*projection_matrix
        setups = self.wfs.setups

        coulomb_factors_aL = self.calc.density.ghat.integrate(poisson_term)
        W_ap = {}
        
        for a, glm_int_L in coulomb_factors_aL.items():
            W_ap[a] = setups[a].Delta_pL.dot(glm_int_L)

                

        for a, setup in enumerate(self.wfs.setups):
            D_p = pack(self.DeltaD_aii[a])
            W_ap[a] += (2*setup.M_pp).dot(D_p)

        return W_ap
                



    def solve_sternheimer(self, apply_potential, qvector):
        deltapsi_qnG = {}

        for index, kpt in enumerate(self.wfs.mykpts):
            number_of_valence_states = kpt.psit.array.shape[0]
            deltapsi_qnG[index] = self.wfs.pd.zeros(x=(number_of_valence_states,), q=index)


            #Get kpt object for k+q
            k_plus_q_index = self.wfs.kd.find_k_plus_q(qvector, [index])
            assert len(k_plus_q_index) == 1
            k_plus_q_object = self.wfs.mykpts[k_plus_q_index[0]]



            for state_index, (energy, psi) in enumerate(zip(kpt.eps_n, kpt.psit.array)):
                linop = self._get_LHS_linear_operator(k_plus_q_object, energy)
                RHS = self._get_RHS(k_plus_q_object, psi, apply_potential)
                

                deltapsi_G, info = bicgstab(linop, RHS)

                if info != 0:
                    print("bicgstab did not converge. Info: {}".format(info))


                deltapsi_qnG[index][state_index] = deltapsi_G

        return deltapsi_qnG


    def _get_LHS_linear_operator(self, kpt, energy):
        def mv(v):
            return self._apply_LHS_Sternheimer(v, kpt, energy)


        shape_tuple = (kpt.psit.array[0].shape[0], kpt.psit.array[0].shape[0])
        linop = LinearOperator(shape_tuple, matvec = mv)

        return linop


    def _apply_LHS_Sternheimer(self, deltapsi_G, kpt, energy):
        result_G = np.zeros_like(deltapsi_G)
        
        alpha = 1
        
        result_G = alpha*self._apply_valence_projector(deltapsi_G, kpt)
        
        result_G = result_G + self._apply_hamiltonian(deltapsi_G, kpt)
        
        result_G = result_G - energy*deltapsi_G

        return result_G

    def _apply_valence_projector(self, deltapsi_G, kpt):
        result_G = np.zeros_like(deltapsi_G)
        f_n = kpt.f_n/kpt.f_n.max() #TODO ask JJ
        num_occ = f_n.sum()

        for index, psi_G in enumerate(kpt.psit.array):
            if index >= num_occ:
                break
            result_G = result_G + psi_G*(psi_G.conj().dot(psi_G)*f_n[index])


        return result_G


    def _apply_hamiltonian(self, deltapsi_G, kpt):
        #This function mimics this approach in eigensolver.subspace_diagonalize
        kpt.psit.read_from_file() #This is to ensure everything is initialized

        H_partial = partial(self.wfs.apply_pseudo_hamiltonian, kpt, self.calc.hamiltonian)
        psit = kpt.psit

        tmp = psit.new(buf=self.wfs.work_array)
        H = self.wfs.work_matrix_nn
        P2 = kpt.P.new()

        psit.matrix_elements(operator=H_partial, result=tmp, out=H,
                     symmetric=True, cc=True) ##!! Is cc=True correct here?

        #Add corrections (? Ask JJ)
        self.calc.hamiltonian.dH(kpt.P, out = P2)
        
        mmm(1.0, kpt.P, "N", P2, "C", 1.0, H,  symmetric=True) #H is out-variable

        self.calc.hamiltonian.xc.correct_hamiltonian_matrix(kpt, a.array)
        
        result_G = np.zeros_like(deltapsi_G)

        for index1, wvf1_G in enumerate(psit.array):
            for index2, wvf2_G, in enumerate(psit.array):
                inner_prod = wvf2_G.conj().dot(deltapsi_G)
                matrix_element = H.array[index1, index2]
                result_G = result_G + wvf1_G*matrix_element*inner_prod

        return result_G
        
    def _get_RHS(self, kpt, psi_G, apply_potential):
        RHS_G = np.zeros_like(kpt.psit.array[0])
        v_psi_G = apply_potential(psi_G)
        RHS_G = -(v_psi_G - self._apply_valence_projector(v_psi_G, kpt))

        return RHS_G
        





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
        

