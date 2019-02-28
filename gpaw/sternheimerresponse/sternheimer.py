from gpaw import GPAW
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
from gpaw.utilities import pack, unpack


class SternheimerResponse:
    def __init__(self, filename):
        self.restart_filename = filename
        self.calc = GPAW(filename, txt=None)
        #TODO Check that all info is in file
        self.wfs = self.calc.wfs
        self.calc.initialize_positions()
        self.c_qani = {}
        self.c_qnai = {}


        ind1 = self.wfs.mykpts[1].q
        ind0 = self.wfs.mykpts[0].q
        print(f"My k vecs: {[self.wfs.kd.bzk_kc[ind.q] for ind in self.wfs.mykpts]}")
        qvector = self.wfs.kd.bzk_kc[ind1] - self.wfs.kd.bzk_kc[ind0]

        #qvector = np.array([0,0,0])
        #self.calculate([qvector], 1)
        self.deltapsi_qnG = None
        #self.powercalculate([qvector], 1)


    def powercalculate(self, qvectors, num_eigs):
        print("POWERCALCULATE")
        qvector = qvectors[0]

        deltapsi_qnG, eigval = self.initial_guess()
        if self.deltapsi_qnG is not None:
            print("Using previously calculated deltapsi")
            deltapsi_qnG = self.deltapsi_qnG

        error_threshold = 1e-9
        error = 100

        max_iter = 10
        new_norm = 0
        old_norm = 0
        for niter in range(max_iter):
            print(f"Iteration number: {niter}")
            print(f"Eigenvalue: {eigval}. Error: {error}")

            new_deltapsi_qnG = self.apply_K(deltapsi_qnG, qvector)
            new_norm = self.inner_product(new_deltapsi_qnG, new_deltapsi_qnG)
            eigval = np.sqrt(new_norm)
            new_deltapsi_qnG = {q: deltapsi_qnG[q] + 0.4*val/np.sqrt(new_norm) for q, val in new_deltapsi_qnG.items()}
            new_norm = self.inner_product(new_deltapsi_qnG, new_deltapsi_qnG)
            new_deltapsi_qnG = {q: val/np.sqrt(new_norm) for q, val in new_deltapsi_qnG.items()}
            error = np.abs(1 - self.inner_product(deltapsi_qnG, new_deltapsi_qnG))

            if error < error_threshold:
                print(f"Converged. Value: {eigval}")
                self.deltapsi_qnG = deltapsi_qnG
                return

            deltapsi_qnG = new_deltapsi_qnG

        self.deltapsi_qnG = deltapsi_qnG

            
        print("Not converged")
        return


            
            
        

    def calculate(self, qvectors, num_eigs):
        qvector = qvectors[0]
        
        deltapsi_qnG1, eigval1 = self.initial_guess()
        #deltapsi_qnG2, eigval2 = self.initial_guess()
        eigvals = [eigval1]#, eigval2]
        deltapsis = [deltapsi_qnG1]#, deltapsi_qnG2]

        error_threshold = 1e-9
        error = 100
        num_iter = 100
        def Precondition(deltapsi_qnG):
            return deltapsi_qnG
        stepsize = 0.2

        for niter in range(num_iter):
            print(f"Iteration number: {niter}")
            print(f"Eigenvalues: {eigvals}")#. Error: {error}")
            if (niter+1) % 10 == 0:
                stepsize /= 1.1
            #Calculate residual
            for k, deltapsi_qnG in enumerate(deltapsis):
                eigval = eigvals[k]
                new_deltapsi_qnG = self.apply_K(deltapsi_qnG, qvector)
                residual_qnG = {q : new_deltapsi_qnG[q] - eigval*deltapsi_nG for q, deltapsi_nG in deltapsi_qnG.items()}



                error = self.calculate_error(residual_qnG)
                print(f"Index#-#Error = {k}#-#{error}")
                if error < error_threshold:
                    return eigval, deltapsi_qnG




            
                #Precondition and step
                for q, deltapsi_nG in deltapsi_qnG.items():
                    deltapsi_qnG[q] = deltapsi_nG + stepsize*Precondition(residual_qnG[q])
                    #deltapsi_qnG = {q : deltapsi_nG + stepsize*Precondition(residual_qnG[q]) for q, deltapsi_nG in deltapsi_qnG.items()}
            
            
                    #Orthonormalize ##Replace this with Gram-Schmidt if solving for more eigenvectors
            #for deltapsi_qnG in deltapsis[1:]:
            # deltapsis[1] = {q: deltapsi_nG - deltapsis[0][q]*self.inner_product(deltapsis[0], deltapsis[1]) for q, deltapsi_nG in deltapsis[1].items()}
            # norm = np.sqrt(self.inner_product(deltapsis[0], deltapsis[0]))
            # deltapsis[0] = {q : deltapsi_nG/norm for q, deltapsi_nG in deltapsis[0].items()}
            # norm = np.sqrt(self.inner_product(deltapsis[1], deltapsis[1]))
            # deltapsis[1] = {q : deltapsi_nG/norm for q, deltapsi_nG in deltapsis[1].items()}

            # matrix = np.zeros((2,2), dtype=np.complex128)
            # for i in range(2):
            #     for j in range(2):
            #         matrix[i,j] = self.inner_product(deltapsis[i], self.apply_K(deltapsis[j], qvector))

            # eigvals, eigvecs = np.linalg.eig(matrix)
            # #{q : val*multiplier for q, val in dic.items()}
            # n = {q: delta*eigvecs[0,0] + deltapsis[1][q]*eigvecs[1,0] for q, delta in deltapsis[0].items()}
            # #deltapsis[0]*eigvecs[0][0] + deltapsis[1]*eigvecs[0][1]
            # deltapsis[1] = {q: delta*eigvecs[0, 1] + deltapsis[1][q]*eigvecs[1,1] for q, delta in deltapsis[0].items()}
            # #deltapsis[1] = deltapsis[0]*eigvecs[1][0] + deltapsis[1]*eigvecs[1][1]
            # deltapsis[0] = n

            norm = np.sqrt(self.inner_product(deltapsi_qnG, deltapsi_qnG))
            deltapsi_qnG = {q : deltapsi_nG/norm for q, deltapsi_nG in deltapsi_qnG.items()}
            deltapsis[0] = deltapsi_qnG

            ##Replace this with subspace diagonalization if solving for more eigenvectors concurrently
            eigval = self.inner_product(deltapsi_qnG, self.apply_K(deltapsi_qnG, qvector))
            eigvals[0] = eigval

        return
        #raise ValueError("Calculation did not converge")


    def initial_guess(self):
        deltapsi_qnG = {}
        
        for index, kpt in enumerate(self.wfs.mykpts):
            ##TODO Replace this with better start guess probably
            
            deltapsi_qnG[index] = self.wfs.pd.zeros(self.wfs.setups.nvalence//2, q=index) + 1# + index*np.random.rand() #np.ones(kpt.psit.array.shape, dtype=np.complex128)
            numgs = deltapsi_qnG[index][0].shape[0]
            deltapsi_qnG[index][:, np.random.randint(numgs)] = 2
            # numns = deltapsi_qnG[index].shape[0]
            # deltapsi_qnG[index][np.random.randint(numns), np.random.randint(numgs)] = 1
            # deltapsi_qnG[index][np.random.randint(numns), np.random.randint(numgs)] = 1
            # deltapsi_qnG[index][np.random.randint(numns), np.random.randint(numgs)] = 1


            
        ##Replace this with guess dependent on other guess/found values to speed up convergence
        eigval = 2

        return deltapsi_qnG, eigval

    def calculate_error(self, residual_qnG):
        error = 0

        for q, residual_nG in residual_qnG.items():
            for residual_G in residual_nG:
                error += np.linalg.norm(residual_G)/len(residual_G)
            error *= 1/len(residual_nG)
        return error
        
        


    def inner_product(self, deltapsi1_qnG, deltapsi2_qnG):
        fine_delta_tilde_n1_G = self.get_fine_delta_tilde_n(deltapsi1_qnG)
        fine_delta_tilde_n2_G = self.get_fine_delta_tilde_n(deltapsi2_qnG)

        DeltaD1_aii = self.get_density_matrix(deltapsi1_qnG)
        DeltaD2_aii = self.get_density_matrix(deltapsi2_qnG)


        comp_charge1_G = self.get_compensation_charges(deltapsi1_qnG, DeltaD1_aii)
        comp_charge2_G = self.get_compensation_charges(deltapsi2_qnG, DeltaD2_aii)

        poisson_G = self.solve_poisson(fine_delta_tilde_n1_G + comp_charge1_G)
        poisson_R = self.calc.density.pd3.ifft(poisson_G.T)
        charge2_R = self.calc.density.pd3.ifft(fine_delta_tilde_n2_G.T + comp_charge2_G.T)

        P12 = np.einsum("ijk, ijk", charge2_R, poisson_R) #(fine_delta_tilde_n2_G + comp_charge2_G).dot(poisson_G)
        

        #C_app = {a : 2*setup.M_pp for a, setup in enumerate(self.wfs.setups)}

        Pd = 0
        for a, DeltaD1_ii in DeltaD1_aii.items():
            DeltaD1_p = pack(DeltaD1_ii)
            DeltaD2_p = pack(DeltaD2_aii[a])
            Pd = DeltaD1_p.dot((2*self.wfs.setups[a].M_pp).dot(DeltaD2_p))

        return P12 + Pd


    def get_density_matrix(self, deltapsi_qnG):
        pt = self.wfs.pt
        D_aii = {}
        nvalence = self.wfs.setups.nvalence//2
        for q_index, deltapsi_nG in deltapsi_qnG.items():
            c_ani = pt.integrate(deltapsi_nG, q=q_index)


            kpt = self.wfs.mykpts[q_index]
            for a, c_ni in c_ani.items():
                P_ni = kpt.projections[a][:nvalence]
                U_ij = np.dot(P_ni.T.conj()*kpt.f_n[:nvalence], c_ni)
                if a in D_aii:
                    D_aii[a] += U_ij + U_ij.conj().T
                else:
                    D_aii[a] = U_ij + U_ij.conj().T
        return D_aii
        


    def apply_K(self, deltapsi_qnG, qvector):
        potential_function = self.epsilon_potential(deltapsi_qnG)

        new_deltapsi_qnG = self.solve_sternheimer(potential_function, qvector)

        return {q: -val for q, val in new_deltapsi_qnG.items()}

            

    def epsilon_potential(self, deltapsi_qnG):
        #Should add qvector to args to construct potential suitable for individual kpt?


        #delta_tilde_n = self.get_delta_tilde_n(deltawfs_wfs)
        fine_delta_tilde_n = self.get_fine_delta_tilde_n(deltapsi_qnG)

       
        #Comp charges is dict: atom -> charge
        DeltaD_aii = self.get_density_matrix(deltapsi_qnG)
        total_delta_comp_charge = self.get_compensation_charges(deltapsi_qnG, DeltaD_aii)


        poisson_term = self.solve_poisson(fine_delta_tilde_n + total_delta_comp_charge)
        soft_term_G = self.transform_to_coarse_grid_operator(poisson_term)
        soft_term_R = self.calc.density.pd2.ifft(soft_term_G)


        W_ap = self.get_charge_derivative_terms(poisson_term, DeltaD_aii)


        def apply_potential2(wvf_nG, q_index):
            #wvf_nR = self.wfs.pd.zeros(x = len(wvf_nG), q=q_index)
            wvf_nR = np.array([self.wfs.pd.ifft(wvf_G, q=q_index) for wvf_G in wvf_nG])
            #for n, wvf_G in enumerate(wvf_nG):
            #    wvf_nR[n] = self.wfs.pd.ifft(wvf_G, q=q_index)
            #wvf_nR = self.wfs.pd.ifft(wvf_nG, q=q_index)

            pt = self.wfs.pt

            if q_index in self.c_qani:
                c_ani = self.c_qani[q_index]
                V1_nG = np.zeros_like(wvf_nG)
                for n, wvf_R in enumerate(wvf_nR):
                    V1_nG[n] = self.wfs.pd.fft(wvf_R*soft_term_R, q=q_index)
                #V1_nG = self.wfs.pd.fft(wvf_nR*soft_term_R, q=q_index)
                pt.add(V1_nG, c_ani, q=q_index)

            else:
                c_ani = pt.integrate(wvf_nG, q=q_index)
                self.c_qani[q_index] = c_ani
                V1_nG = np.zeros_like(wvf_nG)
                for n, wvf_R in enumerate(wvf_nR):
                    V1_nG[n] = self.wfs.pd.fft(wvf_R*soft_term_R, q=q_index)
                #V1_nG = self.wfs.pd.fft(wvf_nR*soft_term_R, q=q_index)
                pt.add(V1_nG, c_ani, q=q_index)
            return V1_nG



        def apply_potential(wvf_G, level_index, q_index):
            wvf_R = self.wfs.pd.ifft(wvf_G, q_index)
            pt = self.wfs.pt
            if (q_index, level_index) in self.c_qnai:
                c_ai = self.c_qnai[(q_index, level_index)]
            else:
                c_ai = pt.integrate(wvf_G, q=q_index)
                self.c_qani[(q_index, level_index)] = c_ai
            
            for a, c_i in c_ai.items():
                W_ii = unpack(W_ap[a])
                c_i[:] = c_i.dot(W_ii)

            V1 = self.wfs.pd.fft(soft_term_R*wvf_R, q=q_index)
            pt.add(V1, c_ai, q=q_index)
            return V1   


        return apply_potential




    def get_compensation_charges(self, deltapsi_qnG, DeltaD_aii):                        

        setups = self.wfs.setups
        Q_aL = {}
        for a in DeltaD_aii:
            setup = setups[a]
            DeltaD_ii = DeltaD_aii[a]
            Delta_iiL = setup.Delta_iiL
            Q_L = np.einsum("ij, ij...", DeltaD_ii, Delta_iiL)
            Q_aL[a] = np.real(Q_L)


                
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
        for q_index, deltapsi_nG in deltapsi_qnG.items():
            for state_index, deltapsi_G in enumerate(deltapsi_nG):
                deltapsi_R = pd.ifft(deltapsi_G, q=q_index)
                wf_R = pd.ifft(self.wfs.mykpts[q_index].psit.array[state_index], q=q_index)
                delta_n_R += 2*np.real(deltapsi_R*wf_R.conj())

        fine_delta_n_R, fine_delta_n_G = pd2.interpolate(delta_n_R, pd3)

        return pd3.fft(fine_delta_n_R)

    def solve_poisson(self, charge_G):
        #charge(q)
        pd3 = self.calc.density.pd3
        G2 = pd3.G2_qG[0].copy()
        G2[0] = 1.0
        
        return charge_G * 4 * np.pi/G2
        
    def transform_to_coarse_grid_operator(self, fine_grid_operator):        
        dens = self.calc.density
        coarse_potential = dens.pd2.zeros()

        dens.map23.add_to1(coarse_potential, fine_grid_operator)

        return coarse_potential
        





    def get_charge_derivative_terms(self, poisson_term, DeltaD_aii):
        setups = self.wfs.setups

        coulomb_factors_aL = self.calc.density.ghat.integrate(poisson_term)
        W_ap = {}
        
        for a, glm_int_L in coulomb_factors_aL.items():
            W_ap[a] = setups[a].Delta_pL.dot(glm_int_L.T)#.astype(np.complex128) 

                

        for a, setup in enumerate(self.wfs.setups):
            D_p = pack(DeltaD_aii[a])
            val = ((2*setup.M_pp).dot(D_p))
            W_ap[a] = W_ap[a] + val

        return W_ap
                



    def solve_sternheimer(self, apply_potential, qvector):
        deltapsi_qnG = {}
        nvalence = self.wfs.setups.nvalence//2
        for index, kpt in enumerate(self.wfs.kd.bzk_kc):
            kpt = self.wfs.kd.what_is(kpt)
            number_of_valence_states = self.wfs.setups.nvalence//2  #kpt.psit.array.shape[0]
            kpt.psit.read_from_file()


            #Get kpt object for k+q
            k_plus_q_index = self.wfs.kd.find_k_plus_q(qvector, [index])
            assert len(k_plus_q_index) == 1
            k_plus_q_index = k_plus_q_index[0]
            if k_plus_q_index >= len(self.wfs.mykpts):
                print(f"k_plus_q_index: {k_plus_q_index}")
                print(f"Len of mykpts: {len(self.wfs.mykpts)}. Len of bzk_kc: {len(self.wfs.kd.bzk_kc)}.")
                raise ValueError("index too large")
            k_plus_q_object = self.wfs.mykpts[k_plus_q_index]
            deltapsi_qnG[k_plus_q_index] = self.wfs.pd.zeros(x=(number_of_valence_states,), q=k_plus_q_index)

            #print(f"array shape: {kpt.psit.array.shape}")
            #print(f"object shape: {kpt.psit.array.shape}")
            #print(f"deltapsi shape: {deltapsi_qnG[index].shape}")
            eps_n = kpt.eps_n[:nvalence]
            psi_nG = kpt.psit.array[:nvalence]


            # linop = self._get_LHS_linear_operator2(k_plus_q_object, eps_n, k_plus_q_index)

            # RHS_nG = self._get_RHS2(k_plus_q_object, psi_nG, apply_potential, k_plus_q_index)
            # deltapsi_nG, info = bicgstab(linop, RHS_nG.reshape(-1), maxiter=1000)
            # if info != 0:
            #     raise ValueError("bicgstab did not converge. Info: {}".format(info))

            # deltapsi_qnG[index] = deltapsi_nG.reshape(RHS_nG.shape)
            

            for state_index, (energy, psi) in enumerate(zip(kpt.eps_n, kpt.psit.array)):
                if state_index >= nvalence:
                    break
                linop = self._get_LHS_linear_operator(k_plus_q_object, energy, k_plus_q_index)
                #print(f"state index: {state_index}, index: {index}")
                #print(f"psi shape: {psi.shape}")
                RHS = self._get_RHS(k_plus_q_object, psi, apply_potential, state_index, k_plus_q_index, kpt, index)
                assert RHS.shape[0] == linop.shape[1]
                #print("got RHS")

                deltapsi_G, info = bicgstab(linop, RHS, maxiter=1000)
                
                if info != 0:
                    print(f"Final error: {np.linalg.norm(linop.matvec(deltapsi_G) - RHS)}")
                    raise ValueError("bicgstab did not converge. Info: {}".format(info))
                    
                    

                # print(f"Shape of deltapsi_nG: {deltapsi_qnG[k_plus_q_index].shape}. Shape of bicgstab result: {deltapsi_G.shape}")
                # print(f"kpt index: {index}, kplusqindex: {k_plus_q_index}, state_index: {state_index}")
                # print(f"Shape of kpt.psit.array: {kpt.psit.array.shape}")
                # print(f"Shape of kplusq.array: {k_plus_q_object.psit.array.shape}")

                deltapsi_qnG[k_plus_q_index][state_index] = deltapsi_G

        return deltapsi_qnG


    def _get_LHS_linear_operator(self, kpt, energy, k_index):
        def mv(v):
            return self._apply_LHS_Sternheimer(v, kpt, energy, k_index)


        shape_tuple = (kpt.psit.array[0].shape[0], kpt.psit.array[0].shape[0])
        linop = LinearOperator(dtype=np.complex128, shape=shape_tuple, matvec=mv)

        return linop

    def _get_LHS_linear_operator2(self, kpt, eps_n, k_index):
        nvalence = self.wfs.setups.nvalence//2
        numGs = len(kpt.psit.array[0])

        def mv(v):
            #result = np.zeros(nvalence*numGs, dtype=np.complex128)
            #for n in range(nvalence):
            #    result[n*numGs:(n+1)*numGs] = self._apply_LHS_Sternheimer(v[n*numGs:(n+1)*numGs], kpt, eps_n[n], k_index)
            #return result
            return self._apply_LHS_Sternheimer2(v, kpt, eps_n, k_index)
        
        shape_tuple = (nvalence*numGs, nvalence*numGs)

        linop = LinearOperator(dtype=np.complex128, shape=shape_tuple, matvec=mv)
        return linop


    def _apply_LHS_Sternheimer(self, deltapsi_G, kpt, energy, k_index):
        
        alpha = 1
      
        result_G = alpha*self._apply_valence_projector(self._apply_overlap(deltapsi_G, k_index), kpt, k_index)
        result_G += self._apply_hamiltonian(deltapsi_G, kpt, k_index)
        result_G -= energy*self._apply_overlap(deltapsi_G, k_index)
        return result_G

    def _apply_LHS_Sternheimer2(self, deltapsi_nG, kpt, eps_n, k_index):
        
        alpha = 1
        
        result_nG = alpha*self._apply_LHS_valence_projector2(self._apply_LHS_overlap2(deltapsi_nG, k_index), kpt, k_index)
        result_nG += self._apply_hamiltonian2(deltapsi_nG, kpt, k_index)
        num_occ = self.wfs.setups.nvalence//2
        num_gs = len(deltapsi_nG)//num_occ
        result_nG -= np.kron(np.diag(eps_n), np.eye(num_gs)).dot(self._apply_LHS_overlap2(deltapsi_nG, k_index))
        return result_nG



    def _apply_valence_projector(self, deltapsi_G, kpt, k_index):
        result_G = np.zeros_like(deltapsi_G)
        num_occ = self.wfs.setups.nvalence//2
        pd = self.wfs.pd
        #print(f"K index: {k_index}")
        #print(f"array shape: {kpt.psit.array.shape}")
        for index, psi_G in enumerate(kpt.psit.array):
            if index >= num_occ: #We assume here that the array is ordered according to energy
                break
            #print(f"index: {index}")
            #print(f"shape psi: {psi_G.shape}, shape delta: {deltapsi_G.shape}")
            result_G += psi_G*(pd.integrate(psi_G, deltapsi_G)) 

        return result_G

    def _apply_LHS_valence_projector2(self, deltapsi_nG, kpt, k_index):
        result_nG = np.zeros_like(deltapsi_nG)
        num_occ = self.wfs.setups.nvalence//2

        pd = self.wfs.pd
        kpt.psit.read_from_file()
        psi_nG = kpt.psit.array[:num_occ]


        intt = np.diag(pd.integrate(psi_nG, deltapsi_nG.reshape(num_occ, -1)))

        result_nG = ((psi_nG.T*(intt)).T).reshape(result_nG.shape)

        return result_nG


    def _apply_valence_projector2(self, deltapsi_nG, kpt, k_index):
        result_nG = np.zeros_like(deltapsi_nG)
        num_occ = len(deltapsi_nG)

        pd = self.wfs.pd
        kpt.psit.read_from_file()
        psi_nG = kpt.psit.array[:num_occ]


        intt = np.diag(pd.integrate(psi_nG, deltapsi_nG))

        result_nG = (psi_nG.T*(intt)).T

        return result_nG

                           

    def _apply_overlap(self, wvf_G, q_index):
        pt = self.wfs.pt
        c_ai = pt.integrate(wvf_G, q=q_index)

        for a, c_i in c_ai.items():
            dO_ii = self.wfs.setups[a].dO_ii
            c_i[:] = c_i.dot(dO_ii)

        result_G = wvf_G.copy()
        pt.add(result_G, c_ai, q=q_index)
        return result_G

        
    def _apply_overlap2(self, wvf_nG, q_index):
        pt = self.wfs.pt
        c_ani = pt.integrate(wvf_nG, q=q_index)

        for a, c_ni in c_ani.items():
            dO_ii = self.wfs.setups[a].dO_ii
            c_ni[:, :] = c_ni.dot(dO_ii)

        result_nG = wvf_nG.copy()
        pt.add(result_nG, c_ani, q=q_index)
        return result_nG

    def _apply_LHS_overlap2(self, wvf_nG, q_index):
        num_occ = self.wfs.setups.nvalence//2
        pt = self.wfs.pt
        c_ani = pt.integrate(wvf_nG.reshape(num_occ, -1), q=q_index)

        for a, c_ni in c_ani.items():
            dO_ii = self.wfs.setups[a].dO_ii
            c_ni[:, :] = c_ni.dot(dO_ii)

        result_nG = wvf_nG.copy().reshape(num_occ,-1)
        pt.add(result_nG, c_ani, q=q_index)
        return result_nG.reshape(-1)
        

    def _apply_hamiltonian(self, deltapsi_G, kpt, k_index):
        kpt.psit.read_from_file() #This is to ensure everything is initialized

        result_G = deltapsi_G.copy()
        self.wfs.apply_pseudo_hamiltonian(kpt, self.calc.hamiltonian, deltapsi_G[None], result_G[np.newaxis])

        pt = self.wfs.pt
        c_ai = pt.integrate(deltapsi_G, q=k_index)

        for a, c_i in c_ai.items():
            dH_ii = unpack(self.calc.hamiltonian.dH_asp[a][0])
            c_i[:] = c_i.dot(dH_ii)

        pt.add(result_G, c_ai, q=k_index)
        return result_G

    def _apply_hamiltonian2(self, deltapsi_nG, kpt, k_index):
        kpt.psit.read_from_file()
        num_occ = self.wfs.setups.nvalence//2
        
        result_nG = deltapsi_nG.copy().reshape(num_occ, -1)
        dpsi_nG = deltapsi_nG.reshape(num_occ, -1)
        self.wfs.apply_pseudo_hamiltonian(kpt, self.calc.hamiltonian, dpsi_nG, result_nG)

        pt = self.wfs.pt
        c_ani = pt.integrate(dpsi_nG, q=k_index)
        
        for a, c_ni in c_ani.items():
            dH_ii = unpack(self.calc.hamiltonian.dH_asp[a][0])
            c_ni[:] = c_ni.dot(dH_ii)
        pt.add(result_nG, c_ani, q=k_index)
        return result_nG.reshape(-1)


    def _get_RHS(self, k_plus_q_pt, psi_G, apply_potential, level_index, k_plus_q_index, kpt, k_index): 
        v_psi_G = self.wfs.pd.zeros(q=k_plus_q_index)
        vpsi = apply_potential(psi_G, level_index, k_index)


        if len(v_psi_G) > len(vpsi):
            v_psi_G[:len(vpsi)] = vpsi
        else:
            v_psi_G = vpsi[:len(v_psi_G)]

        RHS_G = -(v_psi_G - self._apply_overlap(self._apply_valence_projector(v_psi_G, k_plus_q_pt, k_plus_q_index), k_plus_q_index))


        return RHS_G
        

    def _get_RHS2(self, kpt, psi_nG, apply_potential, k_index):

        v_psi_nG = apply_potential(psi_nG, k_index)

        RHS_nG = -(v_psi_nG - self._apply_overlap2(self._apply_valence_projector2(v_psi_nG, kpt, k_index), k_index))
        return RHS_nG



if __name__=="__main__":
    filen = "test.gpw"
    import os
    if False and os.path.isfile(filen):
        print("Reading file")
        respObj = SternheimerResponse(filen)
    else:
        from time import time
        t1 = time()
        ro = SternheimerResponse(filen)
        t2 = time()
        print(f"Calculation took {t2 - t1} seconds")
        exit()
        print("Generating new test file")
        from ase.build import bulk
        from gpaw import PW, FermiDirac
        a = 10
        c = 5
        d = 0.74
        #atoms = Atoms("H2", positions=([c-d/2, c, c], [c+d/2, c,c]),
        #       cell = (a,a,a))
        atoms = bulk("Si", "diamond", 5.43)
        calc = GPAW(mode=PW(150, force_complex_dtype=True),
                xc ="PBE",                
                kpts=(4,4,4),
                random=True,
                #symmetry=False,
                occupations=FermiDirac(0.01),
                txt = "test.out")
        
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        calc.write(filen, mode = "all")
        #exit()
        respObj = SternheimerResponse(filen)
        

