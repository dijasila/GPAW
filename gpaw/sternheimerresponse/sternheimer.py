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
        self.calc.initialize_positions()
        qvector = np.array([0,0,0])
        self.calculate([qvector], 1)
        

    def calculate(self, qvectors, num_eigs):
        qvector = qvectors[0]
        
        deltapsi_qnG, eigval = self.initial_guess()

        

        error_threshold = 1e-9
        error = 100
        num_iter = 200
        def Precondition(deltapsi_qnG):
            return deltapsi_qnG
        stepsize = 0.1

        for niter in range(num_iter):
            print(f"Iteration number: {niter}")
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
            deltapsi_qnG[index] = np.ones(kpt.psit.array.shape, dtype=np.complex128)

            
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
        pt = self.wfs.pt
        D_aii = {}
        for q_index, deltapsi_nG in deltapsi_qnG.items():
            c_axi = pt.integrate(deltapsi_nG, q=q_index)


            kpt = self.wfs.mykpts[q_index]
            for a, c_xi in c_axi.items():
                P_ni = kpt.projections[a]
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
        soft_term_R = self.calc.density.pd2.ifft(soft_term_G)


        W_ap = self.get_charge_derivative_terms(poisson_term, DeltaD_aii)



        def apply_potential(wvf_G, q_index):
            wvf_R = self.wfs.pd.ifft(wvf_G)
            pt = self.wfs.pt
            c_ai = pt.integrate(wvf_G, q=q_index)
            
            for a, c_i in c_ai.items():
                W_ii = unpack(W_ap[a])
                c_i[:] = W_ii.dot(c_i.T).T ##TODO ask JJ about these transposes

            V1 = self.wfs.pd.fft(soft_term_R*wvf_R)
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
            Q_aL[a] = np.real(Q_L) ##TODO ask JJ about this real part


                
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
                deltapsi_R = pd.ifft(deltapsi_G)
                wf_R = pd.ifft(self.wfs.mykpts[q_index].psit.array[state_index])
                delta_n_R += 2*np.real(deltapsi_R*wf_R)

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
            W_ap[a] = setups[a].Delta_pL.dot(glm_int_L.T).astype(np.complex128) ##TODO ask JJ about transpose

                

        for a, setup in enumerate(self.wfs.setups):
            D_p = pack(DeltaD_aii[a])
            val = ((2*setup.M_pp).dot(D_p))
            W_ap[a] = W_ap[a] + val

        return W_ap
                



    def solve_sternheimer(self, apply_potential, qvector):
        deltapsi_qnG = {}

        for index, kpt in enumerate(self.wfs.mykpts):
            number_of_valence_states = kpt.psit.array.shape[0]
            deltapsi_qnG[index] = self.wfs.pd.zeros(x=(number_of_valence_states,), q=index)


            #Get kpt object for k+q
            k_plus_q_index = self.wfs.kd.find_k_plus_q(qvector, [index])
            assert len(k_plus_q_index) == 1
            k_plus_q_index = k_plus_q_index[0]
            k_plus_q_object = self.wfs.mykpts[k_plus_q_index]



            for state_index, (energy, psi) in enumerate(zip(kpt.eps_n, kpt.psit.array)):
                linop = self._get_LHS_linear_operator(k_plus_q_object, energy, k_plus_q_index)
                RHS = self._get_RHS(k_plus_q_object, psi, apply_potential, k_plus_q_index)
                
                deltapsi_G, info = bicgstab(linop, RHS, maxiter=1000)

                if info != 0:
                    print(f"Final error: {np.linalg.norm(linop.matvec(deltapsi_G) - RHS)}")
                    raise ValueError("bicgstab did not converge. Info: {}".format(info))
                    


                deltapsi_qnG[index][state_index] = deltapsi_G

        return deltapsi_qnG


    def _get_LHS_linear_operator(self, kpt, energy, k_index):
        def mv(v):
            return self._apply_LHS_Sternheimer(v, kpt, energy, k_index)


        shape_tuple = (kpt.psit.array[0].shape[0], kpt.psit.array[0].shape[0])
        linop = LinearOperator(dtype=np.complex128, shape=shape_tuple, matvec=mv)

        return linop


    def _apply_LHS_Sternheimer(self, deltapsi_G, kpt, energy, k_index):
        result_G = np.zeros_like(deltapsi_G)
        
        alpha = 1
        
        result_G = alpha*self._apply_valence_projector(self._apply_overlap(deltapsi_G, k_index), kpt, k_index)
        
        result_G = result_G + self._apply_hamiltonian(deltapsi_G, kpt)
        
        result_G = result_G - energy*self._apply_overlap(deltapsi_G, k_index)

        return result_G

    def _apply_valence_projector(self, deltapsi_G, kpt, k_index):
        result_G = np.zeros_like(deltapsi_G)
        f_n = kpt.f_n/kpt.f_n.max() #TODO ask JJ
        num_occ = f_n.sum()

        for index, psi_G in enumerate(kpt.psit.array):
            if index >= num_occ: #We assume here that the array is ordered according to energy
                break
            result_G = result_G + psi_G*(psi_G.conj().dot(deltapsi_G)*f_n[index]) 


        return result_G

    def _apply_overlap(self, wvf_G, q_index):
        pt = self.wfs.pt
        c_ai = pt.integrate(wvf_G, q=q_index)
        result_G = self.wfs.pd.zeros()
        pt.add(result_G, c_ai, q=q_index)
        return result_G


    def _apply_hamiltonian(self, deltapsi_G, kpt):
        #TODO ask JJ about this function
        #This function mimics this approach in eigensolver.subspace_diagonalize
        kpt.psit.read_from_file() #This is to ensure everything is initialized

        H_partial = partial(self.wfs.apply_pseudo_hamiltonian, kpt, self.calc.hamiltonian)
        psit = kpt.psit

        tmp = psit.new(buf=self.wfs.work_array)
        H = self.wfs.work_matrix_nn
        P2 = kpt.projections.new()

        psit.matrix_elements(operator=H_partial, result=tmp, out=H,
                     symmetric=True, cc=True) ##TODO ask JJ about whether cc=True is correct here

        #Add corrections (? Ask JJ)
        self.calc.hamiltonian.dH(kpt.projections, out = P2)
        
        mmm(1.0, kpt.projections, "N", P2, "C", 1.0, H,  symmetric=True) #H is out-variable

        self.calc.hamiltonian.xc.correct_hamiltonian_matrix(kpt, H.array)
        
        result_G = np.zeros_like(deltapsi_G)

        for index1, wvf1_G in enumerate(psit.array):
            for index2, wvf2_G, in enumerate(psit.array):
                inner_prod = wvf2_G.conj().dot(deltapsi_G)
                matrix_element = H.array[index1, index2]
                result_G = result_G + wvf1_G*matrix_element*inner_prod

        return result_G
        
    def _get_RHS(self, kpt, psi_G, apply_potential, k_index):
        RHS_G = np.zeros_like(kpt.psit.array[0])
        v_psi_G = apply_potential(psi_G, k_index)
        RHS_G = -(v_psi_G - self._apply_overlap(self._apply_valence_projector(v_psi_G, kpt, k_index), k_index))

        return RHS_G
        





if __name__=="__main__":
    filen = "test.gpw"
    import os
    if False and os.path.isfile(filen):
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
        calc = GPAW(mode=PW(50, force_complex_dtype=True),
                xc ="PBE",                
                kpts=(1,1,1),
                random=True,
                #symmetry=False,
                occupations=FermiDirac(0.01),
                txt = "test.out")
        
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        calc.write(filen, mode = "all")
        exit()
        respObj = SternheimerResponse(filen)
        

