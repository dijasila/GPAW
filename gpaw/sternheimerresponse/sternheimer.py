from gpaw import GPAW
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
from gpaw.utilities import pack, unpack, unpack2
from time import time


class CustomLinearOperator:
    def __init__(self, matvec):
        self.matvec = matvec


def custombicgstab(A_nGG, b_nG, max_iter = 200, tol=1e-8):
    '''
    See wikipedia page on Unpreconditioned BiCGStab method for variable defs
    
    '''
    error_threshold = tol    
    x0_nG = np.zeros_like(b_nG)
    r0_nG = b_nG

    rhat0_nG = r0_nG.copy()

    rho0_n = np.ones(len(b_nG))
    alpha_n = np.ones(len(b_nG))
    w0_n = np.ones(len(b_nG))

    nu0_nG = np.zeros_like(b_nG)
    p0_nG = np.zeros_like(b_nG)
    for niter in range(max_iter):
        rhoi_n = np.einsum("ij, ij -> i", rhat0_nG, r0_nG)

        beta_n = (rhoi_n/rho0_n)*(alpha_n/w0_n)

        pi_nG = r0_nG + np.einsum("i, ij->ij", beta_n, p0_nG) - np.einsum("i, i, ij -> ij", beta_n, w0_n, nu0_nG)

        nui_nG = A_nGG.matvec(pi_nG)#np.einsum("ijk, ik -> ij", A_nGG, pi_nG)

        thing_n = np.einsum("ij, ij->i", rhat0_nG, nui_nG)
        alpha_n = rhoi_n/thing_n


        h_nG = x0_nG + np.einsum("i, ij->ij", alpha_n, pi_nG)
        dotprod_nG = A_nGG.matvec(pi_nG)#np.einsum("ijk, ik->ij", A_nGG, h_nG)
        if np.allclose(dotprod_nG, b_nG, atol=error_threshold):
            return h_nG, niter

        s_nG = r0_nG - np.einsum("i, ij->ij", alpha_n, nui_nG)
    
        t_nG = A_nGG.matvec(s_nG)#np.einsum("ijk, ik->ij", A_nGG, s_nG)        
        norm_n = np.einsum("ij, ij->i", t_nG, t_nG)
        assert norm_n.all()
        wi_n = np.einsum("ij, ij->i", t_nG, s_nG)/norm_n


        xi_nG = h_nG + np.einsum("i, ij->ij", wi_n, s_nG)
        dotprod_nG = A_nGG.matvec(xi_nG)#np.einsum("ijk, ik->ij", A_nGG, xi_nG)
        if np.allclose(dotprod_nG, b_nG, atol=error_threshold):
            return xi_nG, niter
        ri_nG = s_nG - np.einsum("i, ij->ij", wi_n, t_nG)


        rho0_n = rhoi_n.copy()
        nu0_nG = nui_nG.copy()
        w0_n = wi_n.copy()
        x0_nG = xi_nG.copy()
        r0_nG = ri_nG.copy()
        p0_nG = pi_nG.copy()


    raise ValueError(f"BiCGStab did not converge. Iter: {niter}. Error: {np.max(np.abs(A_nGG.matvec(x0_nG) - b_nG))}")






class SternheimerResponse:
    def __init__(self, filename, runstuff=True):
        self.restart_filename = filename
        self.calc = GPAW(filename, txt=None)
        #TODO Check that all info is in file
        self.wfs = self.calc.wfs
        self.calc.initialize_positions()
        self.nbands = self.wfs.bd.nbands#//2
        print("nbands=", self.nbands)
        self.c_qani = {}
        self.c_qnai = {}
        #bz_to_ibz = self.calc.get_bz_to_ibz_map()
        #ntotal_kpts = len(bz_to_ibz)
        spin_factor = 1
        self.kpt_weight = spin_factor*self.wfs.kd.weight_k
        #np.array([(bz_to_ibz == k).sum() for k in list(set(bz_to_ibz))])*
        #self.kpt_weight = np.zeros_like(self.kpt_weight) + 1
        #ind1 = self.wfs.mykpts[1].q
        #ind0 = self.wfs.mykpts[0].q
        #print(f"My k vecs: {[self.wfs.kd.bzk_kc[ind.q] for ind in self.wfs.mykpts]}")
        #qvector = self.wfs.kd.ibzk_kc[1] - self.wfs.kd.ibzk_kc[0]
        self.t1s = []
        self.t2s = []
        qvector = np.array([0,0,0])
        #self.calculate([qvector], 1)
        self.deltapsi_qnG = None
        self.print = False
        t1 = time()
        #self.deflatedarnoldicalculate([qvector], 1)
        if runstuff:
            self.eigval, self.fdnt_R = self.powercalculate([qvector], 1)
            #self.krylovcalculate([qvector], 1)
        t2 = time()
        # print(f"Calculation took {t2 - t1} seconds.")
        # print(f"BiCGStab took {np.mean(self.t1s)} seconds on avg.")
        # print(f"Performed {len(self.t1s)} BiCGStab calls")
        # print(f"Get LHS+RHS took {np.mean(self.t2s)} seconds on avg.")
        #print(f"Performed {len(self.t2s)} LHS+RHS calls")




    def deflatedarnoldicalculate(self, qvectors, num_eigs):
        print("DEFLATEDARNOLDICALCULATE")
        qvector = qvectors[0]
        self.calculate_kplusq(qvector)

        deltapsi_qnG, eigval = self.initial_guess(qvector)
        if self.deltapsi_qnG is not None:
            print("Using previously calculated deltapsi")
            deltapsi_qnG = self.deltapsi_qnG
        norm = self.inner_product(deltapsi_qnG, deltapsi_qnG)
        deltapsi_qnG = {q: val/np.sqrt(norm) for q, val in deltapsi_qnG.items()}
        error_threshold = 1e-4
        error = 100

        max_iter = 100
        new_norm = 0
        old_norm = 0

        Kv_SqnG = [] #S is a step-number index
        v_SqnG = [deltapsi_qnG]
        init_ip = self.inner_product(v_SqnG[0], self.apply_K(v_SqnG[0], qvector))
        K_ij = np.array([[0]]) #Initial Krylov space is empty, so matrix is zero
        check = False

        

        krylov_size = 5
        max_iter = 10 
        num_eigenpairs = 5
        
        K_ij = np.zeros((krylov_size, krylov_size), dtype=np.complex128)

        k = 0

        c = 0
        while c <= max_iter:
            c = c + 1
            print(f"Iteration number: {c}")
            t1 = time()


            for j in range(k, krylov_size):
                print(f"Subloop index: {j}")
                vj_qnG = v_SqnG[j]
                
                w_qnG = self.apply_K(vj_qnG, qvector)
                
                for i in range(j+1):
                    K_ij[i,j] = self.inner_product(v_SqnG[i], w_qnG)
                    w_qnG = {q: val - K_ij[i,j]*v_SqnG[i][q] for q, val in w_qnG.items()}
                norm = np.sqrt(self.inner_product(w_qnG, w_qnG))
                w_qnG = {q: val/norm for q, val in w_qnG.items()}

                if j != krylov_size-1:
                    K_ij[j+1, j] = norm
                    if len(v_SqnG) < krylov_size:
                        v_SqnG.append(w_qnG)
                    else:
                        v_SqnG[j+1] = w_qnG
                # else:
                #     v_SqnG[0] = w_qnG
                
            heigs, hvecs = np.linalg.eig(K_ij)

            pairs = sorted(zip(np.real(heigs), hvecs.T), key=lambda t: t[0])
            
            eigval, vec = pairs[-1]

            eigen_qnG = {q: val*vec[0] for q, val in v_SqnG[0].items()}

            for S, v_qnG in enumerate(v_SqnG[1:]):
                eigen_qnG = {q: val + vec[S]*v_qnG[q] for q, val in eigen_qnG.items()}
                


            residual_qnG = self.apply_K(eigen_qnG, qvector)
            residual_qnG = {q: val - eigval*eigen_qnG[q] for q, val in residual_qnG.items()}
            
            residual_norm = np.sqrt(self.inner_product(residual_qnG, residual_qnG))
            print(f"Estimated eigval: {eigval}")
            print(f"Residual norm was: {residual_norm}")

            for S, v_qnG in enumerate(v_SqnG[:k]):
                ip = self.inner_product(eigen_qnG, v_qnG)
                eigen_qnG = {q: val - ip*v_qnG[q] for q, val in eigen_qnG.items()}
                
            eigen_norm = np.sqrt(self.inner_product(eigen_qnG, eigen_qnG))
            eigen_qnG = {q: val/eigen_norm for q, val in eigen_qnG.items()}
            
            v_SqnG = v_SqnG[:k+1]
            v_SqnG[k] = eigen_qnG
            

            if residual_norm < 1e+1:
                for i in range(k+1):
                    K_ij[i, k] = self.inner_product(v_SqnG[i], self.apply_K(eigen_qnG, qvector))

                for i in range(k+1, krylov_size):
                    K_ij[i,k] = 0

                k = k + 1
                if k >= num_eigenpairs:
                    break
                print("Incremented k")
                print(f"Used {c} iterations to converge vector")
                c = 0

                new_start, _ = self.initial_guess(qvector)
                new_norm = np.sqrt(self.inner_product(new_start, new_start))
                new_start = {q: val/new_norm for q, val in new_start.items()}
                v_SqnG.append(new_start)
                
                
        eigvals, eigvecs = np.linalg.eig(K_ij)
        print(f"Final eigs: {np.sort(np.real(eigvals))}")
        print("Converged")
        return
















    def residualkrylovcalculate(self, qvectors, num_eigs):
        print("RESIDUALKRYLOVCALCULATE")
        qvector = qvectors[0]
        self.calculate_kplusq(qvector)

        deltapsi_qnG, eigval = self.initial_guess(qvector)
        if self.deltapsi_qnG is not None:
            print("Using previously calculated deltapsi")
            deltapsi_qnG = self.deltapsi_qnG
        norm = self.inner_product(deltapsi_qnG, deltapsi_qnG)
        deltapsi_qnG = {q: val/np.sqrt(norm) for q, val in deltapsi_qnG.items()}
        error_threshold = 1e-4
        error = 100
        stepsize = 0.2

        max_iter = 100
        new_norm = 0
        old_norm = 0
        leading_eig = 14
        Kv_SqnG = [] #S is a step-number index
        v_SqnG = [deltapsi_qnG]
        new_deltapsi_qnG = self.apply_K(v_SqnG[0], qvector)
        init_ip = self.inner_product(v_SqnG[0], new_deltapsi_qnG)
        K_ij = np.array([[init_ip]]) #Initial Krylov space is empty, so matrix is zero
        check = False
        for niter in range(max_iter):
            print(f"Iteration number: {niter}")
            t1 = time()
            if check:
                for j, jvec in enumerate(v_SqnG):
                    for k, kvec in enumerate(v_SqnG):
                        if k == j:
                            b = np.allclose(self.inner_product(jvec, kvec), 1)
                            if not b:
                                print(f"Not-one IP for index: {j}")
                                print(f"Inner product: {self.inner_product(jvec, kvec)}")
                            assert b
                        else:
                            b = np.allclose(self.inner_product(jvec, kvec), 0)
                            if not b:
                                print(f"Non-zero IP for pair: {j}-{k}")
                                print(f"Inner product: {self.inner_product(jvec, kvec)}")
                            assert b


            ##new_deltapsi_qnG = self.apply_K(v_SqnG[niter], qvector)
            Kv_SqnG.append(new_deltapsi_qnG.copy())
            ##Step in direction of residual
            ##Could also be any operator
            new_deltapsi_qnG = {q: v_SqnG[niter][q] +stepsize*(val - leading_eig*v_SqnG[niter][q]) for q, val in new_deltapsi_qnG.items()}


            #Change matrix dims from m x m to (m+1) x (m+1)
            new_shape = tuple(np.array(K_ij.shape) + np.array((1,1)))
            new_K_ij = np.zeros(new_shape, dtype=np.complex128)
            new_K_ij[:-1, :-1] = K_ij
            
            #Gram-Schmidt 
            for Sindex, v_qnG in enumerate(v_SqnG):
                innerprod = self.inner_product(v_qnG, new_deltapsi_qnG)                
                new_deltapsi_qnG = {q : val - innerprod*v_qnG[q] for q, val in new_deltapsi_qnG.items()}

            norm = np.sqrt(self.inner_product(new_deltapsi_qnG, new_deltapsi_qnG))

            new_v_qnG = {q: val/norm for q, val in new_deltapsi_qnG.items()}

            new_deltapsi_qnG = self.apply_K(new_v_qnG, qvector)

            for Sindex, v_qnG in enumerate(v_SqnG):
                new_K_ij[Sindex, niter+1] = self.inner_product(v_qnG, new_deltapsi_qnG)
                new_K_ij[niter+1, Sindex] = self.inner_product(Kv_SqnG[Sindex], v_qnG)
            new_K_ij[niter+1, niter+1] = self.inner_product(new_v_qnG, new_deltapsi_qnG)

            ##Then append the orthonormalized vector v_SqnG
            v_SqnG.append(new_v_qnG.copy())

            # for i in range(new_shape[0]):
            #     new_K_ij[-1, i] = self.inner_product(v_SqnG[-1], Kv_SqnG[i])
            #     new_K_ij[i, -1] = self.inner_product(v_SqnG[i], Kv_SqnG[-1])

            K_ij = new_K_ij            

            eigvals, eigvecs = np.linalg.eig(K_ij)
            leading_eig = np.sort(np.real(eigvals))[-1]
            print(f"Eigvals: {np.sort(np.real(eigvals))}")
            t2 = time()
            print(f"Iteration took: {t2 - t1} seconds")

        self.deltapsi_qnG = deltapsi_qnG

            
        print("Not converged")
        return






    def krylovcalculate(self, qvectors, num_eigs):
        print("KRYLOVCALCULATE")
        qvector = qvectors[0]
        #self.calculate_kplusq(qvector)

        deltapsi_qnG, eigval = self.initial_guess(qvector)
        if self.deltapsi_qnG is not None:
            print("Using previously calculated deltapsi")
            deltapsi_qnG = self.deltapsi_qnG
        norm = self.inner_product(deltapsi_qnG, deltapsi_qnG)
        deltapsi_qnG = {q: val/np.sqrt(norm) for q, val in deltapsi_qnG.items()}
        error_threshold = 1e-4
        error = 100

        max_iter = 100
        new_norm = 0
        old_norm = 0

        Kv_SqnG = [] #S is a step-number index
        v_SqnG = [deltapsi_qnG]
        init_ip = self.inner_product(v_SqnG[0], self.apply_K(v_SqnG[0], qvector))
        K_ij = np.array([[0]]) #Initial Krylov space is empty, so matrix is zero
        check = True
        for niter in range(max_iter):
            print(f"Iteration number: {niter}")
            t1 = time()
            if check:
                for j, jvec in enumerate(v_SqnG):
                    for k, kvec in enumerate(v_SqnG):
                        if k == j:
                            continue
                        else:
                            b = np.allclose(self.inner_product(jvec, kvec), 0)
                            if not b:
                                print(f"Inner product: {self.inner_product(jvec, kvec)}")
                            assert b


            new_deltapsi_qnG = self.apply_K(v_SqnG[niter], qvector)




            Kv_SqnG.append(new_deltapsi_qnG.copy())
            #Some code to orthonormalize            
            #if (niter % 2) == 0: ##Does not work. Eigenvalues of subspace matrix is incorrect b/c basis vectors are not linearly independent
            new_shape = tuple(np.array(K_ij.shape) + np.array((1,1)))
            new_K_ij = np.zeros(new_shape, dtype=np.complex128)
            new_K_ij[:-1, :-1] = K_ij
            ##Update orthonormalization. Can apparently be done in less steps, see wikipedia
            for Sindex, v_qnG in enumerate(v_SqnG):
                innerprod = self.inner_product(v_qnG, new_deltapsi_qnG)
                new_K_ij[Sindex, niter] = innerprod
                new_deltapsi_qnG = {q : val - innerprod*v_qnG[q] for q, val in new_deltapsi_qnG.items()}

            norm = np.sqrt(self.inner_product(new_deltapsi_qnG, new_deltapsi_qnG))
            new_K_ij[niter+1, niter] = norm


            new_deltapsi_qnG = {q: val/norm for q, val in new_deltapsi_qnG.items()}
            ##Then append the orthonormalized vector v_SqnG
            v_SqnG.append(new_deltapsi_qnG.copy())

            #Update K matrix
            # for b, vec in enumerate(v_SqnG):
            #     for c, vecc in enumerate(v_SqnG):
            #         new_K_ij[b,c] = self.inner_product(vec, self.apply_K(vecc, qvector))


            # for i in range(new_shape[0]):
            #     new_K_ij[-1, i] = self.inner_product(v_SqnG[-1], Kv_SqnG[i])
            #     new_K_ij[i, -1] = self.inner_product(v_SqnG[i], Kv_SqnG[-1])

            K_ij = new_K_ij            

            eigvals, eigvecs = np.linalg.eig(K_ij)
            print(f"Eigvals: {np.sort(np.real(eigvals))}")
            t2 = time()
            print(f"Iteration took: {t2 - t1} seconds")

        self.deltapsi_qnG = deltapsi_qnG

            
        print("Not converged")
        return




    def powercalculate(self, qvectors, num_eigs):
        print("POWERCALCULATE")
        qvector = qvectors[0]
        #self.calculate_kplusq(qvector)

        deltapsi_qnG, eigval = self.initial_guess(qvector)
        #if self.deltapsi_qnG is not None:
            #print("Using previously calculated deltapsi")
            #deltapsi_qnG = self.deltapsi_qnG

        error_threshold = 1e-6
        error = 100
        error2 = 100

        max_iter = 100
        new_norm = 0
        old_norm = 0
        for niter in range(max_iter):
            print(f"Iteration number: {niter}")
            print(f"Eigenvalue: {eigval}. Error: {error}")
            print(f"Error2: {error2}")
            if niter == 3:
                self.print = True
            new_deltapsi_qnG = self.apply_K(deltapsi_qnG, qvector)

            #new_deltapsi_qnG = {q: np.random.rand(*deltapsi_qnG[q].shape).astype(np.complex128) for q in deltapsi_qnG}


            new_norm = self.inner_product(new_deltapsi_qnG, new_deltapsi_qnG)
            nn2 = new_norm
            eigval = np.sqrt(new_norm)
            #eigval = self.inner_product(deltapsi_qnG, self.apply_K(deltapsi_qnG, qvector))
            new_deltapsi_qnG = {q: 0*deltapsi_qnG[q] + 1*val/np.sqrt(new_norm) for q, val in new_deltapsi_qnG.items()}
            new_norm = self.inner_product(new_deltapsi_qnG, new_deltapsi_qnG)

            new_deltapsi_qnG = {q: val/np.sqrt(new_norm) for q, val in new_deltapsi_qnG.items()}
            error = np.abs(1 - self.inner_product(deltapsi_qnG, new_deltapsi_qnG))
            error2 = np.abs(1- self.inner_product(new_deltapsi_qnG, new_deltapsi_qnG))
            #error = 100
            if error < error_threshold:
                print(f"Converged. Value: {eigval}")
                self.deltapsi_qnG = deltapsi_qnG
                #return
                deltapsi_qnG = new_deltapsi_qnG
                break

            deltapsi_qnG = new_deltapsi_qnG

        self.deltapsi_qnG = deltapsi_qnG
        
        fdnt_R = self.get_fine_delta_tilde_n(deltapsi_qnG, return_real_space=True)
        # import matplotlib.pyplot as plt
        # plt.plot(fdnt_R[:,0,0], label="x")
        # plt.plot(fdnt_R[0,:,0], label="y")
        # plt.plot(fdnt_R[0,0,:], label="z")
        # plt.legend()
        # plt.show()
        
        #print("Not converged")
        return eigval, fdnt_R


            
            
        

    def calculate(self, qvectors, num_eigs):
        qvector = qvectors[0]
        self.calculate_kplusq(qvector)
        
        deltapsi_qnG1, eigval1 = self.initial_guess(qvector)
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



    def initial_guess(self, qvector):
        deltapsi_qnG = {}
        np.random.seed(123)
        for index, kpt in enumerate(self.wfs.mykpts):
            if kpt.s == 1:
                continue
            ##TODO Replace this with better start guess probably

            bzk_index = self.wfs.kd.ibz2bz_k[index]
            #Get kpt object for k+q
            k_plus_q_index = self.wfs.kd.find_k_plus_q(qvector, [bzk_index])
            assert len(k_plus_q_index) == 1
            k_plus_q_index = k_plus_q_index[0]
            k_plus_q_index = self.wfs.kd.bz2ibz_k[k_plus_q_index]
            
            

            deltapsi_qnG[k_plus_q_index] = self.wfs.pd.zeros(self.nbands, q=k_plus_q_index)# + index*np.random.rand() #np.ones(kpt.psit.array.shape, dtype=np.complex128)

            if np.allclose(self.wfs.kd.bzk_kc[bzk_index], [0, 0, 0]):
                deltapsi_qnG[k_plus_q_index][:, 1] += 1
            
            numgs = deltapsi_qnG[k_plus_q_index][0].shape[0]
            deltapsi_qnG[k_plus_q_index][:, np.random.randint(numgs)] = 5
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
        poisson_R = self.calc.density.pd3.ifft(poisson_G)
        charge2_R = self.calc.density.pd3.ifft(fine_delta_tilde_n2_G + comp_charge2_G)


        P12 = self.calc.hamiltonian.finegd.integrate(charge2_R, poisson_R)
        

        
        #fakeP12 = np.einsum("ijk, ijk", charge2_R, poisson_R) #(fine_delta_tilde_n2_G + comp_charge2_G).dot(poisson_G)

        

        #C_app = {a : 2*setup.M_pp for a, setup in enumerate(self.wfs.setups)}

        Pd = 0
        for a, DeltaD1_ii in DeltaD1_aii.items():
            DeltaD1_p = pack(DeltaD1_ii)
            DeltaD2_p = pack(DeltaD2_aii[a])
            Pd += DeltaD1_p.dot((2*self.wfs.setups[a].M_pp).dot(DeltaD2_p))

        return P12 + Pd


    def get_density_matrix(self, deltapsi_qnG):
        pr = True
        pt = self.wfs.pt
        D_aii = {}
        nbands = self.nbands
        for q_index, deltapsi_nG in deltapsi_qnG.items():

            c_ani = pt.integrate(deltapsi_nG, q=q_index)
            kpt = self.wfs.mykpts[q_index]

            for a, c_ni in c_ani.items():
                P_ni = kpt.projections[a][:nbands]

                U_ij = np.dot(P_ni.conj().T*kpt.f_n[:nbands], c_ni)

                if a in D_aii:
                    D_aii[a] +=  U_ij + U_ij.conj().T# + U_ij.conj().T
                else:
                    D_aii[a] = U_ij + U_ij.conj().T# + U_ij.conj().T
        #print(D_aii)
        return D_aii
        D_asp = {a : np.array([ pack(D_aii[a]).real ]) for a in D_aii} 

        a_sa = self.wfs.kd.symmetry.a_sa

        for s in range(self.wfs.nspins):
            D_aii = [unpack2(D_asp[a][s])
                     for a in range(len(D_asp))]
            for a, D_ii in enumerate(D_aii):
                setup = self.wfs.setups[a]
                D_asp[a][s] = pack(setup.symmetrize(a, D_aii, a_sa))


        # #print(f"Size of D_ii: {D_aii[0].shape}")
        D_aii = {a: unpack2(D_asp[a][0]) for a in D_asp}
        # print("Delta D:")
        # print(D_aii[0][:4, :4])
        # print("Delta D sum")
        # print(D_aii[0].sum().sum())
        # print("Old Delta D:")
        # print(oldD_aii[0][:4, :4])
        # print("Old Delta D sum")
        # print(oldD_aii[0].sum().sum())
        # exit()
        return D_aii
        


    def apply_K(self, deltapsi_qnG, qvector):
        # from gpaw.response.df import DielectricFunction
        # df = DielectricFunction(self.restart_filename, ecut=250, eta=5)
        # eps_GG = df.get_dielectric_matrix(symmetric=False)[0]
        
        # #self.wfs.pd.zeros(x=(number_of_valence_states,), q=k_plus_q_index)



        # test_vector_nG = self.wfs.pd.zeros(x = (4,), q=0)
        # test_vector_nG[0,3] = 1
        # potential = self.epsilon_potential({0: test_vector_nG})
        # sternheimer_result = self.solve_sternheimer(potential, 0)[0][0]
        # exact_result = np.dot(eps_GG, test_vector_nG[0])
        # ##Actually these two things should not be equal. One is epsV other is eps*wvf

        # if not np.allclose(sternheimer_result, exact_result):
        #     import matplotlib.pyplot as plt
        #     plt.plot(sternheimer_result, label="Sternheimer")
        #     plt.plot(exact_result, label="Exact")
        #     plt.legend()
        #     plt.show()
        #     raise ValueError()
        #exit()
        

        potential_function = self.epsilon_potential(deltapsi_qnG)

        new_deltapsi_qnG = self.solve_sternheimer(potential_function, qvector)

        return {q:  -val for q, val in new_deltapsi_qnG.items()}

            

    def epsilon_potential(self, deltapsi_qnG):
        # Should add qvector to args to construct potential suitable for individual kpt?

        
        # ## Compensation charge on fine grid
        # # delta_tilde_n = self.get_delta_tilde_n(deltawfs_wfs)
        # fine_delta_tilde_n_G = self.get_fine_delta_tilde_n(deltapsi_qnG)
       
        # # Comp charges is dict: atom -> charge
        # DeltaD_aii = self.get_density_matrix(deltapsi_qnG)
        # # print("Delta D:")
        # # print(DeltaD_aii[0][:4, :4])
        # # exit()
        # total_delta_comp_charge_G = self.get_compensation_charges(deltapsi_qnG, DeltaD_aii)
       
       
        # poisson_term_G = self.solve_poisson(fine_delta_tilde_n_G + total_delta_comp_charge_G)

        # soft_term_G = self.transform_to_coarse_grid_operator(poisson_term_G)
        # soft_term_R = self.calc.density.pd2.ifft(soft_term_G)
        # W_ap = self.get_charge_derivative_terms(poisson_term_G, DeltaD_aii)
        # ## END Compensation charge on fine grid

        # ## Compensation charge on coarse grid
        delta_tilde_n_G = self.get_delta_tilde_n(deltapsi_qnG)
        DeltaD_aii = self.get_density_matrix(deltapsi_qnG)
        comp_charge_G = self.get_comp_charges(deltapsi_qnG, DeltaD_aii)
        soft_term_G = self.solve_poisson_eq(delta_tilde_n_G + comp_charge_G)
        soft_term_R=  self.calc.density.pd2.ifft(soft_term_G)
        

        W_ap = self.get_charge_derivative_terms(soft_term_G, DeltaD_aii)
        # ## END Compensation charge on coarse grid

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
            kpt = self.wfs.mykpts[q_index]
            num_gs = kpt.psit.array.shape[1]
            if num_gs != len(wvf_G):
                raise ValueError("Number of G-vectors does not match length of vector")
            try:
                wvf_R = self.wfs.pd.ifft(wvf_G, q=q_index)
            except Exception as e:
                print("q=", q_index)
                raise e

            pt = self.wfs.pt
            
            c_ai = {a: val[level_index, :] for a, val in kpt.projections.items()}
                
            
            for a, c_i in c_ai.items():
                W_ii = unpack(W_ap[a])
                c_i[:] = W_ii.dot(c_i)#c_i.dot(W_ii) 
                ##Shape here is (1,13) is that correct?
                ##doesnt seem to matter
                #print(c_i.shape)

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


    def get_comp_charges(self, deltapsi_qnG, DeltaD_aii):
        setups = self.wfs.setups
        Q_aL = {}
        for a in DeltaD_aii:
            setup = setups[a]
            DeltaD_ii = DeltaD_aii[a]
            Delta_iiL = setup.Delta_iiL
            Q_L = np.einsum("ij, ij...", DeltaD_ii, Delta_iiL)
            Q_aL[a] = np.real(Q_L)

        comp_charge_G = self.calc.density.pd2.zeros()

        self.calc.density.ghat.add(comp_charge_G, Q_aL)
        
        return comp_charge_G


    def get_delta_tilde_n(self, deltapsi_qnG):
        pd = self.wfs.pd
        pd2 = self.calc.density.pd2
        delta_n_R = pd2.gd.zeros()
        for q_index, deltapsi_nG in deltapsi_qnG.items():
            for state_index, deltapsi_G in enumerate(deltapsi_nG):
                deltapsi_R = pd.ifft(deltapsi_G, q=q_index)
                wf_R = pd.ifft(self.wfs.mykpts[q_index].psit.array[state_index], q=q_index)
                delta_n_R += 2*np.real(deltapsi_R*wf_R.conj())
        return pd2.fft(delta_n_R)


    def get_fine_delta_tilde_n(self, deltapsi_qnG, return_real_space=False):
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

        if return_real_space:
            return fine_delta_n_R

        return pd3.fft(fine_delta_n_R)

    def solve_poisson_eq(self, charge_G):
        pd2 = self.calc.density.pd2
        G2 = pd2.G2_qG[0].copy()
        G2[0] = 1.0
        R = self.calc.atoms.cell[0, 0] / 2
        from ase.units import Bohr
        R /= Bohr
        
        return charge_G * 4 * np.pi / G2 * (1 - np.cos(np.sqrt(G2) * R))

    def solve_poisson(self, charge_G):
        #charge(q)
        pd3 = self.calc.density.pd3
        G2 = pd3.G2_qG[0].copy()
        G2[0] = 1.0

        R = self.calc.atoms.cell[0,0]/2
        from ase.units import Bohr
        R /= Bohr

        return charge_G* 4 * np.pi/G2*(1-np.cos(np.sqrt(G2)*R)) #charge_G* 4 * np.pi/G2
        
    def transform_to_coarse_grid_operator(self, fine_grid_operator):        
        dens = self.calc.density
        coarse_potential = dens.pd2.zeros()

        dens.map23.add_to1(coarse_potential, fine_grid_operator)

        return coarse_potential
        

    def get_charge_derivative_terms(self, poisson_term, DeltaD_aii):
        setups = self.wfs.setups

        G_aL = self.calc.density.ghat.integrate(poisson_term)
        W_ap = {}
        
        for a, glm_int_L in G_aL.items():
            W_ap[a] = setups[a].Delta_pL.dot(glm_int_L.T)[:,0]#.astype(np.complex128) 

                

        for a, setup in enumerate(self.wfs.setups):
            D_p = pack(DeltaD_aii[a])
            val = ((2*setup.M_pp).dot(D_p))

            W_ap[a] = W_ap[a] + val

        return W_ap
                

    def calculate_kplusq(self, qvector):
        self.k_plus_q = {}
        for index, kpt in enumerate(self.wfs.mykpts):
            kpt.psit.read_from_file()

            
            bzk_index = self.wfs.kd.ibz2bz_k[index]
            #Get kpt object for k+q
            k_plus_q_index = self.wfs.kd.find_k_plus_q(qvector, [bzk_index])
            assert len(k_plus_q_index) == 1
            k_plus_q_index = k_plus_q_index[0]
            ibz_k_plus_q_index = self.wfs.kd.bz2ibz_k[k_plus_q_index]
            k_plus_q_index = ibz_k_plus_q_index
            if ibz_k_plus_q_index >= len(self.wfs.mykpts):
                print(f"k_plus_q_index: {k_plus_q_index}")
                print(f"Len of mykpts: {len(self.wfs.mykpts)}. Len of bzk_kc: {len(self.wfs.kd.bzk_kc)}.")
                raise ValueError("index too large")
            k_plus_q_object = self.wfs.mykpts[ibz_k_plus_q_index]
            k_plus_q_object.psit.read_from_file()

            self.k_plus_q[index] = (k_plus_q_index, k_plus_q_object)


    def solve_sternheimer(self, apply_potential, qvector):
        
        deltapsi_qnG = {}
        nbands = self.nbands

        for index, kpt in enumerate(self.wfs.mykpts):
            #if kpt.s == 1:
            #    continue

            
            
            ##k_plus_q_index, k_plus_q_object = self.k_plus_q[index]



            ######################
            kpt.psit.read_from_file()

            
            #bzk_index = self.wfs.kd.ibz2bz_k[index]
            bzk_index = kpt.k ##TODO ASK JJ is this correct
            #Get kpt object for k+q
            k_plus_q_index = self.wfs.kd.find_k_plus_q(qvector, [bzk_index])
            assert len(k_plus_q_index) == 1
            k_plus_q_index = k_plus_q_index[0]
            ind2 = k_plus_q_index
            ibz_k_plus_q_index = self.wfs.kd.bz2ibz_k[k_plus_q_index]
            k_plus_q_index = ibz_k_plus_q_index
            if ibz_k_plus_q_index >= len(self.wfs.mykpts):
                print(f"k_plus_q_index: {k_plus_q_index}")
                print(f"Len of mykpts: {len(self.wfs.mykpts)}. Len of bzk_kc: {len(self.wfs.kd.bzk_kc)}.")
                raise ValueError("index too large")
                
            k_plus_q_object = self.wfs.mykpts[ibz_k_plus_q_index + kpt.s]
            #assert k_plus_q_object == kpt
            #k_plus_q_object = kpt
            k_plus_q_object.psit.read_from_file()

            ########################

            if k_plus_q_index not in deltapsi_qnG:
                numgs = k_plus_q_object.psit.array.shape[1]
                
                deltapsi_qnG[k_plus_q_index] = self.wfs.pd.zeros(x=(nbands,), q=k_plus_q_index)
            # else:
            #     pass
                #raise ValueError("Shouldnt go here")
                #print("Shouldnt go here")
            #print(f"array shape: {kpt.psit.array.shape}")
            #print(f"object shape: {kpt.psit.array.shape}")
            #print(f"deltapsi shape: {deltapsi_qnG[index].shape}")
            # eps_n = kpt.eps_n[:nvalence]
            # psi_nG = kpt.psit.array[:nvalence]


            # linop = self._get_LHS_linear_operator2(k_plus_q_object, eps_n, k_plus_q_index)

            # RHS_nG = self._get_RHS2(k_plus_q_object, psi_nG, apply_potential, k_plus_q_index)
            # deltapsi_nG, info = custombicgstab(linop, RHS_nG, max_iter=1000, tol=1e-2)
            # error = np.max(np.abs(linop.matvec(deltapsi_nG) - RHS_nG))
            # if error > 0:
            #     print(f"Error = {error}")
            # #if info != 0:
            # #    raise ValueError("bicgstab did not converge. Info: {}".format(info))

            # deltapsi_qnG[k_plus_q_index] += deltapsi_nG*self.kpt_weight[index]



            

            for state_index, (energy, psi) in enumerate(zip(kpt.eps_n, kpt.psit.array)):
                if state_index >= nbands:
                    break
                if kpt.f_n[state_index] < 0.01:
                    break


                t3 = time()
                linop = self._get_LHS_linear_operator(k_plus_q_object, energy, k_plus_q_index)

                #print(f"state index: {state_index}, index: {index}")
                #print(f"psi shape: {psi.shape}")


                #print(index, state_index, k_plus_q_object.psit.array.shape, kpt.psit.array.shape)



                RHS = self._get_RHS(k_plus_q_object, psi, apply_potential, state_index, k_plus_q_index, kpt, kpt.k)
                t4 = time()
                assert RHS.shape[0] == linop.shape[1]
                #print("got RHS")
                t1 = time()
                deltapsi_G, info = bicgstab(linop, RHS, maxiter=1000, tol=1e-8)
                t2 = time()
                #deltapsi_G = 0
                
                if info != 0:
                    print(f"Final error: {np.linalg.norm(linop.matvec(deltapsi_G) - RHS)}")
                    raise ValueError("bicgstab did not converge. Info: {}".format(info))
                
                #deltapsi_G = psi.copy()
                
                # print(f"Shape of deltapsi_nG: {deltapsi_qnG[k_plus_q_index].shape}. Shape of bicgstab result: {deltapsi_G.shape}")
                # print(f"kpt index: {index}, kplusqindex: {k_plus_q_index}, state_index: {state_index}")
                # print(f"Shape of kpt.psit.array: {kpt.psit.array.shape}")
                # print(f"Shape of kplusq.array: {k_plus_q_object.psit.array.shape}")
                deltapsi_qnG[k_plus_q_index][state_index] += deltapsi_G#*self.kpt_weight[index]

                self.t2s.append(t4-t3)

                self.t1s.append(t2-t1)
        return deltapsi_qnG




    def modular_solve_sternheimer(self, apply_potential, qvector):
        deltapsi_qnG = {}
        nbands = self.nbands

        for kpt_index, kpt in enumerate(self.wfs.mykpts):
            #if kpt.s == 1:
            #    continue
            kpq_index, kpq_object = self.get_k_plus_q(kpt_index, qvector)

            self.init_sternheimer_sol(deltapsi_qnG, kpq_index)

            deltapsi_qnG = self.solve_sternheimer_for_bands(deltapsi_qnG, (kpt_index, kpt), (kpq_indx, kpq_object), apply_potential)

        return deltapsi_qnG


    def get_k_plus_q(self, kpt_index, qvector):
        bzk_i = self.wfs.kd.ibz2bz_k[kpt_index]
        kpq_index = self.wfs.kd.find_k_plus_q(qvector, [bzk_i])[0]
        ibz_kpq_i = self.wfs.kd.bz2ibz_k[kpq_index]
        kpq_object = self.wfs.mykpts[ibz_kpq_i]

        return ibz_kpq_i, kpq_object

    def init_sternheimer_sol(self, deltapsi_qnG, kpq_index):
        #Set intial solution guess for this kpq equal to zero
        nbands = self.nbands
        deltapsi_qnG[kpq_index] = self.wfs.pd.zeros(x=(nbands,), q=kpq_index)
        


    def solve_sternheimer_for_bands(self, deltapsi_qnG, kpt_tuple, kpq_tuple, apply_potential):
        #Solve the Sternheimer equations for the given potential at the given kpt
        kpt_i, kpt_o = kpt_tuple
        kpq_i, kpq_o = kpq_tuple

        eps_n = kpt_o.eps_n
        psi_nG = kpt_o.psit.array

        for n, (eps, psi_G) in enumerate(zip(eps_n, psi_nG)):
            if n >= self.nbands:
                break
            
            LHS = self._get_LHS_linear_operator(kpq_o, eps, kpq_i)

            RHS = self._get_RHS(kpq_o, psi_G, apply_potential, n, kpq_i, kpt_o, kpt_i)
            
            deltapsi_G, info = bicgstab(LHS, RHS, maxiter=1000, tol=1e-8)

            deltapsi_qnG[kpq_i][n] = deltapsi_G
        return deltapsi_qnG




    def _get_LHS_linear_operator(self, kpt, energy, k_index):
        def mv(v):
            return self._apply_LHS_Sternheimer(v, kpt, energy, k_index)


        shape_tuple = (kpt.psit.array[0].shape[0], kpt.psit.array[0].shape[0])
        linop = LinearOperator(dtype=np.complex128, shape=shape_tuple, matvec=mv)

        return linop

    def _get_LHS_linear_operator2(self, kpt, eps_n, k_index):

        numGs = len(kpt.psit.array[0])

        def mv(v_nG):
            #result = np.zeros(nvalence*numGs, dtype=np.complex128)
            #for n in range(nvalence):
            #    result[n*numGs:(n+1)*numGs] = self._apply_LHS_Sternheimer(v[n*numGs:(n+1)*numGs], kpt, eps_n[n], k_index)
            #return result

            return self._apply_LHS_Sternheimer2(v_nG, kpt, eps_n, k_index)
        
        #shape_tuple = (nvalence*numGs, nvalence*numGs)

        #linop = LinearOperator(dtype=np.complex128, shape=shape_tuple, matvec=mv)
        linop = CustomLinearOperator(matvec=mv)
        return linop


    def _apply_LHS_Sternheimer(self, deltapsi_G, kpt, energy, k_index, custom_alpha=None):
        
        alpha = custom_alpha or 1
      
        result_G = alpha*self._apply_valence_projector(self._apply_overlap(deltapsi_G, k_index), kpt, k_index)
        result_G += self._apply_hamiltonian(deltapsi_G, kpt, k_index)

        result_G -= energy*self._apply_overlap(deltapsi_G, k_index)
        return result_G

    def _apply_LHS_Sternheimer2(self, deltapsi_nG, kpt, eps_n, k_index):
        
        alpha = 1
        
        result_nG = alpha*self._apply_LHS_valence_projector2(self._apply_LHS_overlap2(deltapsi_nG, k_index), kpt, k_index)
        result_nG += self._apply_hamiltonian2(deltapsi_nG, kpt, k_index)
        num_occ = self.nbands
        num_gs = len(deltapsi_nG)//num_occ
        result_nG -= np.einsum("i, ij -> ij", eps_n, self._apply_LHS_overlap2(deltapsi_nG, k_index))

        return result_nG



    def _apply_valence_projector(self, deltapsi_G, kpt, k_index):
        result_G = np.zeros_like(deltapsi_G)
        num_occ = self.nbands
        pd = self.wfs.pd
        kpt.psit.read_from_file()
        #print(f"K index: {k_index}")
        #print(f"array shape: {kpt.psit.array.shape}")
        for index, psi_G in enumerate(kpt.psit.array):
            if index >= num_occ: #We assume here that the array is ordered according to energy
                break
            if len(psi_G) != len(result_G):
                raise ValueError("HEY")
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


        intt_n = np.einsum("ii->i", pd.integrate(psi_nG, deltapsi_nG))

        result_nG = np.einsum("ij, i -> ij", psi_nG, intt_n)

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
        pt.initialize()
        c_ai = pt.integrate(wvf_G, q=q_index)

        for a, c_i in c_ai.items():
            dO_ii = self.wfs.setups[a].dO_ii
            assert np.allclose(dO_ii, dO_ii.T)
            ccopy = c_i.copy()
            c_i[:] = dO_ii.dot(c_i.T).T ##TODO ASK JJ about sqrt 4pi




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
        c_ani = pt.integrate(wvf_nG, q=q_index)

        for a, c_ni in c_ani.items():
            dO_ii = self.wfs.setups[a].dO_ii
            c_ni[:, :] = c_ni.dot(dO_ii)

        result_nG = wvf_nG.copy()
        pt.add(result_nG, c_ani, q=q_index)
        return result_nG
        

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
        
        result_nG = deltapsi_nG.copy()
        dpsi_nG = deltapsi_nG
        self.wfs.apply_pseudo_hamiltonian(kpt, self.calc.hamiltonian, dpsi_nG, result_nG)

        pt = self.wfs.pt
        c_ani = pt.integrate(dpsi_nG, q=k_index)
        
        for a, c_ni in c_ani.items():
            dH_ii = unpack(self.calc.hamiltonian.dH_asp[a][0])
            c_ni[:] = c_ni.dot(dH_ii)
        pt.add(result_nG, c_ani, q=k_index)
        return result_nG


    def _get_RHS(self, k_plus_q_pt, psi_G, apply_potential, level_index, k_plus_q_index, kpt, k_index): 
        v_psi_G = self.wfs.pd.zeros(q=k_plus_q_index)
        vpsi = apply_potential(psi_G, level_index, k_index)


        if len(v_psi_G) > len(vpsi):
            v_psi_G[:len(vpsi)] = vpsi
        else:
            v_psi_G[:] = vpsi[:len(v_psi_G)]

        X = self._apply_overlap(self._apply_valence_projector(v_psi_G, k_plus_q_pt, k_plus_q_index), k_plus_q_index)

        RHS_G = -(v_psi_G - X)


        return RHS_G
        

    def _get_RHS2(self, kpt, psi_nG, apply_potential, k_index):

        v_psi_nG = apply_potential(psi_nG, k_index)
        RHS_nG = -(v_psi_nG - self._apply_overlap2(self._apply_valence_projector2(v_psi_nG, kpt, k_index), k_index))
        return RHS_nG





def tests(ro):
    ###Overlap tests
    pd = ro.wfs.pd
    for kind, kpt in enumerate(ro.wfs.mykpts):
        for i1, wf1 in enumerate(kpt.psit.array):
            for i2, wf2 in enumerate(kpt.psit.array):
                inner_prod = pd.integrate(wf1, ro._apply_overlap(wf2,kind), kind) #wf1.dot(ro._apply_overlap(wf2, kind))
                if i1 == i2:
                    b = np.allclose(inner_prod, 1)
                else:
                    b = np.allclose(inner_prod, 0)
                if not b:
                    print(f"Overlap test failed for: kind: {kind}, i1: {i1}, i2: {i2}")
                    print(f"Inner product was: {inner_prod}")
                assert b
    print("Passed overlap test")

    ###Valence projector tests
    kindex = 0
    kpt = ro.wfs.mykpts[kindex]
    wf = kpt.psit.array[2]
    overlap_wf = ro._apply_overlap(wf, kindex)
    valence_overlap_wf = ro._apply_valence_projector(overlap_wf, kpt, kindex)
    overlap_valence_overlap_wf = ro._apply_overlap(valence_overlap_wf, kindex)
    valence_overlap_valence_overlap_wf = ro._apply_valence_projector(overlap_valence_overlap_wf, kpt, kindex)
    assert np.allclose(valence_overlap_valence_overlap_wf, valence_overlap_wf)
    print("Passed projector test")
    


    ### Hamiltonian tests
    kindex = 0
    kpt = ro.wfs.mykpts[kindex]
    wfindex = 0
    wf0 = kpt.psit.array[wfindex]
    Hwf0 = ro._apply_hamiltonian(wf0, kpt, kindex)
    ratios = []
    pd = ro.wfs.pd

    overlap_wf0 = ro._apply_overlap(wf0, kindex)
    energy = kpt.eps_n[wfindex]
    b = np.allclose(Hwf0, energy*overlap_wf0)
    if not b:
        print(f"Hamiltonian test failed")
        print(f"Max abs size: {np.max(np.abs(Hwf0-energy*overlap_wf0))}")
    assert b

    print("Passed Hamiltonian test")

    ###Solve Sternheimer tests
    qvector = np.array([0,0,0])
    kindex = 0
    bandindex = 0
    nvalence = ro.wfs.setups.nvalence//2
    deltapsi_nG = ro.wfs.pd.zeros(x=(nvalence,), q=kindex)
    deltapsi_nG[bandindex,4] = 1
    deltapsi_qnG = {index: ro.wfs.pd.zeros(x=(nvalence,), q=index) for index, _ in enumerate(ro.wfs.mykpts)}
    deltapsi_qnG[kindex] = deltapsi_nG

    
    apply_potential = ro.epsilon_potential(deltapsi_qnG)

    solution_qnG = ro.solve_sternheimer(apply_potential, qvector)

    ##Hmm SternheimerResponse.epsilon_potential is tightly coupled to solve Sternheimer - and these are both huge functions. Can we modularize it for easier testing and reading?

    kpt = ro.wfs.mykpts[kindex]
    should_be_zeros = ro._apply_valence_projector(ro._apply_overlap(solution_qnG[kindex][bandindex], kindex), kpt, kindex)
    b = np.allclose(should_be_zeros, 0)
    if not b:
        print("Solve Sternheimer test failed")
        print(f"Max abs size (should be zero): {np.max(np.abs(should_be_zeros))}")
    assert b
    

    ##Need to test that the result is correct

    print("Passed Solve Sternheimer test")






    ##Test get_k_plus_q
    #Input k_index: (int), q: (array)
    
    index = min(len(ro.wfs.kd.bzk_kc)-1, 1)
    q = ro.wfs.kd.bzk_kc[index] - ro.wfs.kd.bzk_kc[0]
    k_index = 0
    
    kpq_i, kpq_o = ro.get_k_plus_q(k_index, q)
    
    #b = kpq_i == self.wfs.kd.find_k_plus_q(q, [k_index])[0]
    b = np.allclose(ro.wfs.kd.ibzk_kc[kpq_o.k], q + ro.wfs.kd.bzk_kc[0])
    if not b:
        print("Get k plus q failed")
    assert b
    print("Passed get k plus q test")



    ##Test init solution
    deltapsi_qnG = {}
    kpq_index = 0
    ro.init_sternheimer_sol(deltapsi_qnG, kpq_index)
    
    b = np.allclose(deltapsi_qnG[kpq_index], 0)
    if not b:
        print("Init Sternheimer solution test failed")
    assert b
    print("Passed Init Sternheimer solution")




    ##Test get RHS
    kpt_o = ro.wfs.mykpts[0]
    kpt_i = 0
    kpq_o = kpt_o
    kpq_i = kpt_i
    psi_G = kpt.psit.array[0]
    n = 0
    apply_potential = lambda a, b, c : a
    RHS_G = ro._get_RHS(kpq_o, psi_G, apply_potential, n, kpq_i, kpt_o, kpt_i)
    
    pRHS_G = ro._apply_valence_projector(RHS_G, kpq_o, kpq_i)
    assert np.allclose(ro.wfs.pd.integrate(psi_G, RHS_G), 0)
    assert np.allclose(pRHS_G, 0)


    print("Passed get RHS test")
    




    ##Test apply LHS
    kpt_o = ro.wfs.mykpts[0]
    kpt_i = 0
    eps = kpt_o.eps_n[0]
    deltapsi_G = kpt_o.psit.array[0]+1
    overlapped = ro._apply_overlap(deltapsi_G, k_index)
    deltapsi_G = deltapsi_G - ro._apply_valence_projector(overlapped, kpt_o, kpt_i)
    result_G = ro._apply_LHS_Sternheimer(deltapsi_G, kpt_o, eps, kpt_i)

    results = []
    for alpha in [1,2,10]:
        res = ro._apply_LHS_Sternheimer(deltapsi_G, kpt_o, eps, kpt_i, custom_alpha=alpha)
        results.append(res)

    for res in results:
        for res2 in results:
            assert np.allclose(res, res2)
    

    print("Passed apply LHS test")
        






    ##Test solve sternheimer for bands
    nbands = ro.nbands
    perturb = ro.wfs.pd.zeros(x=(nbands,), q=0)
    perturb[0] = ro.wfs.mykpts[0].psit.array[0]
    in_qnG = {0: perturb}
    apply_potential = ro.epsilon_potential(in_qnG)
    apply_potential = lambda a, b, c : a

    deltapsi_qnG = {}
    kpt_index = 0
    kpt_o = ro.wfs.mykpts[kpt_index]
    qvector = np.array([0,0,0])
    kpq_i, kpq_o = ro.get_k_plus_q(kpt_index, qvector)


    ro.init_sternheimer_sol(deltapsi_qnG, kpt_index)
    deltapsi_qnG = ro.solve_sternheimer_for_bands(deltapsi_qnG, (kpt_index, kpt_o), (kpq_i, kpq_o), apply_potential)
    
    for q, deltapsi_nG in deltapsi_qnG.items():
        for n, deltapsi_G in enumerate(deltapsi_nG):
            overlapped = ro._apply_overlap(deltapsi_G, 0)
            val = ro.wfs.pd.integrate(kpt_o.psit.array[0], overlapped, 0)
            target = 0
            b = np.allclose(val, target)
            if not b:
                print(f"Max abs: {np.max(np.abs(val))}")
            assert b
                
    print("Passed solve sternheimer for all bands")










    print("Passed all tests")
    pass




if __name__=="__main__":
    #filen = "test.gpw"
    import sys
    filen = sys.argv[1] + ".gpw"
    outname = sys.argv[1] + ".out"
    if len(sys.argv) > 2 and sys.argv[2] == "runtests":
        ro = SternheimerResponse(filen, runstuff=False)
        tests(ro)
        exit()

    cutoff = float(sys.argv[2])

    import os

    from ase.build import bulk
    from ase import Atoms
    from gpaw import PW, FermiDirac
    
    
    atoms = Atoms(sys.argv[1], cell=(10, 10, 10), magmoms=[1])

    setup = sys.argv[3]
    if setup != "paw":
        calc = GPAW(mode=PW(cutoff, force_complex_dtype=True),
                    xc="LDA",
                    setups=setup)
    else:
        calc = GPAW(mode=PW(cutoff, force_complex_dtype=True),
                    xc="LDA")

    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()



    prefix = setup + "_" + str(int(cutoff)) + "_"
    filen = prefix + filen
    calc.write(filen, mode = "all")

    respObj = SternheimerResponse(filen)
    eigval, eigmode_R = respObj.eigval, respObj.fdnt_R
    np.save(prefix + "grid", respObj.wfs.gd.get_grid_point_coordinates()[0, :, 0, 0])
    np.save(prefix + "eigmode_x", eigmode_R[:,0,0])
    np.save(prefix + "eigval", eigval)
    os.remove(filen)

















    # a = 10
    # c = 5
    # d = 0.74
    #     #atoms = Atoms("H2", positions=([c-d/2, c, c], [c+d/2, c,c]),
    #     #       cell = (a,a,a))
    #     # atoms = bulk("Si", "diamond", 5.43)
    #     # calc = GPAW(mode=PW(250, force_complex_dtype=True),
    #     #             xc ="PBE",                
    #     #             kpts={"size":(1,1,1), "gamma":True},
    #     #             random=True,
    #     #             symmetry="off",#{"point_group": False},
    #     #             occupations=FermiDirac(0.0001),
    #     #             basis="dzp",
    #     #             convergence={"eigenstates": 4e-14},
    #     #             #setups="ah"/"AH",
    #     #            txt = outname)
