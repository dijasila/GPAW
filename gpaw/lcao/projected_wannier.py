import numpy as np
from numpy import linalg as la

def dots(Ms):
    N = len(Ms)
    x = 1
    i = 0
    while i < N:
        x = np.dot(x, Ms[i])
        i+=1
    return x        

def dagger(a):
    return np.conj(a.T)

def normalize(U, U2=None):
    if U2==None:
        for col in U.T:
            col /= la.norm(col)
    else:
         for col1, col2 in zip(U.T, U2.T):
            L2 = np.vdot(col1, col1) + np.vdot(col2, col2)
            N = 1 / np.sqrt(L2)
            col1 *= N
            col2 *= N
       
def normalize2(C, S):
    norm = 1.0 / np.sqrt(np.dot(np.dot(dagger(C), S), C).diagonal())
    C *= norm


class ProjectedFunctions:
    def __init__(self, projections, h_lcao, overlaps, eps_n, L, M, N=-1):
        """projections: <psi_n|f_i>
           overlaps: <f_i1|f_i2>
           L: Number of extra degrees of freedom
           M: Number of states to exactly span
           N: Total number of bands to consider
           Steps:
           1) calculate_edf       -> self.b_il
           2) calculate_rotations -> self.Uo_mi an self.Uu_li
           3) calculate_overlaps  -> self.S_ii
           4) calculate_hamiltonian_matrix -> self.H_ii
           
           """
        self.eps_n = eps_n
        self.V_ni = projections   #<psi_n1|f_i1>
        self.F_ii = overlaps #F_ii[i1,i2] = <f_i1|f_i2>
        self.h_lcao_ii = h_lcao
        self.M = M              #Number of occupied states
        self.L = L              #Number of EDF's
        self.N = N              #Number of bands

    def calculate_edf(self, useibl=True, N=-1):
        """Calculate the coefficients b_il
           in the expansion of the EDF: 
                        |phi_l> = sum_i b_il |f^u_i>,
           in terms of |f^u_i> = P^u|f_i>.

           To use the infinite band limit set useibl=True.
           N is the total number of bands to use
        """
        
        if self.L==0:
            print "L=0. No EDF to be calculated!"
        
        Vo_ni = self.V_ni[:self.M]
        self.Fo_ii = np.dot(dagger(Vo_ni), Vo_ni)

        if useibl:
            self.Fu_ii = self.F_ii - self.Fo_ii
        else:
            Vu_ni = self.V_ni[self.M:self.N]
            self.Fu_ii = np.dot(dagger(Vu_ni), Vu_ni)

        b_i, b_ii = la.eigh(self.Fu_ii)
        print "Eigenvalues:", b_i.real
        ls = b_i.real.argsort()[-self.L:] #edf indices (L largest eigenvalues)
        print "Using eigenvalue number:", ls
        self.b_i = b_i
        self.b_l = b_i[ls] 
        b_il = b_ii[:, ls] #pick out the eigenvectors of the largest eigenvals.
        normalize2(b_il, self.Fu_ii) #normalize the EDF: <phi_l|phi_l> = 1
        self.b_il = b_il

    def calculate_rotations(self):
        """calculate rotations"""
        Uo_ni = self.V_ni[:self.M].copy()
        Uu_li = np.dot(dagger(self.b_il), self.Fu_ii)
        #Normalize such that <omega_i|omega_i> = 1
        normalize(Uo_ni, Uu_li)
        self.Uo_ni = Uo_ni
        self.Uu_li = Uu_li

    def calculate_overlaps(self):
        Uo_ni = self.Uo_ni
        Uu_li = self.Uu_li
        b_il = self.b_il
        Fu_ii = self.Fu_ii

        Wo_ii = np.dot(dagger(Uo_ni), Uo_ni)
        Wu_ii = dots([dagger(Uu_li), dagger(b_il), Fu_ii, b_il, Uu_li])
        self.Wo_ii = Wo_ii
        self.Wu_ii = Wu_ii
        self.W_ii = Wo_ii + Wu_ii
        eigs = la.eigvalsh(self.W_ii)
        self.conditionnumber = eigs.max() / eigs.min()

    def calculate_hamiltonian_matrix(self):
        """Calculate H_ij = H^o_ij + H^u_ij 
        """
        Uo_ni = self.Uo_ni
        Uu_li = self.Uu_li
        b_il = self.b_il

        epso_n = self.eps_n[:self.M]
        print "using occupied eigenvalues:", epso_n

        self.Ho_ii = np.dot(dagger(Uo_ni) * epso_n, Uo_ni)
        if self.h_lcao_ii!=None:
            print "Using h_lcao"
            Vo_ni = self.V_ni[:self.M]
            Huf_ii = self.h_lcao_ii - np.dot(dagger(Vo_ni) * epso_n, Vo_ni)
        else:
            print "Not using h_lcao"
            epsu_n = self.eps_n[self.M:self.N]
            Vu_ni = self.V_ni[self.M:self.N]
            Huf_ii = np.dot(dagger(Vu_ni) * epsu_n, Vu_ni)
       
        self.Hu_ii = dots([dagger(Uu_li), dagger(b_il), Huf_ii, b_il, Uu_li])
        self.H_ii = self.Ho_ii + self.Hu_ii

    def get_eigenvalues(self):
        eigs = la.eigvals(la.solve(self.W_ii, self.H_ii)).real
        eigs.sort()
        return eigs

    def get_lcao_eigenvalues(self):
        eigs = la.eigvals(la.solve(self.F_ii, self.h_lcao_ii)).real
        eigs.sort()
        return eigs



    
