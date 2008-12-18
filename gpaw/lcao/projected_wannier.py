import numpy as np
from numpy import linalg as la
from gpaw.lfc import LocalizedFunctionsCollection as LFC

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
            L2 = np.dot(dagger(col1), col1) + np.dot(dagger(col2), col2)
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
        self.V_ni = projections #V_ni[n1,i1] = <psi_n1|f_i1>
        self.F_ii = overlaps    #F_ii[i1,i2] = <f_i1|f_i2>a
        self.h_lcao_ii = h_lcao
        self.M = M              #Number of occupied states
        self.L = L              #Number of EDF's
        self.N = N              #Number of bands
        self.nlf = len(self.F_ii)#number of localized functions

    def calculate_overlaps(self):
        Uo_ni = self.Uo_ni
        Uu_li = self.Uu_li
        b_il = self.b_il
        Fu_ii = self.Fu_ii

        Wo_ii = np.dot(dagger(Uo_ni), Uo_ni)
        Wu_ii = dots([dagger(Uu_li), dagger(b_il), Fu_ii, b_il, Uu_li])
        self.W_ii = Wo_ii + Wu_ii
        eigs = la.eigvalsh(self.W_ii)
        self.conditionumber = eigs.max() / eigs.min()

    def calculate_hamiltonian_matrix(self):
        """Calculate H_ij = H^o_ij + H^u_ij 
           h_lcao: Hamiltonian in the lcao basis. Only
                   needed in the infinite band limit.
        """
        epso_n = self.eps_n[:self.M]
        Uo_ni = self.Uo_ni
        Ho_ii = np.dot(dagger(Uo_ni) * epso_n, Uo_ni)
        self.Ho_ii = Ho_ii
        if self.h_lcao_ii!=None:
            print "Using h_lcao"
            Vo_ni = self.V_ni[:self.M]
            Hu_ii = self.h_lcao_ii - np.dot(dagger(Vo_ni) * epso_n, Vo_ni)
        else:
            Uu_ni = self.Uu_ni
            Hu_ii = np.dot(dagger(Uu_ni) * epso_n, Uu_ni)
        
        self.Hu_ii = Hu_ii
        self.H_ii = Ho_ii + Hu_ii

    
    def calculate_rotations(self):
        """calculate rotations"""
        Uo_ni = self.V_ni[:self.M]
        Uu_li = np.dot(dagger(self.b_il), self.Fu_ii)
        #Normalize such that <omega_i|omega_i> = 1
        normalize(Uo_ni, Uu_li)
        self.Uo_ni = Uo_ni
        self.Uu_li = Uu_li


    def calculate_edf(self, useibl=True, N=-1):
        """           Calculate the coefficients b_il
           in the expansion of the EDF: 
                        |phi_l> = sum_i b_il |f^u_i>,
           in terms of |f^u_i> = P^u|f_i>.
           N is the total number of bands"""
        
        if self.L==0:
            print "L=0. No EDF to be calculated!"
        
        Vo_ni = self.V_ni[:self.M]
        self.Fo_ii = np.dot(dagger(Vo_ni), Vo_ni)

        if useibl:
            self.Fu_ii = self.F_ii - self.Fo_ii
        else:
            self.Vu_ni = self.V_ni[self.M:self.N]
            self.Fu_ii = npy.dot(dagger(Vu_ni), Vu_ni)

        b_i, b_ii = la.eigh(self.Fu_ii)
        print "Eigenvalues:", b_i.real
        ls = b_i.real.argsort()[-self.L:] #edf indices (L largest eigenvalues)
        print "Using eigenvalue number:", ls
        self.b_i = b_i
        self.b_l = b_i[ls] 
        b_il = b_ii[:, ls] #pick out the eigenvectors of the largest eigenvals.
        normalize2(b_il, self.Fu_ii) #normalize the EDF: <phi_l|phi_l> = 1
        self.b_il = b_il

    def get_eigenvalues(self):
        return la.eigvals(la.solve(self.W_ii, self.H_ii))

    
#def get_projections(self, q):
#    calc = self.calc
#    spos_ac = calc.atoms.get_scaled_positions()
#    V_nM = 0
#    #non local part
#    for a, P_ni in calc.wfs.kpt_u[q].P_ani.items():
#        dS_ii = calc.wfs.setups[a].O_ii
#        P_Mi = P_aqMi[a][q]
#        V_nM += np.dot(P_ni, np.inner(dS_ii, P_Mi).conj())
#    #Soft part (should use gpaw.lfs.BasisFunctions.integrate when it
#    #is implemented).
#    spos_Ac = []
#    spline_Aj = []
#    for a, spos_c in enumerate(spos_ac):
#        for phit in calc.wfs.setups[a].phit_j:
#            spos_Ac.append(spos_c)
#            spline_Aj.append([phit])
#    #setup LFC
#    bfs = LFC(calc.gd, spline_Aj, calc.wfs.kpt_comm, cut=True)
#    bfs.set_positions(np.array(spos_Ac))
#    V_Ani = bfs.dict(calc.wfs.nbands)
#    bfs.integrate(calc.wfs.kpt_u[q].psit_nG, V_Ani, q)
#    M1 = 0
#    #Unfold the projections
#    for A in range(len(spos_Ac)):
#        V_ni = V_Ani[A]
#        M2 = M1 + V_ni.shape[1]
#        V_nM[:, M1:M2] += V_ni
#        M1 = M2
#    return V_nM


#INFO
#f_i localized projector functions 
#F_ii[i1, i2] = <f_i1|f_i2> overlap matrix with elements.
#Fo_ii[i1, i2] = <f_i1|P^o|f_i2>, "occupied" overlap matrix.
#Fu_ii[i1, i2] = <f_i1|P^u|f_i2>, "unoccupied" overlap matrix. Note, 
#in the infinite band limit it holds that Fu_ii = F_ii - Fo_ii
#
#The calculation of Fo_ii[i1,i2] = <f_i1|P^o|f_i2> is easely done 
#ones we have the overlaps:
#           Vo_ni[n1,i1] = <psi_n1|f_i1>,
#as: 
#           Fo_ii = Vo_ni^d Vo_ni.
#I
#           Fu_ii = Vu_ni^d Vu_ni.
#In the inifinite band limit this can be written as
#           Fu_ii = F_ii - Fo_ii
#Input:
#Vo_ni (and Vu_ni)
#1)
#

