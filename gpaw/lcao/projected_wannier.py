

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
            L2 = np.dot(dagger(col1), col1) + np.dot(dagger(col2), col2)
            N = 1 / np.sqrt(L2)
            col1 *= N
            col2 *= N
       
def normalize2(C, S):
    norm = 1.0 / np.sqrt(np.dot(np.dot(dagger(C), S), C).diagonal())
    C *= norm


class ProjectedFunctions:
    def __init__(self, projections, overlaps, L, M, N=-1):
        
        self.V_ni = projections #V_ni[n1,i1] = <psi_n1|f_i1>
        self.F_ii = overlaps    #F_ii[i1,i2] = <f_i1|f_i2>
        self.M = M              #Number of occupied states
        self.L = L              #Number of EDF's
        self.N = N              #Number of bands
        self.nlf = len(self.F_ii)#number of localized functions

    def calculate_overlaps(self):
        Uo_ni = self.Uo_ni
        Uu_li = self.Uo_li
        b_il = self.b_il
        Fu_ii = self.Fu_ii

        Wo_ii = np.dot(dagger(Uo_ni), Uo_ni)
        Wu_ii = dots([dagger(Uu_li), dagger(b_il), Fu_ii, b_il, Uu_li])
        self.W_ii = Wo_ii + Wu_ii

    def calculate_hamiltonian_matrix(self, epso_n, epsu_n=None, h_lcao=None):
        """Calculate H_ij = H^o_ij + H^u_ij 
           epso_n: occupied eigenvalues
           epsu_n: unoccupied eigenvalues. Only
                   needed in the finite band case.
           h_lcao: Hamiltonian in the lcao basis. Only
                   needed in the infinite band limit.
        """
        Uo_ni = self.Uo_ni
        Ho_ii = dots([dagger(Uo_ni), epso_n, Uo_ni])

        if h_lcao!=None:
            Vo_ni = self.V_ni[:M]
            Hu_ii = h_lcao - np.dot(dagger(Vo_ni) * epso_n, Vo_ni)
        else:
            Uu_ni = self.Uu_ni
            Hu_ii = dots([dagger(Uu_ni), epso_n, Uu_ni])
    
        self.H_ii = Ho_ii + Hu_ii

    
    def calculate_rotations(self):
        """calculate rotations"""
        Uo_ni = self.V_ni[:self.M]
        Uu_li = npy.dot(dagger(self.b_il), self.Fu_ii)
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
        ls = b_i.real.argsort()[-self.L:] #edf indices (L largest eigenvalues)
        self.b_i = b_i
        self.b_l = b_i[ls] 
        b_il = b_ii[:, ls] #pick out the eigenvectors of the largest eigenvals.
        normalize2(b_il, Fu_ii) #normalize the EDF: <phi_l|phi_l> = 1
        self.b_il = b_il


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

