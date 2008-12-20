import numpy as np
from numpy import linalg as la
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.lfc import BasisFunctions
from gpaw.utilities import unpack
from gpaw.lcao.tools import get_bf_centers
from ase import Hartree


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


def get_p_and_tci(calc, q=0, s=0):
    spos_ac = calc.atoms.get_scaled_positions()
    setups = calc.wfs.setups
    domain = calc.domain
    tci = TwoCenterIntegrals(domain, setups, calc.wfs.gamma, calc.wfs.ibzk_qc)
    tci.set_positions(spos_ac)

    nq = len(calc.wfs.ibzk_qc)
    nao = calc.wfs.setups.nao
    S_qMM = np.zeros((nq, nao, nao))
    T_qMM = np.zeros((nq, nao, nao))
    #setup basis functions
    bfs = BasisFunctions(calc.gd, [setup.phit_j for setup in calc.wfs.setups],
                         calc.wfs.kpt_comm, cut=True)
    bfs.set_positions(spos_ac)
    
    P_aqMi = {}
    for a in bfs.my_atom_indices:
        ni = calc.wfs.setups[a].ni
        P_aqMi[a] = np.zeros((nq, nao, ni), calc.wfs.dtype)

    tci.calculate(spos_ac, S_qMM, T_qMM, P_aqMi)

    vt_G = calc.hamiltonian.vt_sG[s]
    H_MM = np.zeros((nao, nao)) 
    bfs.calculate_potential_matrix(vt_G, H_MM, q)
    #non-local corrections
    for a, P_qMi in P_aqMi.items():
        P_Mi = P_qMi[q]
        dH_ii = unpack(calc.hamiltonian.dH_asp[a][s])
        H_MM +=  np.dot(P_Mi, np.inner(dH_ii, P_Mi).conj())

    H_MM += T_qMM[q]#kinetic energy
    S_MM = S_qMM[q]
    tri = np.tri(nao)
    tri.flat[::nao + 1] = 0.5
    H_MM = tri * H_MM
    H_MM += H_MM.copy().T
    H_MM *= Hartree
    S_MM = tri * S_MM
    S_MM += S_MM.copy().T
    # Calculate projections
    V_nM = 0
    #non local corrections
    for a, P_ni in calc.wfs.kpt_u[q].P_ani.items():
        dS_ii = calc.wfs.setups[a].O_ii
        P_Mi = P_aqMi[a][q]
        V_nM += np.dot(P_ni, np.inner(dS_ii, P_Mi).conj())
    #Hack XXX, not needed when BasisFunctions get
    #an integrate method.
    spos_Ac = []
    spline_Aj = []
    for a, spos_c in enumerate(spos_ac):
        for phit in calc.wfs.setups[a].phit_j:
            spos_Ac.append(spos_c)
            spline_Aj.append([phit])
            
    bfs = LFC(calc.gd, spline_Aj, calc.wfs.kpt_comm, cut=True)
    bfs.set_positions(np.array(spos_Ac))
    V_Ani = bfs.dict(calc.wfs.nbands)
    bfs.integrate(calc.wfs.kpt_u[q].psit_nG, V_Ani, q)
    M1 = 0
    for A in range(len(spos_Ac)):
        V_ni = V_Ani[A]
        M2 = M1 + V_ni.shape[1]
        V_nM[:, M1:M2] += V_ni
        M1 = M2

    return V_nM, H_MM, S_MM
  

class ProjectedWannierFunctions:
    def __init__(self, projections, h_lcao, s_lcao, eps_n, 
                 L=None, M=None, N=None, fixedenergy=None):
        """projections[n,i] = <psi_n|f_i>
           h_lcao[i1, i2] = <f_i1|h|f_i2>
           s_lcao[[i1, i2] = <f_i1|f_i2>
           eps_n: Exact eigenvalues
           L: Number of extra degrees of freedom
           M: Number of states to exactly span
           N: Total number of bands in the calculation
           
           Methods:
           -- get_hamiltonian_and_overlap_matrix --
           will return the hamiltonian and identity operator
           in the projected wannier function basis. 
           The following steps are performed:
            
           1) calculate_edf       -> self.b_il
           2) calculate_rotations -> self.Uo_mi an self.Uu_li
           3) calculate_overlaps  -> self.S_ii
           4) calculate_hamiltonian_matrix -> self.H_ii

           -- get_eigenvalues --
           gives the eigenvalues of of the hamiltonian in the
           projected wannier function basis.
           
           """
        
        self.eps_n = eps_n
        self.V_ni = projections   #<psi_n1|f_i1>
        self.s_lcao_ii = s_lcao #F_ii[i1,i2] = <f_i1|f_i2>
        self.h_lcao_ii = h_lcao
        if fixedenergy!=None:
            M = sum(eps_n<=fixedenergy)
            L = self.V_ni.shape[1] - M
        if L==None and M!=None:
            L = self.V_ni.shape[1] - M
        if M==None and L!=None:
            M = self.V_ni.shape[1] - L
        self.M = M              #Number of occupied states
        self.L = L              #Number of EDF's
        if N==None:
            N = M + L
        self.N = N              #Number of bands
        print "(N, M, L) = (%i, %i, %i)" % (N, M, L)
        print "Number of PWFs:", M + L

    def get_hamiltonian_and_overlap_matrix(self, useibl=True):
        self.calculate_edf(useibl=useibl)
        self.calculate_rotations()
        self.calculate_overlaps()
        self.calculate_hamiltonian_matrix()
        return self.H_ii, self.S_ii

    def calculate_edf(self, useibl=True):
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
            self.Fu_ii = self.s_lcao_ii - self.Fo_ii
        else:
            Vu_ni = self.V_ni[self.M:self.N]
            self.Fu_ii = np.dot(dagger(Vu_ni), Vu_ni)

        b_i, b_ii = la.eigh(self.Fu_ii)
        ls = b_i.real.argsort()[-self.L:] #edf indices (L largest eigenvalues)
        self.b_i = b_i #Eigenvalues used for determining the EDF
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
        self.S_ii = Wo_ii + Wu_ii
        eigs = la.eigvalsh(self.S_ii)
        self.conditionnumber = eigs.max() / eigs.min()

    def calculate_hamiltonian_matrix(self):
        """Calculate H_ij = H^o_ij + H^u_ij 
        """
        Uo_ni = self.Uo_ni
        Uu_li = self.Uu_li
        b_il = self.b_il

        epso_n = self.eps_n[:self.M]

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
        eigs = la.eigvals(la.solve(self.S_ii, self.H_ii)).real
        eigs.sort()
        return eigs

    def get_lcao_eigenvalues(self):
        eigs = la.eigvals(la.solve(self.s_lcao_ii, self.h_lcao_ii)).real
        eigs.sort()
        return eigs


    def get_norm_of_projection(self):
        norm_n = np.zeros(self.N)
        Sinv_ii = la.inv(self.S_ii)
        Uo_ni = self.Uo_ni
        norm_n[:self.M] = dots([Uo_ni, Sinv_ii, dagger(Uo_ni)]).diagonal()
        Vu_ni = self.V_ni[self.M:self.N]
        Pu_ni = dots([Vu_ni, self.b_il, self.Uu_li])
        norm_n[self.M:self.N] = dots([Pu_ni, Sinv_ii, dagger(Pu_ni)]).diagonal()
        return norm_n
        


if __name__=='__main__':
    from ase import molecule
    from gpaw import GPAW
    from projected_wannier import ProjectedWannierFunctions
    from projected_wannier import get_p_and_tci

    atoms = molecule('C6H6')
    atoms.center(vacuum=2.5)
    calc = GPAW(basis='sz',h=0.3, nbands=20)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    eps_n = calc.get_eigenvalues()
    V_nM, H_MM, S_MM = get_p_and_tci(calc, q=0, s=0)
    pwf = ProjectedWannierFunctions(V_nM, h_lcao=H_MM, s_lcao=S_MM, eps_n=eps_n, M=17, N=20)
    h, s = pwf.get_hamiltonian_and_overlap_matrix(useibl=True)
    eps2_n = pwf.get_eigenvalues()
    print "Difference in eigenvalues:", abs(eps2_n[:len(eps_n)] - eps_n)
    print "norm of proj.:", pwf.get_norm_of_projection()
    print "condition number:", pwf.conditionnumber

    
