import numpy as np
from gpaw.utilities import unpack2

class HubU:
    def __init__(self, paw):
        self.paw = paw
        print 'dafadsjhfasjklfhsda'

    def get_linear_response_U0(self,a,n,l,s,
                               HubU_dict = {},
                               alpha = 0.002,
                               scale = 1,
                               NbP = 1,
                               ):
        """
        spin {0,1,2} (2: on both)
        0.05eV/Hartree=0.00183746618: potential shift  
        """
        if a not in HubU_dict:
            HubU_dict[a]={}
        if n not in HubU_dict[a]:
            HubU_dict[a][n]={}
        if l not in HubU_dict[a][n]:
            HubU_dict[a][n][l]={}
        if s not in HubU_dict[a][n][l]:
            if s == 2:
                HubU_dict[a][n][l][0]={'alpha':alpha}
                HubU_dict[a][n][l][1]={'alpha':alpha}
            else:
                HubU_dict[a][n][l][s]={'alpha':alpha}      
        
        c = self.paw
        c.calculate()
        D_p = c.density.D_asp[a][0]
        
        Nanls_ref = np.trace(c.hamiltonian.aoom(unpack2(D_p),
                          a,n,l,NbP=NbP,scale=scale)[0])
        
        c.hamiltonian.set_hubbard_u(HubU_dict = HubU_dict)
        c.scf.HubAlphaIO = True
        c.scf.reset()
        c.calculate()
        
        D_p = c.scf.D_asp_0[a][0]
        Nanls_0 = np.trace(c.hamiltonian.aoom(unpack2(D_p),
                          a,n,l,NbP=NbP,scale=scale)[0])
        
        D_p = c.density.D_asp[a][0]
        Nanls_KS = np.trace(c.hamiltonian.aoom(unpack2(D_p),
                          a,n,l,NbP=NbP,scale=scale)[0])
        
        Xres_0 = (Nanls_0-Nanls_ref)/alpha
        Xres_KS = (Nanls_KS-Nanls_ref)/alpha
        
        U0 = Xres_0**-1 - Xres_KS**-1
        return U0
        #print 'Linear response Hubbard U0 on Sc 3d orbital', U0        @
