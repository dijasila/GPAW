import numpy as np
from gpaw.utilities import unpack2
from copy import copy
import gpaw.mpi as mpi

class HubU:
    def __init__(self, paw):
        self.paw = paw
        
        # Is this allowed or should world be an input to the class?
        self.world = mpi.world
        
    def get_MS_linear_response_U0(self,
                               HubU_IO_dict, 
                               HubU_dict = {},
                               background = True,
                               alpha = 0.002,
                               scale = 1,
                               NbP = 1,
                               ):
        """
        HubU_IO_dict = {a:{n:{l:
                                #options : 0,1,2
                            }}}}
        0: off
        1: site on, spin follow each other
        2: site on, spins independent
        
        where different sites are enabled by setting them to 1:
        if s=2 is set, then the site will be treat the two spin equally 
        and the spins are not to be set separately. 

        Background specify that the background increase in charge is included.
        """
         
        HubU_dict_base = copy(HubU_dict)
        
        c = self.paw
        c.calculate()
        
        # Get number of spins
        nspin=np.array([0])
        if c.density.rank_a[0] == self.world.rank:
            nspin = len(c.density.D_asp[0])
            nspin=np.array([nspin])
        self.world.broadcast(nspin, c.density.rank_a[0])
        nspin=nspin[0]
        
        # Find number of Hub sites
        
        Nanls_ref = {}
        sites = 0
        for a in HubU_IO_dict:
            Nanls_ref[a]={} 
            for n in HubU_IO_dict[a]:
                Nanls_ref[a][n]={}
                for l in HubU_IO_dict[a][n]:
                    Nanls_ref[a][n][l]={}
                    if nspin == 2:
                        if (HubU_IO_dict[a][n][l]==1):
                            sites+=1
                            Nanls_ref[a][n][l][0] = (
                                                    self.get_Nocc(a, n, l, 0,
                                                          scale=scale,NbP = NbP)  
                                                  + self.get_Nocc(a, n, l, 1,
                                                          scale=scale,NbP = NbP)
                                                    )
                        elif HubU_IO_dict[a][n][l]==2:
                            sites+=2
                            Nanls_ref[a][n][l][0] = self.get_Nocc(a, n, l, 0,
                                                          scale=scale,NbP = NbP)  
                            Nanls_ref[a][n][l][1] = self.get_Nocc(a, n, l, 1,
                                                          scale=scale,NbP = NbP)                                                      
                    else:
                        if (HubU_IO_dict[a][n][l]==1 or
                            HubU_IO_dict[a][n][l]==2):
                            sites+=1
                        Nanls_ref[a][n][l][0] = self.get_Nocc(a, n, l, 0,
                                                              scale=scale,NbP = NbP) 
        
        X0, Xks =   np.zeros((sites+background,sites+background)), \
                    np.zeros((sites+background,sites+background))
        
        c.scf.HubAlphaIO = True
        HubU_alpha_dict = {}
        ii = 0
        for a in HubU_IO_dict:
            if a not in HubU_dict_base:
                HubU_dict_base[a] = {}
            #HubU_alpha_dict[a]={}
            for n in HubU_IO_dict[a]:
                if n not in HubU_dict_base[a]:
                    HubU_dict_base[a][n] = {}
                #HubU_alpha_dict[a][n]={}
                for l in HubU_IO_dict[a][n]:
                    if l not in HubU_dict_base[a][n]:
                        HubU_dict_base[a][n][l] = {}
                    #HubU_alpha_dict[a][n][l]={}
                    
                    for s in range(HubU_IO_dict[a][n][l]):
                        if s==1 and nspin==1:
                            continue
                        
                        if s not in HubU_dict_base[a][n][l]:
                            HubU_dict_base[a][n][l][s] = {}
                        #HubU_alpha_dict[a][n][l][s]={}
                        
                        HubU_alpha_dict = copy(HubU_dict_base)
                        if HubU_IO_dict[a][n][l] == 1:
                            HubU_alpha_dict[a][n][l][0]['alpha']=alpha
                            if 1 not in HubU_alpha_dict[a][n][l]:
                                HubU_alpha_dict[a][n][l][1] = {}
                            HubU_alpha_dict[a][n][l][1]['alpha']=alpha
                        else:
                            HubU_alpha_dict[a][n][l][s]['alpha']=alpha
                        
                        c.hamiltonian.set_hubbard_u(HubU_dict = HubU_alpha_dict)
                        c.scf.reset()
                        c.calculate()
                        
                        # Make zero
                        if HubU_IO_dict[a][n][l] == 1:
                            HubU_alpha_dict[a][n][l][0]['alpha']=0.
                            if 1 not in HubU_alpha_dict[a][n][l]:
                                HubU_alpha_dict[a][n][l][1] = {}
                            HubU_alpha_dict[a][n][l][1]['alpha']=0.
                        else:
                            HubU_alpha_dict[a][n][l][s]['alpha']=0.
                        
                        #print 'HubU_alpha_dict', HubU_alpha_dict
                             
                        jj = 0
                        for aa in HubU_IO_dict:
                            for nn in HubU_IO_dict[aa]:
                                for ll in HubU_IO_dict[aa][nn]:
                                    if nspin == 2:
                                        if HubU_IO_dict[a][n][l]==2:
                                            for ss in range(2):
                                                Nals_ref = Nanls_ref[aa][nn][ll][ss]
                                                Nals_0 = self.get_Nocc(aa, nn, ll, ss,
                                                                       scale=scale,
                                                                       NbP = NbP,
                                                                       mode='0') 
                                                Nals_KS = self.get_Nocc(aa, nn, ll, ss,
                                                                       scale=scale,
                                                                       NbP = NbP)
                                                
                                                X0[ii,jj] = (Nals_0-Nals_ref)/alpha
                                                Xks[ii,jj] = (Nals_KS-Nals_ref)/alpha
                                                jj+=1
                                            
                                        elif HubU_IO_dict[a][n][l]==1:
                                            Nals_ref = Nanls_ref[aa][nn][ll][0]
                                            Nals_0 = (self.get_Nocc(aa, nn, ll, 0,
                                                                   scale=scale,
                                                                   NbP = NbP,
                                                                   mode='0')
                                                     + self.get_Nocc(aa, nn, ll, 1,
                                                                     scale=scale,
                                                                     NbP = NbP,
                                                                     mode='0')
                                                      )
                                                    
                                            Nals_KS = (self.get_Nocc(aa, nn, ll, 0,
                                                                   scale=scale,
                                                                   NbP = NbP)
                                                     + self.get_Nocc(aa, nn, ll, 1,
                                                                     scale=scale,
                                                                     NbP = NbP)
                                                      )
                                            
                                            X0[ii,jj] = (Nals_0-Nals_ref)/alpha
                                            Xks[ii,jj] = (Nals_KS-Nals_ref)/alpha
                                            jj+=1                                            
                                        else:
                                            print 'PROBLEM!!! \n\n\n'
                                    
                                    else:    
                                        Nals_ref = Nanls_ref[aa][nn][ll][0]
                                        Nals_0 = self.get_Nocc(aa, nn, ll, 0,
                                                               scale=scale,
                                                               NbP = NbP,
                                                               mode='0'
                                                               )   
                                        Nals_KS = self.get_Nocc(aa, nn, ll, 0,
                                                               scale=scale,
                                                               NbP = NbP)
                                        
                                        X0[ii,jj] = (Nals_0-Nals_ref)/alpha
                                        Xks[ii,jj] = (Nals_KS-Nals_ref)/alpha
                                        jj+=1 
                                        
                        if background:
                            X0[ii,-1] -= np.sum(X0[ii,:])
                            Xks[ii,-1] -= np.sum(Xks[ii,:])
                        ii+=1
        if background:
            for jj in range(sites):
                X0[-1,jj]  -= np.sum(X0[:,jj]) 
                Xks[-1,jj] -= np.sum(Xks[:,jj]) 
        
        U = (np.linalg.inv(X0)- np.linalg.inv(Xks))
        
        return U, X0, Xks

    def get_Nocc(self,
                 a, n, l, s,
                 p=0,
                 scale=1,
                 NbP = 1, 
                 mode='KS'
                 ):
        """
        Get occupancy of site
        p: projector
        """ 
        c = self.paw
        N = np.array([0.])  
        if c.density.rank_a[a] == self.world.rank:
            if mode=='KS':
                D_asp = c.density.D_asp
            elif mode=='0':
                D_asp = c.scf.D_asp_0
            else:
                print 'unknown mode:', mode
                assert(0)
            N = np.trace(c.hamiltonian.aoom(unpack2(D_asp[a][s]),
                                        a,n,l,NbP=NbP,scale=scale)[p])
            N = np.array([N])
        self.world.broadcast(N, c.density.rank_a[a])
        return  N[0]

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
        
        D_p = []
        
        Nanls_ref = np.array([0.])
        
        if c.density.rank_a[a] == self.world.rank:
            D_p = c.density.D_asp[a][:].sum(0)  
            Nanls_ref = np.array([np.trace(c.hamiltonian.aoom(unpack2(D_p),
                          a,n,l,NbP=NbP,scale=scale)[0])
                                 ])
        self.world.broadcast(Nanls_ref, c.density.rank_a[a])
        
        
        c.hamiltonian.set_hubbard_u(HubU_dict = HubU_dict)
        c.scf.HubAlphaIO = True
        c.scf.reset()
        c.calculate()
        
        
        if c.density.rank_a[a] == self.world.rank:
            D_p = c.scf.D_asp_0[a][:].sum(0)  
            Nanls_0 = np.array([np.trace(c.hamiltonian.aoom(unpack2(D_p),
                          a,n,l,NbP=NbP,scale=scale)[0])
                                 ])
        self.world.broadcast(Nanls_0, c.density.rank_a[a])
        
        
        if c.density.rank_a[a] == self.world.rank:
            D_p = c.density.D_asp[a][:].sum(0)  
            Nanls_KS = np.array([np.trace(c.hamiltonian.aoom(unpack2(D_p),
                          a,n,l,NbP=NbP,scale=scale)[0])
                                 ])
        self.world.broadcast(Nanls_KS, c.density.rank_a[a])
        
        
        Xres_0 = (Nanls_0[0]-Nanls_ref[0])/alpha
        Xres_KS = (Nanls_KS[0]-Nanls_ref[0])/alpha
        
        U0 = Xres_0**-1 - Xres_KS**-1
        return U0
        #print 'Linear response Hubbard U0 on Sc 3d orbital', U0        @
