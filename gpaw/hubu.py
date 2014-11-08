import numpy as np
import gpaw.mpi as mpi
from gpaw.utilities import unpack2
from copy import copy

class HubU:
    def __init__(self, paw):
        self.paw = paw
        
        # Is this allowed or should world be an input to the class?
        self.world = mpi.world

    def get_MS_Usc(self,
                   HubU_IO_dict, 
                   background = True,
                   alpha = 0.002,
                   scale = 1,
                   NbP = 1,
                   factors = [ 0.5, 0.75, 1.0],
                   out = None,
                   ):
        factors = np.array(factors)
        HubU_dict_U0 = self.get_MS_linear_response_U0(
                                       HubU_IO_dict, 
                                       HubU_dict = {},
                                       background = background,
                                       alpha = alpha,
                                       scale = scale,
                                       NbP = NbP,
                                       )

        HubU_dict_li = {'0.0':HubU_dict_U0}
        
        for factor in (factors):
            HubU_dict_scaled = self.get_scaled_HubU_dict(HubU_dict_U0, factor)
            HubU_dict_out = self.get_MS_linear_response_U0(
                                   HubU_IO_dict, 
                                   HubU_dict = HubU_dict_scaled,
                                   background = background,
                                   alpha = alpha,
                                   scale = scale,
                                   NbP = NbP,
                                   )
            HubU_dict_li[factor]=HubU_dict_out
        
        # Calculate Usc Hub_dict
        HubUsc_dict = {}
        for a in HubU_IO_dict:
            HubUsc_dict[a]={}
            for n in HubU_IO_dict[a]:
                HubUsc_dict[a][n]={}
                for l in HubU_IO_dict[a][n]:
                    Uout_li = []
                    HubUsc_dict[a][n][l]={}
                    for factor in factors:
                        Uout_li.append(HubU_dict_li[factor][a][n][l]['U'])
                    Uout_li = np.array(Uout_li)
                    Usc = np.polyfit(factors, Uout_li, 1)[1]
                    HubUsc_dict[a][n][l]['U']=Usc
                    print 'Usc', Usc
        if out == 'all':
            return HubUsc_dict, HubU_dict_li
        else:
            return HubUsc_dict
    
    def get_scaled_HubU_dict(self, HubU_dict_U0, factor):
        """
        Scale all the U values by a factor. Used in Usc. 
        """
        HubU_dict = {}
        
        for a in HubU_dict_U0:
            HubU_dict[a]={} 
            for n in HubU_dict_U0[a]:
                HubU_dict[a][n]={}
                for l in HubU_dict_U0[a][n]:
                    HubU_dict[a][n][l]={'U':HubU_dict_U0[a][n][l]['U']*factor}
        return HubU_dict

    def get_MS_linear_response_U0(self,
                               HubU_IO_dict, 
                               HubU_dict = {},
                               background = True,
                               alpha = 0.002,
                               scale = 1,
                               NbP = 1,
                               out = None, # options: all
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
        
        c.hamiltonian.set_hubbard_u(HubU_dict = HubU_dict_base)
        
        if c.scf != None:
            c.scf.reset()
        c.calculate()
        
        # Get number of spins
        nspin = self.get_nspin()
        
        # Find number of Hub sites
        Nanls_ref = {}
        sites = 0
        for a in HubU_IO_dict:
            Nanls_ref[a]={} 
            for n in HubU_IO_dict[a]:
                Nanls_ref[a][n]={}
                for l in HubU_IO_dict[a][n]:
                    sites+=1
                    Nanls_ref[a][n][l] = self.get_Nocc(a, n, l, 
                                                scale=scale, NbP = NbP)         
        if sites < 2:
            print 'WARNING: overwriting settings - setting background=False'
            background = 0
        
        X0, Xks =   np.zeros((sites+background,sites+background)), \
                    np.zeros((sites+background,sites+background))
        
        c.scf.HubAlphaIO = True
        HubU_alpha_dict = {}
        ii = 0
        
        for a in HubU_IO_dict:
            if a not in HubU_dict_base:
                HubU_dict_base[a] = {}
            for n in HubU_IO_dict[a]:
                if n not in HubU_dict_base[a]:
                    HubU_dict_base[a][n] = {}
                for l in HubU_IO_dict[a][n]:
                    if l not in HubU_dict_base[a][n]:
                        HubU_dict_base[a][n][l] = {}
                        
                    HubU_alpha_dict = copy(HubU_dict_base)
                    HubU_alpha_dict[a][n][l]['alpha']=alpha
                        
                    c.hamiltonian.set_hubbard_u(HubU_dict = HubU_alpha_dict)
                    c.scf.reset()
                    c.calculate()
                        
                    del HubU_alpha_dict[a][n][l]['alpha']
                         
                    jj = 0
                    for aa in HubU_IO_dict:
                        for nn in HubU_IO_dict[aa]:
                            for ll in HubU_IO_dict[aa][nn]:
                                Nals_ref = Nanls_ref[aa][nn][ll]
                                Nals_0 = self.get_Nocc(aa, nn, ll, 
                                                       scale=scale,
                                                       NbP = NbP,
                                                       mode='0'
                                                       )   
                                Nals_KS = self.get_Nocc(aa, nn, ll,
                                                       scale=scale,
                                                       NbP = NbP)
                                
                                X0[ii,jj] = (Nals_0-Nals_ref)/alpha
                                Xks[ii,jj] = (Nals_KS-Nals_ref)/alpha
                                jj+=1 
                                
                        if background:
                            X0[ii,-1] -= np.sum(X0[ii,:])
                            Xks[ii,-1] -= np.sum(Xks[ii,:])

        if background:
            for jj in range(sites):
                X0[-1,jj]  -= np.sum(X0[:,jj]) 
                Xks[-1,jj] -= np.sum(Xks[:,jj]) 
        U = (np.linalg.inv(X0)- np.linalg.inv(Xks))
        
        # Create Hub_dict
        new_HubU_dict = {}
        jj=0
        for a in HubU_IO_dict:
            new_HubU_dict[a]={} 
            for n in HubU_IO_dict[a]:
                new_HubU_dict[a][n]={}
                for l in HubU_IO_dict[a][n]:
                    if HubU_IO_dict[a][n][l]==0:
                        pass
                    else:
                        new_HubU_dict[a][n][l]={'U':U[jj,jj]}
                        jj+=1
        
        if out == 'all':
            return new_HubU_dict, U, X0, Xks 
        else:
            return new_HubU_dict


    def get_LR_U0_range(self,a,n,l,
                               HubU_dict = {},
                               alpha = 0.002,
                               steps = 5, 
                               negative = 1,
                               scale = 1,
                               NbP = 1,
                               mode='all',
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
        HubU_dict[a][n][l]['alpha']=alpha

        c = self.paw
        c.calculate()
        #print 'magmom', c.density.magmom_av
        
        #print 'c.occupations.magmom', c.occupations.magmom
        #print 'c.get_magnetic_moments()', c.get_magnetic_moments()
        #STOP
        
        
        mag = c.occupations.magmom
        Nanl_ref = self.get_Nocc(a, n, l, scale=scale, NbP = NbP)
        print Nanl_ref
         
        #STOP
        if negative:
            alpha_range = np.linspace(-1*alpha, alpha, steps)
        else:
            alpha_range = np.linspace(0., alpha, steps)
            
        Nanl_0_li = np.array([])
        Nanl_KS_li = np.array([])
        
        mag_li = c.get_magnetic_moments()
        
        for i, alphai in enumerate(alpha_range):
            c.scf.HubAlphaIO = False
            HubU_dict[a][n][l]['alpha']=0.
            c.hamiltonian.set_hubbard_u(HubU_dict = HubU_dict)
            c.scf.reset()
            c.calculate()
            
            HubU_dict[a][n][l]['alpha']=alphai
            c.hamiltonian.set_hubbard_u(HubU_dict = HubU_dict)
            c.scf.HubAlphaIO = True
            c.scf.reset()
            c.calculate()
            
            mag_li = np.vstack((mag_li, c.get_magnetic_moments()))
            
            Nanl_0 = self.get_Nocc(a, n, l, scale=scale, NbP = NbP, mode='0')
            Nanl_KS = self.get_Nocc(a, n, l, scale=scale, NbP = NbP)
        
            Nanl_0_li = np.hstack((Nanl_0_li,Nanl_0))
            Nanl_KS_li = np.hstack((Nanl_KS_li,Nanl_KS))
        
        Xres_0 = np.polyfit(alpha_range, Nanl_0_li-Nanl_ref, 1)[0]
        Xres_KS = np.polyfit(alpha_range, Nanl_KS_li-Nanl_ref, 1)[0]
        
        
        #print 'Nanl_ref, Nanl_0, Nanl_KS', Nanl_ref, Nanl_0, Nanl_KS
        #Xres_0 = (Nanl_0-Nanl_ref)/alpha
        #Xres_KS = (Nanl_KS-Nanl_ref)/alpha
        
        U0 = Xres_0**-1 - Xres_KS**-1
        if mode=='all':
            return U0, Nanl_0_li, Nanl_KS_li, Nanl_ref, mag_li, alpha_range
        else:
            return U0
    
    def get_linear_response_U0(self,a,n,l,
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
        HubU_dict[a][n][l]['alpha']=alpha

        c = self.paw
        c.calculate()
        
        Nanl_ref = self.get_Nocc(a, n, l, scale=scale, NbP = NbP)
            
        c.hamiltonian.set_hubbard_u(HubU_dict = HubU_dict)
        c.scf.HubAlphaIO = True
        c.scf.reset()
        c.calculate()
        
        Nanl_0 = self.get_Nocc(a, n, l, scale=scale, NbP = NbP, mode='0')
        Nanl_KS = self.get_Nocc(a, n, l, scale=scale, NbP = NbP)
        print 'Nanl_ref, Nanl_0, Nanl_KS', Nanl_ref, Nanl_0, Nanl_KS
        Xres_0 = (Nanl_0-Nanl_ref)/alpha
        Xres_KS = (Nanl_KS-Nanl_ref)/alpha
        
        U0 = Xres_0**-1 - Xres_KS**-1
        return U0

    def get_nspin(self):
        c = self.paw
        nspin=np.array([0])
        if c.density.rank_a[0] == self.world.rank:
            nspin = len(c.density.D_asp[0])
            nspin=np.array([nspin])
        self.world.broadcast(nspin, c.density.rank_a[0])
        nspin=nspin[0]
        return nspin

    def get_Nocc(self,
                 a, n, l,
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
        nspin = self.get_nspin()
        N = np.array([0.])  
        if c.density.rank_a[a] == self.world.rank:
            if mode=='KS':
                D_asp = c.density.D_asp
            elif mode=='0':
                D_asp = c.scf.D_asp_0
            else:
                print 'unknown mode:', mode
                assert(0)
            
            N = 0.
            for s in range(nspin):
                N += np.trace(c.hamiltonian.aoom(unpack2(D_asp[a][s]),
                                a,n,l,NbP=NbP,scale=scale)[p])
            N = np.array([N])
        self.world.broadcast(N, c.density.rank_a[a])
        return  N[0]
