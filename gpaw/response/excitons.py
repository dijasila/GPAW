import numpy as np
from fractions import gcd
import pickle
import sys

from ase.units import Bohr,Hartree
from ase.dft.kpoints import monkhorst_pack

from gpaw import GPAW
from gpaw.response.cell import get_primitive_cell,set_Gvectors
from gpaw.kpt_descriptor import to1bz

from scipy.special import jn
from scipy.special import yn
from scipy.special import struve
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import eig


class Exciton:
    """This class defines the Excitons properties"""

    def __init__(self,
                 calc=None,
                 q_list = None,   #Needed only when generating response file
                 chi_name=None,
                 response_name=None,
                 file_name='exciton',              
                 eff_mass=None,
                 d_layer=None,
                 gap = None,
                 npwz = 500,
                 K_exc_c = None,
                 nv=None,
                 nc=None,
                 ecut=50./Hartree,
                 qtoskip=[],
                 save_pckl=True): 

        print('Initializing Exciton Object')
        self.file_name = file_name
        self.eff_mass = eff_mass
        self.d_layer = d_layer
        self.gap = gap
        self.npwz = npwz
        self.K_exc = np.array(K_exc_c)
        self.nv = nv
        self.nc = nc
        self.qtoskip = qtoskip
        self.save_pckl = save_pckl

        if calc is not None:
            self.calc = GPAW(calc) 
            print ''
            print '####################################################'    
            print('Getting grid and kpoints descriptor')  
            self.gd = self.calc.wfs.gd.new_descriptor()
            self.kd = self.calc.wfs.kd
            print('Getting reducible BZ')  
            self.bzq_qc = self.kd.get_bz_q_points()
            #(self.ibzq_qc, self.ibzq_q, self.iop_q,self.timerev_q, self.diff_qc) = self.kd.get_ibz_q_points(self.bzq_qc, self.kd.symmetry.op_scc)
            
            rpad = np.ones(3, int)                                                                            
            self.acell_cv = self.gd.cell_cv
            self.acell_cv, self.bcell_cv, self.vol, self.BZvol = get_primitive_cell(self.acell_cv,rpad=rpad)
           
            if response_name is not None:
                if response_name is 'NotNeeded':
                    pass
                else:
                    print('Getting the response file')
                    self.qpts_cv,self.npw_q,self.Gvec_qGcv,self.epsinv3D_qwGG,self.chi3D_qwGG = pickle.load(open(response_name))
            else:
                print('Getting the response file')
                self.qpts_cv,self.npw_q,self.Gvec_qGcv,self.epsinv3D_qwGG,self.chi3D_qwGG = self.get_response_file(chi_name=chi_name, q_list=q_list)
            

            print('Done with the Initialization')

#--------------------------------------------------------------------------------------------------------------

    def get_response_file(self,chi_name,q_list):
        nq = len(q_list)
        nw = 1
        
        npw_q = [] 
        Gvec_qGcv = []         
        chi3D_qwGG = []
        epsinv3D_qwGG = []
        q_cv_list = []

        for iq in range(nq):
            if q_list[iq] in self.qtoskip:
                continue
            if q_list[iq]==0:
                q_cv_list.append([0.,0.,0.])
                pd, chi_wGG, eps_wGG = pickle.load(open(chi_name+'-q-%s.pckl'%q_list[1]))
                npw_q.append(chi_wGG.shape[1])
                Gvec_qGcv.append(pd.get_reciprocal_vectors(add_q=False))
                temp = np.zeros([nw,npw_q[0],npw_q[0]])
                temp[0,:,:] = np.eye(npw_q[0])
                eps3D_qwGG.append(temp) 
            else:
                pd, chi_wGG, eps_wGG = pickle.load(open(chi_name+'-q-%s.pckl'%q_list[iq]))
                npw_q.append(chi_wGG.shape[1])
                Gvec_qGcv.append(pd.get_reciprocal_vectors(add_q=False))
                if q_list[iq]<1:
                    q_cv_list.append(np.array([0.,q_list[iq],0.]))
                    #q_c_list.append(np.dot(np.array([q_list[iq],0.,0.]),np.linalg.inv(self.bcell_cv)))
                else:
                    q_cv_list.append(pd.K_qv)
               
                chi3D_qwGG.append(chi_wGG)

                epsinv_temp = np.zeros_like(eps_wGG,dtype='complex')
                epsinv_temp[0] = np.linalg.inv(eps_wGG[0])
                epsinv3D_qwGG.append(epsinv_temp)

        qpts_cv = np.array(q_cv_list)
        less_mem = 0.               
        pickle.dump((qpts_cv,npw_q,Gvec_qGcv,epsinv3D_qwGG,less_mem), open(self.file_name+'_response.pckl', 'wb'), pickle.HIGHEST_PROTOCOL)              
        
        return qpts_cv,npw_q,Gvec_qGcv,epsinv3D_qwGG,chi3D_qwGG

#--------------------------------------------------------------------------------------------------------------

    def get_localized_step(self,width=None):
        print('Getting step function around the layer')  
        Nint = self.npwz
        if width is None:
            width = self.d_layer
        else:
            width = width
        L = self.acell_cv[2,2]
        z_grid = np.linspace(0.,L,Nint+1)
        slab_pos = L/2.

        nv_z = np.zeros(Nint+1)
        nc_z = np.zeros(Nint+1)
        for iz in range(Nint+1):
            if z_grid[iz]>slab_pos-width/2. and z_grid[iz]<slab_pos+width/2.:
                nv_z[iz] = 1./width
                nc_z[iz] = 1./width
            else:
                nv_z[iz] = 0.
                nc_z[iz] = 0.

        return z_grid,nv_z,nc_z
        
#--------------------------------------------------------------------------------------------------------------        

    def get_z_charge_distribution(self,kpoint=None):
        Nint = self.npwz
        L = self.acell_cv[2,2]
        z_grid = np.linspace(0.,L,Nint+1)
        slab_pos = self.acell_cv[2,2]/2.
       
        # Finding K point to plot
        #-----------------------------------------------------
        if kpoint is None:
            K_exc=self.K_exc
        else:
            K_exc=kpoint   

        print('Getting the out-of-plane wavefunction at k=',K_exc)      
        nkpt = self.kd.where_is_q(K_exc, self.bzq_qc) 
        nkpt = self.kd.bz2ibz_k[nkpt]

        # Getting the wave functions squared
        #-----------------------------------------------------
        wf_v = self.calc.get_pseudo_wave_function(band=self.nv[0],kpt=nkpt)
        wf_c = self.calc.get_pseudo_wave_function(band=self.nc[0],kpt=nkpt)

        abs_wf_v_2 = np.abs(wf_v)**2
        abs_wf_c_2 = np.abs(wf_c)**2

        abs_wf_v_2_z = np.zeros(wf_v.shape[2])
        abs_wf_c_2_z = np.zeros(wf_c.shape[2])
        for iz in range(0,wf_v.shape[2]):
            abs_wf_v_2_z[iz] = np.sum(abs_wf_v_2[:,:,iz])
            abs_wf_c_2_z[iz] = np.sum(abs_wf_c_2[:,:,iz])

        # Normalization
        #-------------------------------
        gd = self.gd.get_grid_point_coordinates()       

        abs_wf_v_2_z *= (gd[0,1,0,0]-gd[0,0,0,0])*(gd[1,0,1,0]-gd[1,0,0,0])
        abs_wf_c_2_z *= (gd[0,1,0,0]-gd[0,0,0,0])*(gd[1,0,1,0]-gd[1,0,0,0])
        norm_v = (gd[2,0,0,1]-gd[2,0,0,0]) * np.sum(abs_wf_v_2_z)
        norm_c = (gd[2,0,0,1]-gd[2,0,0,0]) * np.sum(abs_wf_c_2_z)

        # Potential and Densities Initialization
        #----------------------------------------
        z_abinitio = gd[2,0,0,:]
        zv_fit = InterpolatedUnivariateSpline(z_abinitio,abs_wf_v_2_z/norm_v)
        zc_fit = InterpolatedUnivariateSpline(z_abinitio, abs_wf_c_2_z/norm_c)

        nv_z = np.zeros(Nint+1)
        nc_z = np.zeros(Nint+1)

        nv_z = zv_fit(z_grid)
        nv_z /= np.sum(nv_z)*(z_grid[1]-z_grid[0]) 
        nc_z = zc_fit(z_grid)
        nc_z /= np.sum(nc_z)*(z_grid[1]-z_grid[0]) 


        if self.save_pckl is True:
            pickle.dump((z_grid,nv_z,nc_z),open(self.file_name+'_z_densities.pckl','w'))
        return z_grid,nv_z,nc_z

#--------------------------------------------------------------------------------------------------------------   

    def get_epsilon_Q2D(self):  
        #print('Getting the Q2D dielectric Function') 
        qpts_cv = self.qpts_cv
        d_Q2D = self.d_layer
        L = self.acell_cv[2,2]
        rec_cell = self.bcell_cv

        z0 = L/2.

        qpts_list = []
        epsM_Q2D_list = []
        for iq in range(0,qpts_cv.shape[0]):
            if iq in self.qtoskip:
                continue    
            q_abs = np.linalg.norm(qpts_cv[iq])
            qpts_list.append(q_abs)
            eps3D_inv_GG = self.epsinv3D_qwGG[iq][0]
            
            epsM_Q2D_inv_temp=eps3D_inv_GG[0,0]
            Gvec_Gcv = self.Gvec_qGcv[iq]   
            for iG in range(0,Gvec_Gcv.shape[0]):
                if Gvec_Gcv[iG,0] == 0 and Gvec_Gcv[iG,1] == 0 and iG!=0:
                    G_perp = Gvec_Gcv[iG,2]
                    epsM_Q2D_inv_temp += (2./d_Q2D)*(np.exp(1j*G_perp*z0))*np.sin(G_perp*d_Q2D/2.)\
                                          *eps3D_inv_GG[iG,0]/G_perp

            epsM_Q2D_list.append((1./(epsM_Q2D_inv_temp)).real)


        q_abs_array = np.array(qpts_list)
        epsM_Q2D_array = np.array(epsM_Q2D_list)

        if self.save_pckl is True:
            pickle.dump((q_abs_array, epsM_Q2D_array),open(self.file_name+'_eps_q.pckl','w'))

        return q_abs_array, epsM_Q2D_array

#--------------------------------------------------------------------------------------------------------------   

    def get_epsilon_Q2D_slope(self,Wq_name=None,slope=None):
        #print('Getting the slope of the Q2D dielectric Function') 
        if slope is not None:
            q = []
            slope_Q2D = slope
        else:
            q, eps_q = self.get_epsilon_Q2D()

            nps = 4      
            #f_eps = InterpolatedUnivariateSpline(q,eps_q)
            #q = np.linspace(0,q[-1],5000)
            #eps_q = f_eps(q)

            def func(x, a):
                return a * x + 1

            slope_Q2D = curve_fit(func,q[0:nps],eps_q[0:nps])[0]
        #print '**********'
        #print 'Slope:', slope_Q2D
        #print '**********'
        return q,slope_Q2D

#--------------------------------------------------------------------------------------------------------------   

    def poisson_solver(self,q_abs,nv_z,BC_v0,BC_vN,Delta,Nint):
        print('Solving the Poisson Equation') 
        M = np.zeros((Nint+1,Nint+1))
        f_z = np.zeros(Nint+1)
        f_z[:] = nv_z[:]
        # Finite Difference Matrix        
        for i in range(1,Nint):
            M[i,i] = -2./(Delta**2)-q_abs**2
            M[i,i+1] = 1./Delta**2
            M[i,i-1] = 1./Delta**2
        M[0,0] = -1.
        M[Nint,Nint] = -1.
        
        f_z[0] = BC_v0
        f_z[Nint] = BC_vN

        # Getting the Potential
        M_inv = np.linalg.inv(M)
        V_z = -4.*np.pi*q_abs**2 * np.dot(M_inv,f_z)
        return V_z

#--------------------------------------------------------------------------------------------------------------   

    def get_screened_pot_q(self,mode='Q2D',Wq_name=None,slope=None):
        #print('Getting the potential in q space') 
        if Wq_name is not None and mode is not '2D':
            q_abs,W_q = pickle.load(open(Wq_name))
        else: 
            if mode == 'z-dependent' or mode == 'real-steps':       
                cell = self.acell_cv
                rec_cell = self.bcell_cv
                q_cv = self.qpts_cv
                
                
                if mode == 'z-dependent':
                    z_grid,nv_z,nc_z = self.get_z_charge_distribution()
                elif mode == 'real-steps':
                    z_grid,nv_z,nc_z = self.get_localized_step()
                Delta = z_grid[1]-z_grid[0]
                

                qpts_list = []
                W_list = []
                for iq in range(0,len(q_cv)):
                    if iq in self.qtoskip:                # Remove at some point
                        continue
                    npw = self.npw_q[iq]
                    Gvec_Gcv = self.Gvec_qGcv[iq]
                    Glist = []   
                    for iG in range(npw):       # List of G with Gx,Gy = 0
                        if Gvec_Gcv[iG, 0] == 0 and Gvec_Gcv[iG, 1] == 0:
                            Glist.append(iG)

                    q_abs = np.linalg.norm(q_cv[iq])
                    qpts_list.append(q_abs)
                    BC_V0 = 1./2./q_abs*np.exp(-q_abs*cell[2,2]/2.)
                    BC_VL = 1./2./q_abs*np.exp(-q_abs*cell[2,2]/2.)                    

                    V_z = self.poisson_solver(q_abs,nv_z,BC_V0,BC_VL,Delta,len(z_grid)-1)
                    #V_z = np.zeros(len(z_grid))
                    #V_z[:] = 4.*np.pi*(1.-np.exp(-q_abs*self.d_layer/2.))/self.d_layer 
                    
                    epsinv3D_GG = self.epsinv3D_qwGG[iq][0].take(Glist,0).take(Glist,1)
                    #epsinv3D_GG = np.identity(len(Glist))
                 
                    first_int_G = np.zeros(len(Glist), dtype=complex)
                    second_int_G = np.zeros(len(Glist), dtype=complex)
                   
                    for iG in range(len(Glist)):
                        Gz = Gvec_Gcv[Glist[iG],2]
                        first_int_G[iG] = np.sum(np.exp(1.j*Gz*z_grid)*nc_z)*Delta
                        second_int_G[iG] = np.sum(np.exp(-1.j*Gz*z_grid)*V_z)*Delta

                    W_q = -1./q_abs**2/cell[2,2]*np.dot(first_int_G,np.dot(epsinv3D_GG,second_int_G))                   
                    W_list.append(W_q.real)

                q_abs = np.array(qpts_list)
                W_q = np.array(W_list)

            if mode == 'Q2D':
                d_slab = self.d_layer
                q_abs, eps_q = self.get_epsilon_Q2D()
                W_q = -4.*np.pi/d_slab/q_abs**2*(1.-2.*np.exp(-q_abs*d_slab/2.)*np.sinh(q_abs*d_slab/2.)/q_abs/d_slab)*eps_q**(-1)

            if mode == '2D':
                q_abs,alpha = self.get_epsilon_Q2D_slope(Wq_name=Wq_name,slope=slope)
                if Wq_name is not None:
                    q_abs,W_temp_q = pickle.load(open(Wq_name))
                W_q = -2.*np.pi/q_abs/(1+alpha*q_abs)

        if self.save_pckl is True:
            pickle.dump((q_abs,W_q),open(self.file_name+'_W_'+mode+'_q.pckl','w'))
        return q_abs, W_q

#--------------------------------------------------------------------------------------------------------------   

    def get_screened_pot_r(self,r_array,mode='Q2D',Wq_name=None,d_par=None,slope=None):
        #print('Getting the potential in r space') 
        if mode=='Coulomb':
            W_r=-1./r_array
        elif mode=='2D':
            xxx,alpha = self.get_epsilon_Q2D_slope(Wq_name=Wq_name,slope=slope)
            polar = alpha/2./np.pi
            W_r=1/4./polar*(yn(0,r_array/(2.*np.pi*polar))-struve(0,r_array/(2.*np.pi*polar)))
        else:
            if d_par is None:
                d_par = self.d_layer
            q_temp,W_q = self.get_screened_pot_q(mode=mode,Wq_name=Wq_name)

            import matplotlib.pyplot as plt
            plt.plot(q_temp,W_q*q_temp,'ob')

            W_q *= q_temp
            q = np.linspace(q_temp[0],q_temp[-1],10000)
            #fW_interp = InterpolatedUnivariateSpline(q_temp,W_q)
            #Wt_q = fW_interp(q)
            Wt_q = np.interp(q,q_temp,W_q)
            Dq_Q2D = q[1]-q[0]
            Coulombt_q = -4.*np.pi/q*(1.-np.exp(-q*d_par/2.))/d_par        #### CHANGE THIS BACK TO Q2D!!!####

            plt.plot(q,Wt_q,'--r')
            plt.show()

            W_r = np.zeros(len(r_array))
            for ir in range(0,len(r_array)):
                J_q = jn(0, q*r_array[ir])
                if r_array[ir]>np.exp(-13):
                    Int_temp = -1./d_par*np.log((d_par/2. + np.sqrt(r_array[ir]**2 + d_par**2/4.))\
                                /(-d_par/2. + np.sqrt(r_array[ir]**2 + d_par**2/4.)))
                else:
                    Int_temp = -1./d_par*np.log(d_par**2/r_array[ir]**2)
                W_r[ir] =  Dq_Q2D/2./np.pi * np.sum(J_q*(Wt_q-Coulombt_q)) + Int_temp 

        if self.save_pckl is True:
            pickle.dump((r_array,W_r),open(self.file_name+'_W_'+mode+'_r.pckl','w'))

        return r_array,W_r

#--------------------------------------------------------------------------------------------------------------   

    def get_binding_energies(self,L_min=-50,L_max=10,Delta=0.1,mode='Q2D',Wq_name=None,slope=None):
        #print('Calculating binding energies') 
        r_space = np.arange(L_min,L_max,Delta)
        Nint = len(r_space) 

        r,W_r = self.get_screened_pot_r(r_array=np.exp(r_space),mode=mode,Wq_name=Wq_name,slope=slope)
        #import matplotlib.pyplot as plt
        #plt.plot(r,W_r)
        #plt.plot(r,-1./r,'or')
        #plt.ylim(-1.5,0.1)
        #plt.show()
        H = np.zeros((Nint,Nint),dtype=complex)
        for i in range(0,Nint):
            r_abs = np.exp(r_space[i])
            H[i,i] = - 1./r_abs**2/2./self.eff_mass*(-2./Delta**2 + 1./4.) + W_r[i]
            if i+1 < Nint:
                H[i,i+1] = -1./r_abs**2/2./self.eff_mass*(1./Delta**2-1./2./Delta)
            if i-1 >= 0:
                H[i,i-1] = -1./r_abs**2/2./self.eff_mass*(1./Delta**2+1./2./Delta)

        ee, ev = eig(H)
        index_sort = np.argsort(ee.real)
        ee = ee[index_sort]
        ev = ev[:,index_sort]
        
        if self.save_pckl is True:
            pickle.dump((ee, ev), open(self.file_name+'_eigenproblem.pckl', 'w'))
        
        return ee, ev

#--------------------------------------------------------------------------------------------------------------   
    
    def model_BSE(self,mode_bands='real',hl_mass=None,el_mass=None,mode='Q2D',Wq_name=None,\
                  integrateGamma=False,overlap=False,return_more=False):
        print('Model BSE Calculation') 
        print('Getting grid quantities') 
        q_c = np.array([1e-09, 0., 0.])
        ftol=1e-5
        eff_mass = self.eff_mass
        calc = self.calc
        ef = calc.get_fermi_level()/Hartree
        gap = self.gap
        acell_cv = self.acell_cv
        bcell_cv = self.bcell_cv
        vol = self.vol
        kd = self.kd
        ibzq_qc = kd.ibzk_kc
        kq_k = kd.find_k_plus_q(q_c)
        kp_1bz = to1bz(kd.bzk_kc,acell_cv)
        nspins = calc.wfs.nspins

        # Single particle transitions
        #-------------------------------------
        print('Collecting single particle eigenvalues') 
        e_skn = {}
        f_skn = {}
        e_S = {}
        focc_s = {}
        Sindex_S3 = {}
        e_S_list = []

  
        for ispin in range(nspins):
            e_skn[ispin] = np.array([calc.get_eigenvalues(kpt=k, spin=ispin)
                                          for k in range(kd.nibzkpts)]) / Hartree
            f_skn[ispin] = np.array([calc.get_occupation_numbers(kpt=k, spin=ispin)
                                          / kd.weight_k[k]
                                          for k in range(kd.nibzkpts)]) 

        iS = 0
        for k1 in range(kd.nbzkpts):
            ibzkpt1 = kd.bz2ibz_k[k1]
            ibzkpt2 = kd.bz2ibz_k[kq_k[k1]]
            for n1 in range(self.nv[0], self.nv[1]): 
                for m1 in range(self.nc[0], self.nc[1]): 
                    focc = f_skn[0][ibzkpt1,n1] - f_skn[0][ibzkpt2,m1]
                    check_ftol = focc > ftol
                    if check_ftol:     
                        if mode_bands=='parabolic':        
                            e_skn[0][ibzkpt2,m1] = ef+gap/2.+np.linalg.norm(np.dot(ibzq_qc[ibzkpt2],bcell_cv))**2/2./el_mass
                            e_skn[0][ibzkpt1,n1] = ef-gap/2.-np.linalg.norm(np.dot(ibzq_qc[ibzkpt1],bcell_cv))**2/2./hl_mass
                            e_S[iS] = e_skn[0][ibzkpt2,m1] - e_skn[0][ibzkpt1,n1]   
                        elif mode_bands=='real':
                            e_S[iS] = e_skn[0][ibzkpt2,m1] - e_skn[0][ibzkpt1,n1]     
                        focc_s[iS] = focc
                        Sindex_S3[iS] = (k1, n1, m1)
                        iS += 1     



        nS = iS
        focc_S = np.zeros(nS)
        for iS in range(nS):
            focc_S[iS] = focc_s[iS]


        # Calculating the Kernel
        #------------------------------------
        print('Building the interaction kernel') 
        K_SS = np.zeros((nS, nS), dtype=complex)
       
        q_W_abs,W_ex_q = self.get_screened_pot_q(mode=mode, Wq_name=Wq_name)
        W_ex_q *= q_W_abs
        f_Q2D = InterpolatedUnivariateSpline(q_W_abs,W_ex_q)

        overlap_SS = self.get_wfs_overlap(overlap,nS,Sindex_S3)
        pot_gamma = self.integrate_around_gamma(integrateGamma,bcell_cv,300)       
        for iS in range(nS):
            k1, n1, m1 = Sindex_S3[iS]
            for jS in range(nS):
                k2, n2, m2 = Sindex_S3[jS]
        
                q_Gkk = kp_1bz[k2] - kp_1bz[k1]            
                q_diff = np.linalg.norm(np.dot(q_Gkk,bcell_cv))
                if q_diff>1e-06:
                    tmp_K = -overlap_SS[iS,jS]*np.interp(q_diff,q_W_abs,W_ex_q)/q_diff #f_Q2D(q_diff)/q_diff #*overlap
                else:
                    tmp_K = pot_gamma
                K_SS[iS, jS] =  tmp_K

        K_SS *= acell_cv[2,2]/vol/kd.nbzkpts  

        # Get the Hamiltonian
        #------------------------------------
        H_sS = np.zeros_like(K_SS)


        for iS in range(nS):
            H_sS[iS,iS] = e_S[iS]
            for jS in range(nS):
                H_sS[iS,jS] += -0.5 * focc_S[iS] * K_SS[iS,jS]

        # Force matrix to be Hermitian
        H_Ss = H_sS
        H_sS = (np.real(H_sS) + np.real(H_Ss.T)) / 2. + 1j * (np.imag(H_sS) - np.imag(H_Ss.T)) /2.
        print('Diagonalizing the two particle Hamiltonian') 
        w_S, A_SS = eig(H_sS)
        if return_more is not False:
            if self.save_pckl is False:
                pickle.dump((w_S, A_SS, Sindex_S3, e_skn, H_sS), open(self.file_name+'_bse_model_full_matrix.pckl', 'w'))            
            return w_S, A_SS, Sindex_S3, e_skn, H_sS
        else:
            if self.save_pckl is True:
                pickle.dump((w_S, A_SS), open(self.file_name+'_bse_model.pckl', 'w')) 
            return w_S, A_SS

#--------------------------------------------------------------------------------------------------------------        

    def integrate_around_gamma(self,integrateGamma,bcell_cv,Ngrid):     
        if integrateGamma is False:
            Int = 0.
        else:
            bcellt_cv = bcell_cv/self.kd.N_c
            q_grid_c = monkhorst_pack((Ngrid, Ngrid, 1))
            q_grid_v = np.dot(q_grid_c, bcellt_cv)
            v_grid_q = np.zeros(len(q_grid_v))
            for iq in range(0,len(q_grid_v)):
                v_grid_q[iq] = 2.*np.pi/np.linalg.norm(q_grid_v[iq])
            bcell_Area = np.abs(np.linalg.det(bcellt_cv))
            dA = bcell_Area / len(q_grid_v)
            Int = np.sum(v_grid_q)*dA/bcell_Area
        return Int

#--------------------------------------------------------------------------------------------------------------

    def get_wfs_overlap(self,overlap,nS,Sindex_S3):
        Nint = 100
        L = self.acell_cv[2,2]
        z_grid = np.linspace(0.,L,Nint)

        gd = self.gd.get_grid_point_coordinates()
        dxdy = (gd[0,1,0,0]-gd[0,0,0,0])*(gd[1,0,1,0]-gd[1,0,0,0])
        dz = (gd[2,0,0,1]-gd[2,0,0,0])
        z_abinitio = gd[2,0,0,:]

        overlap_SS = np.zeros((nS, nS), dtype=complex)
        if overlap == True:
            wfs_vSz = np.zeros((nS,len(z_abinitio)),dtype=complex)
            wfs_cSz = np.zeros((nS,len(z_abinitio)),dtype=complex)
            for iS in range(nS):
                k1, n1, m1 = Sindex_S3[iS]      
                # Getting the wave functions squared
                #-----------------------------------------------------
                nkpt = self.kd.bz2ibz_k[k1]          
                wf_v = self.calc.get_pseudo_wave_function(band=n1,kpt=nkpt)
                wf_c = self.calc.get_pseudo_wave_function(band=m1,kpt=nkpt)

                wf_v = self.kd.transform_wave_function(psit_G=wf_v, k=k1)
                wf_c = self.kd.transform_wave_function(psit_G=wf_c, k=k1)             

                wf_vz = np.zeros(wf_v.shape[2],dtype=complex)
                wf_cz = np.zeros(wf_c.shape[2],dtype=complex)
                for iz in range(0,wf_v.shape[2]):
                    wf_vz[iz] = np.sum(wf_v[:,:,iz])*dxdy
                    wf_cz[iz] = np.sum(wf_c[:,:,iz])*dxdy

                wf_vz /= np.sqrt(np.sum(np.abs(wf_vz)**2)*dz)
                wf_cz /= np.sqrt(np.sum(np.abs(wf_cz)**2)*dz)  

                #wf_vz = np.abs(wf_vz)/np.sqrt(np.sum(np.abs(wf_vz)**2)*dz)
                #wf_cz = np.abs(wf_cz)/np.sqrt(np.sum(np.abs(wf_cz)**2)*dz)  

                                
                '''
                # Fitting
                #-----------------------------------------------------
                zv_fit = InterpolatedUnivariateSpline(z_abinitio,wf_vz)
                zc_fit = InterpolatedUnivariateSpline(z_abinitio,wf_cz)

                wf_vz = zv_fit(z_grid)
                wf_vz /= np.sum(wf_vz)*(z_grid[1]-z_grid[0]) 
                wf_cz = zc_fit(z_grid)
                wf_cz /= np.sum(wf_cz)*(z_grid[1]-z_grid[0])  
                '''
               
                wfs_vSz[iS] = wf_vz   
                wfs_cSz[iS] = wf_cz   

    
            overlap_SS = np.dot(wfs_vSz,wfs_vSz.T.conj())*np.dot(wfs_cSz,wfs_cSz.T.conj())*dz**2
            #for i in range(iS-1):
            #    print overlap_SS[i,i+1]
        else:
            overlap_SS = np.ones((nS, nS))
        return overlap_SS

#--------------------------------------------------------------------------------------------------------------   

    def find_k_from_kpoints(self, kpoints,plot_BZ=True):
        print('Finding kpoint already in the grid') 
        acell_cv = self.acell_cv
        bcell_cv = self.bcell_cv
        gd = self.gd
        kd = self.kd
        
        if plot_BZ is True:
            #--------------------------------------------
            # Plotting the points in the Brillouin Zone
            #--------------------------------------------
            kp_1bz = to1bz(kd.bzk_kc,acell_cv)
            
            bzk_kcv=np.dot(kd.bzk_kc,bcell_cv)
            kp_1bz_v=np.dot(kp_1bz,bcell_cv)
            
            import matplotlib.pyplot as plt
            plt.plot(bzk_kcv[:,0],bzk_kcv[:,1],'xg')
            plt.plot(kp_1bz_v[:,0],kp_1bz_v[:,1],'ob')
            for ik in range(1,len(kpoints)):    
                kpoint1_v = np.dot(kpoints[ik],bcell_cv)
                kpoint2_v = np.dot(kpoints[ik-1],bcell_cv)
                plt.plot([kpoint1_v[0],kpoint2_v[0]],[kpoint1_v[1],kpoint2_v[1]],'--vr')


        #--------------------------------------------
        # Finding the points along given directions
        #--------------------------------------------      
        op_scc = kd.symmetry.op_scc
        N_c = kd.N_c
        wpts_xc = kpoints
        
        x_x = []
        k_xc = []
        k_x = []
        x = 0.
        X = [0]
        for nwpt in range(1, len(wpts_xc)):
            X.append(x)
            to_c = wpts_xc[nwpt]
            from_c = wpts_xc[nwpt - 1]
            vec_c = to_c - from_c
            print 'From ',from_c,' to ',to_c
            Nv_c = (vec_c * N_c).round().astype(int)
            Nv = abs(gcd(gcd(Nv_c[0], Nv_c[1]), Nv_c[2]))
            print Nv,' points found'
            dv_c = vec_c / Nv
            dv_v = np.dot(dv_c, bcell_cv)
            dx = np.linalg.norm(dv_v)
            if nwpt == len(wpts_xc) - 1:
                X.append(Nv * dx)
                Nv += 1
            for n in range(Nv):
                k_c = from_c + n * dv_c
                bzk_c = to1bz(np.array([k_c]), acell_cv)[0]
                ikpt = kd.where_is_q(bzk_c,kd.bzk_kc)
                x_x.append(x)
                k_xc.append(k_c)
                k_x.append(ikpt)
                x += dx
        if plot_BZ is True:      
            for ik in range(len(k_xc)):
                ktemp_xcv = np.dot(k_xc[ik],bcell_cv)
                plt.plot(ktemp_xcv[0],ktemp_xcv[1],'xr',markersize=10)
            plt.show()

        return x_x,k_xc,k_x,X      

#--------------------------------------------------------------------------------------------------------------   

    def find_Sindex_S3(self,k, n, m, Sindex_S3):
        nS = len(Sindex_S3)
        index = None
        for iS in range(0, nS):
            if Sindex_S3[iS]==(k, n, m):
                index = iS
        if index is None:
            raise('S index not found!')
        return index

#--------------------------------------------------------------------------------------------------------------     
    
    def get_excitation_weigths(self, bands, kpoints, filename_SS=None,mode_bands='real',hl_mass=None,el_mass=None,mode='Q2D',Wq_name=None,integrateGamma=None,overlap=True):
        """ It gives the weigths of the exciton for given bands and kpoint-directions lists """
        print('Getting exciton weigths') 

        if filename_SS is None:
            w_S,A_SS,Sindex_S3,e_skn,H_SS = self.model_BSE(mode_bands=mode_bands,hl_mass=hl_mass,el_mass=el_mass,\
                                                 mode=mode,Wq_name=Wq_name,integrateGamma=integrateGamma,overlap=overlap,return_more=True) 
        else:
            w_S,A_SS,Sindex_S3,e_skn,H_SS = pickle.load(open(filename_SS))

        x_x,k_xc,k_x,X = self.find_k_from_kpoints(kpoints)

        lamda = np.argmin(w_S) 
        print (' ')
        print ('Exciton Energy')
        print (w_S[lamda]*Hartree)
        print (' ')

        A_S = A_SS[:,lamda]
        A_list = []
        e_v_list = []
        e_c_list = []

        kd = self.kd
        kp_1bz = to1bz(kd.bzk_kc,self.acell_cv)
        
        for ik in k_x:
            iS = self.find_Sindex_S3(ik,bands[0],bands[1],Sindex_S3)
            A_list.append(A_S[iS])

            ibzkpt = kd.bz2ibz_k[ik]
            e_v_list.append(e_skn[0][ibzkpt,bands[0]]) 
            e_c_list.append(e_skn[0][ibzkpt,bands[1]]) 
                
                
        A_k = np.array(A_list)
        e_v_k = np.array(e_v_list)
        e_c_k = np.array(e_c_list)
        return A_k, e_v_k, e_c_k, x_x, k_xc, k_x

#--------------------------------------------------------------------------------------------------------------   

    def get_H_elements(self, bands, kpoints, filename_SS=None,mode_bands='real',hl_mass=None,el_mass=None,mode='Q2D',Wq_name=None,integrateGamma=None):
        """ It gives the matrix elements of H along the direction defined by kpoints (the second index of the matrix being the one corresponding to the firt kpoints)"""

        print('Getting two particles Hamiltonian matrix elements') 

        if filename_SS is None:
            w_S,A_SS,Sindex_S3,e_skn,H_SS = self.model_BSE(mode_bands=mode_bands,hl_mass=hl_mass,el_mass=el_mass,\
                                                 mode=mode,Wq_name=Wq_name,integrateGamma=integrateGamma,return_more=True) 
        else:
            w_S,A_SS,Sindex_S3,e_skn,H_SS = pickle.load(open(filename_SS))

        x_x,k_xc,k_x,X = self.find_k_from_kpoints(kpoints)

        lamda = self.find_Sindex_S3(k_x[0],bands[0],bands[1],Sindex_S3)


        H_S = H_SS[:,lamda]
        H_list = []

        
        for ik in k_x:
            iS = self.find_Sindex_S3(ik,bands[0],bands[1],Sindex_S3)
            H_list.append(H_S[iS])

                
        H_k = np.array(H_list)
        return H_k, x_x, k_xc, k_x

#--------------------------------------------------------------------------------------------------------------   

    def get_chi_2D(self,chi_name,q_list):
        """Calculate the monopole and dipole contribution to the
        2D susceptibillity chi_2D, defined as

        ::

          \chi^M_2D(q, \omega) = \int\int dr dr' \chi(q, \omega, r,r') \\
                              = L \chi_{G=G'=0}(q, \omega)
          \chi^D_2D(q, \omega) = \int\int dr dr' z \chi(q, \omega, r,r') z'
                               = 1/L sum_{G_z,G_z'} z_factor(G_z)
                               chi_{G_z,G_z'} z_factor(G_z'),
          Where z_factor(G_z) =  +/- i e^{+/- i*G_z*z0}
          (L G_z cos(G_z L/2)-2 sin(G_z L/2))/G_z^2

        input parameters:
        
        filenames: list of str
            list of chi_wGG.pckl files for different q calculated with
            the DielectricFunction module in GPAW
        name: str
            name writing output files
        """
        nq = len(q_list)
        nw = 1
        omega_w=np.array([0.])

        q_list_abs = []
        
        pd, chi_wGG, eps_wGG = pickle.load(open(chi_name+'-q-%s.pckl'%q_list[10]))
            
        r = pd.gd.get_grid_point_coordinates()
        z = r[2, 0, 0, :]
        L = pd.gd.cell_cv[2, 2]  # Length of cell in Bohr
        z0 = L / 2.  # position of layer
        chiM_2D_qw = np.zeros([nq, nw], dtype=complex)
        chiD_2D_qw = np.zeros([nq, nw], dtype=complex)
        drho_M_qz = np.zeros([nq, len(z)], dtype=complex)  # induced density
        drho_D_qz = np.zeros([nq, len(z)], dtype=complex)  # induced dipole density

        def z_factor(z0, d, G, sign=1):
            factor = -1j * sign * np.exp(1j * sign * G * z0) * \
                (d * G * np.cos(G * d / 2.) - 2. * np.sin(G * d / 2.)) / G**2
            return factor


        def z_factor2(z0, d, G, sign=1):
            factor = sign * np.exp(1j * sign * G * z0) * np.sin(G * d / 2.)
            return factor


        for iq in range(nq):
            if q_list[iq] in self.qtoskip:
                continue
            pd, chi_wGG, eps_wGG = pickle.load(open(chi_name+'-q-%s.pckl'%q_list[iq]))
            npw = chi_wGG.shape[1]
            Gvec = pd.get_reciprocal_vectors(add_q=False)

            if q_list[iq]<1:
                q_list_abs.append(q_list[iq])
            else:
                q_list_abs.append(np.linalg.norm(pd.K_qv))

            Glist = []
            for iG in range(npw):  # List of G with Gx,Gy = 0
                if Gvec[iG, 0] == 0 and Gvec[iG, 1] == 0:
                    Glist.append(iG)

            chiM_2D_qw[iq] = L * chi_wGG[:, 0, 0]
            drho_M_qz[iq] += chi_wGG[0, 0, 0]

            for iG in Glist[1:]:
                G_z = Gvec[iG, 2]
                qGr_R = np.inner(G_z, z.T).T
                # Fourier transform to get induced density at \omega=0
                drho_M_qz[iq] += np.exp(1j * qGr_R) * chi_wGG[0, iG, 0]
                for iG1 in Glist[1:]:
                    G_z1 = Gvec[iG1, 2]
                    # integrate with z along both coordinates
                    factor = z_factor(z0, L, G_z)
                    factor1 = z_factor(z0, L, G_z1, sign=-1)
                    chiD_2D_qw[iq, :] += 1. / L * factor * chi_wGG[:, iG, iG1] * \
                        factor1
                    # induced dipole density due to V_ext = z
                    drho_D_qz[iq, :] += 1. / L * np.exp(1j * qGr_R) * \
                        chi_wGG[0, iG, iG1] * factor1
        # Normalize induced densities with chi
        drho_M_qz /= np.repeat(chiM_2D_qw[:, 0, np.newaxis], drho_M_qz.shape[1],
                               axis=1)
        drho_D_qz /= np.repeat(chiD_2D_qw[:, 0, np.newaxis], drho_M_qz.shape[1],
                               axis=1)

        n_sqs = q_list.index(1)
        for iq in range(n_sqs):
            drho_M_qz[iq,:] = drho_M_qz[n_sqs,:]
            drho_D_qz[iq,:] = drho_D_qz[n_sqs,:]

        """ Returns q array, frequency array, chi2D monopole and dipole, induced
        densities and z array (all in Bohr)
        """
        pickle.dump((np.array(q_list_abs), omega_w, chiM_2D_qw, chiD_2D_qw,
                     z, drho_M_qz, drho_D_qz), open(self.file_name+'-building_block.pckl', 'w'))

        print('Building Block Completed')
        return np.array(q_list_abs), omega_w, chiM_2D_qw, chiD_2D_qw, z, drho_M_qz, drho_D_qz



