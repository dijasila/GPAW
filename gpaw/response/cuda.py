import _gpaw
import numpy as np
from math import pi, sqrt

sizeofdata =16
sizeofint = 4
sizeofdouble = 8
sizeofpointer = 8

class BASECUDA:



    def __init__(self):
        status, self.handle = _gpaw.cuCreate()

    def chi_init(self, chi, chi0_wGG):
        nmultix = chi.nmultix
        npw = chi.npw
        self.nG0 = nG0 = chi.nG0
        self.nG = nG = chi.nG
        self.matrixlist_w = []
        for iw in range(chi.Nw_local):
            status, matrixGPU_GG = _gpaw.cuMalloc(npw*npw*sizeofdata)
            status = _gpaw.cuSetMatrix(npw, npw, sizeofdata,
                                       chi0_wGG[iw].copy(), npw,
                                       matrixGPU_GG, npw)
            self.matrixlist_w.append(matrixGPU_GG)
        
        status, self.GPU_rho_uG = _gpaw.cuMalloc(nmultix*npw*sizeofdata)
        status, self.dev_Qtmp = _gpaw.cuMalloc(nG0*sizeofdata)
        status, self.dev_psit1_R = _gpaw.cuMalloc(nG0*sizeofdata)
        status, self.dev_psit2_R = _gpaw.cuMalloc(nG0*sizeofdata)
        status, self.dev_index1_Q = _gpaw.cuMalloc(nG0*sizeofint)
        status, self.dev_index2_Q = _gpaw.cuMalloc(nG0*sizeofint)
        status, self.dev_expqr_R = _gpaw.cuMalloc(nG0*sizeofdata)

        self.cufftplan = _gpaw.cufft_plan3d(nG[0],nG[1],nG[2])

        status, self.dev_Gindex_G = _gpaw.cuMalloc(npw*sizeofint)
        _gpaw.cuSetVector(npw,sizeofint,chi.Gindex_G,1,self.dev_Gindex_G,1)

        if chi.optical_limit:
            status, self.dev_opteikr_R = _gpaw.cuMalloc(nG0*sizeofdata)
            status, self.dev_optpsit2_R = _gpaw.cuMalloc(nG0*sizeofdata)
            
        return #matrixlist_w, GPU_rho_uG, dev_Qtmp, dev_psit1_R, dev_psit2_R, dev_expqr_R, dev_Gindex_G


    def paw_init(self, chi):

        calc = chi.calc
        kd = calc.wfs.kd
        self.Na = Na = len(calc.wfs.setups.id_a)
        
        # Non k-point dependence stuff
        spos_ac = calc.atoms.get_scaled_positions()
        a_sa = np.int32(kd.symmetry.a_sa)
        op_scc = np.int32(calc.wfs.kd.symmetry.op_scc)
        ibzk_kc = kd.ibzk_kc
        R_asii = {}
        for a in range(Na):
            R_asii[a] = calc.wfs.setups[a].R_sii
        
        Ns = len(op_scc)
        nibzk = kd.nibzkpts
        nband = calc.get_number_of_bands()
        
        status, self.dev_spos_ac = _gpaw.cuMalloc(Na*3*sizeofdouble)
        _gpaw.cuSetVector(Na*3,sizeofdouble,spos_ac.ravel(),1,self.dev_spos_ac,1)
        
        status, self.dev_a_sa = _gpaw.cuMalloc(Ns*Na*sizeofint)
        _gpaw.cuSetVector(Ns*Na,sizeofint,a_sa.ravel(),1,self.dev_a_sa,1)
        
        status, self.dev_op_scc = _gpaw.cuMalloc(Ns*9*sizeofint)
        _gpaw.cuSetVector(Ns*9,sizeofint,op_scc.ravel(),1,self.dev_op_scc,1)
        
        status, self.dev_ibzk_kc = _gpaw.cuMalloc(nibzk*3*sizeofdouble)
        _gpaw.cuSetVector(nibzk*3,sizeofdouble,ibzk_kc.ravel(),1,self.dev_ibzk_kc,1)
        
        self.P_R_asii = np.zeros(Na, dtype=np.int64)            # P_R_asii is a pointer array on CPU
        self.P_P1_ani = np.zeros(Na, dtype=np.int64)       # for k
        self.P_P1_ai = np.zeros(Na, dtype=np.int64)
        self.P_P2_ani = np.zeros(Na, dtype=np.int64)       # for k+q
        self.P_P2_ai = np.zeros(Na, dtype=np.int64)
        self.Ni_a = Ni_a = np.zeros(Na, dtype=np.int32)
        for a in range(Na):
            Ni = len(R_asii[a][0])
            Ni_a[a] = Ni
            status, dev_R_sii = _gpaw.cuMalloc(Ns*Ni*Ni*sizeofdouble)
            _gpaw.cuSetVector(Ns*Ni*Ni,sizeofdouble,R_asii[a].ravel(),1,dev_R_sii,1)
            self.P_R_asii[a] = dev_R_sii
        
            status, dev_P_ni = _gpaw.cuMalloc(nband*Ni*sizeofdata)
            self.P_P1_ani[a] = dev_P_ni
            status, dev_P_ni = _gpaw.cuMalloc(nband*Ni*sizeofdata)
            self.P_P2_ani[a] = dev_P_ni
            
            status, dev_P_i = _gpaw.cuMalloc(Ni*sizeofdata)
            self.P_P1_ai[a] = dev_P_i
            status, dev_P_i = _gpaw.cuMalloc(Ni*sizeofdata)
            self.P_P2_ai[a] = dev_P_i

        
        status, self.dev_R_asii = _gpaw.cuMalloc(Na*sizeofpointer) # dev_R_asii is a pointer array on GPU
        _gpaw.cuSetVector(Na,sizeofpointer,self.P_R_asii,1,self.dev_R_asii,1)
            
        status, self.dev_Ni_a = _gpaw.cuMalloc(Na*sizeofint)
        _gpaw.cuSetVector(Na,sizeofint,Ni_a,1,self.dev_Ni_a,1)
        
        status, self.dev_P1_ani = _gpaw.cuMalloc(Na*sizeofpointer)
        status, self.dev_P1_ai = _gpaw.cuMalloc(Na*sizeofpointer)
        status, self.dev_P2_ani = _gpaw.cuMalloc(Na*sizeofpointer)
        status, self.dev_P2_ai = _gpaw.cuMalloc(Na*sizeofpointer)

        self.P_phi_aGp = []
        self.P_P_ap = []
        for a, id in enumerate(calc.wfs.setups.id_a):
            phi_Gp = chi.phi_aGp[a]
            npw, npair = phi_Gp.shape
            status,GPU_phi_Gp = _gpaw.cuMalloc(npw*npair*sizeofdata)
            status = _gpaw.cuSetMatrix(npair,npw,sizeofdata,phi_Gp,npair,GPU_phi_Gp,npair)
            self.P_phi_aGp.append(GPU_phi_Gp)

            status,GPU_P_p = _gpaw.cuMalloc(npair*sizeofdata)
            self.P_P_ap.append(GPU_P_p)

        return #P_R_asii, P_P1_ani, P_P1_ai, dev_P1_ani, dev_P1_ai, P_P2_ani, P_P2_ai, dev_P2_ani, dev_P2_ai


    def kspecific_init(self, chi, spin, k, ibzkpt1, ibzkpt2, index1_g, index2_g):

        calc = chi.calc
        nx,ny,nz = chi.nG
        kq_k = chi.kq_k
        kd = chi.kd
        N_c = chi.gd.N_c
        
        self.u1 = u1 = calc.wfs.kd.get_rank_and_index(spin, ibzkpt1)[1]
        self.u2 = u2 = calc.wfs.kd.get_rank_and_index(spin, ibzkpt2)[1]
        Q1_G = calc.wfs.pd.Q_qG[calc.wfs.kpt_u[u1].q]
        Q2_G = calc.wfs.pd.Q_qG[calc.wfs.kpt_u[u2].q]
        self.ncoef = len(Q1_G)
        self.ncoef2 = len(Q2_G)
        status, self.dev_Q1_G = _gpaw.cuMalloc(self.ncoef*sizeofint)
        status, self.dev_Q2_G = _gpaw.cuMalloc(self.ncoef2*sizeofint)
        _gpaw.cuSetVector(self.ncoef,sizeofint,Q1_G,1,self.dev_Q1_G,1)
        _gpaw.cuSetVector(self.ncoef2,sizeofint,Q2_G,1,self.dev_Q2_G,1)
           
        self.s1 = s1 = kd.sym_k[k]
        self.op1_cc = op1_cc = kd.symmetry.op_scc[s1]
        self.trans1 = trans1 = not ( (np.abs(op1_cc - np.eye(3, dtype=int)) < 1e-10).all() )
        self.s2 = s2 = kd.sym_k[kq_k[k]]
        self.op2_cc = op2_cc = kd.symmetry.op_scc[s2]
        self.trans2 = trans2 = not ( (np.abs(op2_cc - np.eye(3, dtype=int)) < 1e-10).all() )
        self.time_rev1 = time_rev1 = kd.time_reversal_k[k]
        self.time_rev2 = time_rev2 = kd.time_reversal_k[kq_k[k]]
        
        _gpaw.cuSetVector(nx*ny*nz,sizeofint,index1_g,1,self.dev_index1_Q,1)
        _gpaw.cuSetVector(nx*ny*nz,sizeofint,index2_g,1,self.dev_index2_Q,1)

        deltak_c = (np.dot(op1_cc, kd.ibzk_kc[ibzkpt1]) -
                    np.dot(op2_cc, kd.ibzk_kc[ibzkpt2]) + chi.q_c)

        deltaeikr_R = np.exp(- 2j * pi * np.dot(np.indices(N_c).T, deltak_c / N_c).T)
        _gpaw.cuSetVector(nx*ny*nz,sizeofdata,deltaeikr_R.ravel(),1,self.dev_expqr_R,1)


        if chi.optical_limit:
#            self.ncoef = len(Q1_G) # the length of Q1_G and Q2_G are different for non optical transition
            self.vol = chi.vol
            self.qq_v = chi.qq_v
            deltak_c = np.dot(op1_cc, kd.ibzk_kc[ibzkpt1]) - kd.bzk_kc[k] 
            deltaeikr_R = np.exp(2j * pi * np.dot(np.indices(N_c).T, deltak_c / N_c).T)
            _gpaw.cuSetVector(nx*ny*nz,sizeofdata,deltaeikr_R.ravel(),1,self.dev_opteikr_R,1)
            status, self.dev_tmp_G = _gpaw.cuMalloc(self.ncoef*sizeofdata)
            status, self.dev_tmp2_G = _gpaw.cuMalloc(self.ncoef*sizeofdata)
            status, self.dev_dir_c = _gpaw.cuMalloc(3*sizeofdata)

            pd=calc.wfs.pd
            G_Gc = (pd.G_Qv[pd.Q_qG[ibzkpt1]] + np.dot(kd.bzk_kc[k], chi.bcell_cv)) * 1j
            status, self.dev_G_cG = _gpaw.cuMalloc(3*self.ncoef*sizeofdata)

 
            for i in range(3):
                _gpaw.cuSetVector(self.ncoef,sizeofdata,G_Gc[:,i].copy(),1,
                                  self.dev_G_cG+i*self.ncoef*sizeofdata,1)
        
        return 



    def get_wfs(self, psit_G, dev_Q_G):
                     
        nx,ny,nz = self.nG
        ncoef = len(psit_G) # k-point dependent

        status, dev_G = _gpaw.cuMalloc(ncoef*sizeofdata)
        _gpaw.cuSetVector(ncoef,sizeofdata,psit_G,1,dev_G,1)
        _gpaw.cuMemset(self.dev_Qtmp, 0, sizeofdata*self.nG0)  # dev_Q has to be zeor

        _gpaw.cuMap_G2Q( dev_G, self.dev_Qtmp, dev_Q_G, ncoef )
        _gpaw.cufft_execZ2Z(self.cufftplan, self.dev_Qtmp, self.dev_Qtmp,1)
        _gpaw.cuZscal(self.handle, self.nG0, 1./self.nG0, self.dev_Qtmp, 1)
        
        _gpaw.cuFree(dev_G)

        return

    def trans_wfs(self, dev_psi_R, dev_index_Q, trans, time_reversal):

        nx,ny,nz = self.nG
        if trans:
            # transform wavefunction here
            _gpaw.cuMemset(dev_psi_R, 0, sizeofdata*self.nG0)  # dev_Q has to be zero
            _gpaw.cuTrans_wfs(self.dev_Qtmp, dev_psi_R, dev_index_Q, self.nG0)
        else:
            # Identity
            _gpaw.cuCopy_vector(self.dev_Qtmp, dev_psi_R, self.nG0)
            
        if time_reversal:
            _gpaw.cuConj_vector(dev_psi_R, self.nG0)

        return 


    def get_P_ai(self, P_P_ani, P_P_ai, dev_P_ani, dev_P_ai, P_ani, time_rev, s, ibzkpt, n):

        nband = len(P_ani[0])
        for a in range(self.Na):
            Ni = self.Ni_a[a]
            _gpaw.cuSetVector(nband*Ni,sizeofdata,P_ani[a].ravel(),1,P_P_ani[a],1)
            _gpaw.cuMemset(P_P_ai[a], 0, sizeofdata*Ni)
    
        _gpaw.cuSetVector(self.Na,sizeofpointer,P_P_ani,1,dev_P_ani,1)
        _gpaw.cuSetVector(self.Na,sizeofpointer,P_P_ai,1,dev_P_ai,1)
        
        _gpaw.cuGet_P_ai(self.dev_spos_ac, self.dev_ibzk_kc, self.dev_op_scc, self.dev_a_sa, self.dev_R_asii, 
                         dev_P_ani, dev_P_ai, self.dev_Ni_a, time_rev, self.Na, s, ibzkpt, n)

        return 


    def calculate_opt(self, dev_psit1_R):

        # multiply the phase to get U_mk(r) since U_mk0(r) has different plw coef 
        _gpaw.cuMul(self.dev_optpsit2_R, self.dev_opteikr_R, self.dev_optpsit2_R, self.nG0)
        _gpaw.cufft_execZ2Z(self.cufftplan, self.dev_optpsit2_R, 
                        self.dev_optpsit2_R, -1)  # R -> Q
        _gpaw.cuMap_Q2G(self.dev_optpsit2_R, self.dev_tmp_G, # reduce planewaves
                        self.dev_Q1_G, self.ncoef) # for optical limit self.dev_Q1_G = self.dev_Q2_G

        for ix in range(3):   # multiple the plw coef by [ 1j (k+G) ]
            _gpaw.cuMul(self.dev_tmp_G, self.dev_G_cG+ix*self.ncoef*sizeofdata,
                         self.dev_tmp2_G, self.ncoef)
#            _gpaw.cuDevSynch()
            _gpaw.cuMemset(self.dev_optpsit2_R, 0, sizeofdata*self.nG0)
            _gpaw.cuMap_G2Q(self.dev_tmp2_G, self.dev_optpsit2_R, 
                            self.dev_Q1_G, self.ncoef )
            _gpaw.cufft_execZ2Z(self.cufftplan, self.dev_optpsit2_R, 
                            self.dev_optpsit2_R, 1)  # Q -> R
            # has to take into account fft (1/self.nG0) and gd.integrate(self.vol/self.nG0)
            _gpaw.cuZscal(self.handle, self.nG0, self.vol/(self.nG0*self.nG0),
                          self.dev_optpsit2_R, 1)
            _gpaw.cuMulc(dev_psit1_R, self.dev_optpsit2_R, # (psit1_R*opteikr_R).conj() * optpsit2_R
                         self.dev_optpsit2_R, self.nG0)
    
            alpha = -1j * self.qq_v[ix]                    
            _gpaw.cugemm(self.handle,alpha, self.dev_optpsit2_R, self.dev_opteikr_R, 0.0,
                         self.dev_dir_c+ix*sizeofdata,
                         1, 1, self.nG0, 1, 1, 1, 2, 0) # transb(Hermitian), transa

        tmpp = np.zeros(3, dtype=complex)               # similar to np.dot(-1j*qq_v*tmp)
        status = _gpaw.cuGetVector(3,sizeofdata,self.dev_dir_c,1, tmpp, 1)
        rhoG0 = np.array([np.sum(tmpp),])

        return rhoG0


    def chi_free(self, chi):

        for iw in range(chi.Nw_local):
            _gpaw.cuFree(self.matrixlist_w[iw])
        _gpaw.cuFree(self.GPU_rho_uG)
        _gpaw.cuFree(self.dev_spos_ac)
        _gpaw.cuFree(self.dev_a_sa)
        _gpaw.cuFree(self.dev_op_scc)
        _gpaw.cuFree(self.dev_ibzk_kc)
        _gpaw.cuFree(self.dev_Qtmp)
        _gpaw.cuFree(self.dev_psit1_R)
        _gpaw.cuFree(self.dev_psit2_R)
        _gpaw.cuFree(self.dev_expqr_R)
        _gpaw.cuFree(self.dev_Gindex_G)
        _gpaw.cuFree(self.dev_index1_Q)
        _gpaw.cuFree(self.dev_index2_Q)

        if chi.optical_limit:
            _gpaw.cuFree(self.dev_opteikr_R)
            _gpaw.cuFree(self.dev_optpsit2_R)

        return

    
    def kspecific_free(self, chi):
        _gpaw.cuFree(self.dev_Q1_G)
        _gpaw.cuFree(self.dev_Q2_G)

        if chi.optical_limit:
            _gpaw.cuFree(self.dev_tmp_G)
            _gpaw.cuFree(self.dev_tmp2_G)
            _gpaw.cuFree(self.dev_dir_c)
            _gpaw.cuFree(self.dev_G_cG)

        return 



    def paw_free(self):

        _gpaw.cuFree(self.dev_P1_ani) # they are pointer array
        _gpaw.cuFree(self.dev_P1_ai)
        _gpaw.cuFree(self.dev_P2_ani)
        _gpaw.cuFree(self.dev_P2_ai)

        for a in range(self.Na):
            _gpaw.cuFree(self.P_phi_aGp[a])                    
            _gpaw.cuFree(self.P_P_ap[a])
            _gpaw.cuFree(self.P_R_asii[a])
            _gpaw.cuFree(self.P_P1_ani[a])
            _gpaw.cuFree(self.P_P1_ai[a])
            _gpaw.cuFree(self.P_P2_ani[a])
            _gpaw.cuFree(self.P_P2_ai[a])
            _gpaw.cuFree(self.dev_R_asii)
            _gpaw.cuFree(self.dev_Ni_a)

