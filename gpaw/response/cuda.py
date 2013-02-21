import _gpaw
import numpy as np
from math import pi, sqrt

sizeofdata = 16
sizeofint = 4
sizeofdouble = 8
sizeofpointer = 8

class GpuAlloc:
    def __init__(self, list):
        sizetot = 0
        for si in list:
            sizetot += si[1]

        status, ptr = _gpaw.cuMalloc(sizetot)
        self.ptr = ptr
        for si in list:
            exec 'self.' + si[0] + '= ptr'
            ptr += si[1]

    def free(self):
        _gpaw.cuFree(self.ptr)

class Indices(GpuAlloc):

    def __init__(self, chi0_wGG, basechi):
        sizelist = []
        self.sizeofint = 4

        sizelist.append(['index1_Q', basechi.nG0*self.sizeofint])
        sizelist.append(['index2_Q', basechi.nG0*self.sizeofint])
        sizelist.append(['Gindex_G', basechi.npw*self.sizeofint])

        GpuAlloc.__init__(self, sizelist)

        _gpaw.cuSetVector(basechi.npw,self.sizeofint,basechi.Gindex_G,1,self.Gindex_G,1)

    def kspecific_init(self, size, index1_g, index2_g):
        _gpaw.cuSetVector(size, self.sizeofint,index1_g,1,self.index1_Q,1)
        _gpaw.cuSetVector(size, self.sizeofint,index2_g,1,self.index2_Q,1)

class KPoints(GpuAlloc):
    def __init__(self, nibzk, ibzk_kc):
        sizelist = []
        sizelist.append(['ibzk_kc', nibzk*3*sizeofdouble])


        GpuAlloc.__init__(self, sizelist)
        _gpaw.cuSetVector(nibzk*3,sizeofdouble,ibzk_kc.ravel(),1,self.ibzk_kc,1)


class BaseCuda:



    def __init__(self):
        status, self.handle = _gpaw.cuCreate()

    def chi_init(self, chi, chi0_wGG):
        nmultix = chi.nmultix
        npw = chi.npw
        self.nG0 = nG0 = chi.nG0
        self.nG = nG = chi.nG
        self.chi0_w = []
        for iw in range(chi.Nw_local):
            status, matrix_GG = _gpaw.cuMalloc(npw*npw*sizeofdata)
            status = _gpaw.cuSetMatrix(npw, npw, sizeofdata,
                                       chi0_wGG[iw].copy(), npw,
                                       matrix_GG, npw)
            self.chi0_w.append(matrix_GG)
        
        status, self.rho_uG = _gpaw.cuMalloc(nmultix*npw*sizeofdata)
        status, self.tmp_Q = _gpaw.cuMalloc(nG0*sizeofdata)
        status, self.tmp_uQ = _gpaw.cuMalloc(nG0*nmultix*sizeofdata)
        status, self.psit1_R = _gpaw.cuMalloc(nG0*sizeofdata)
        status, self.psit2_R = _gpaw.cuMalloc(nG0*sizeofdata)
        status, self.expqr_R = _gpaw.cuMalloc(nG0*sizeofdata)

        self.cufftplanmany = _gpaw.cufft_planmany(nG[0],nG[1],nG[2],nmultix)
        self.cufftplan = _gpaw.cufft_plan3d(nG[0],nG[1],nG[2])
        self.ind =Indices(chi0_wGG, chi)

        if chi.optical_limit:
            status, self.opteikr_R = _gpaw.cuMalloc(nG0*sizeofdata)
            status, self.optpsit2_R = _gpaw.cuMalloc(nG0*sizeofdata)
            
        return 


    def paw_init(self, wfs, spos_ac, chi=None):

        kd = wfs.kd
        self.Na = Na = len(wfs.setups.id_a)
        
        # Non k-point dependence stuff
#        spos_ac = calc.atoms.get_scaled_positions()
        a_sa = np.int32(kd.symmetry.a_sa)
        op_scc = np.int32(wfs.kd.symmetry.op_scc)
        ibzk_kc = kd.ibzk_kc
        R_asii = {}
        for a in range(Na):
            R_asii[a] = wfs.setups[a].R_sii
        
        Ns = len(op_scc)
        nibzk = kd.nibzkpts
        nband = len(wfs.kpt_u[0].psit_nG)
        
        status, self.spos_ac = _gpaw.cuMalloc(Na*3*sizeofdouble)
        _gpaw.cuSetVector(Na*3,sizeofdouble,spos_ac.ravel(),1,self.spos_ac,1)
        
        status, self.a_sa = _gpaw.cuMalloc(Ns*Na*sizeofint)
        _gpaw.cuSetVector(Ns*Na,sizeofint,a_sa.ravel(),1,self.a_sa,1)
        
        status, self.op_scc = _gpaw.cuMalloc(Ns*9*sizeofint)
        _gpaw.cuSetVector(Ns*9,sizeofint,op_scc.ravel(),1,self.op_scc,1)

        self.kpoints = KPoints(nibzk, ibzk_kc)
        
        self.P_R_asii = np.zeros(Na, dtype=np.int64)            # P_R_asii is a pointer array on CPU
        self.P_P1_ani = np.zeros(Na, dtype=np.int64)       # for k
        self.P_P1_ai = np.zeros(Na, dtype=np.int64)
        self.P_P2_ani = np.zeros(Na, dtype=np.int64)       # for k+q
        self.P_P2_ai = np.zeros(Na, dtype=np.int64)
        self.host_Ni_a = Ni_a = np.zeros(Na, dtype=np.int32)
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

        
        status, self.R_asii = _gpaw.cuMalloc(Na*sizeofpointer) # dev_R_asii is a pointer array on GPU
        _gpaw.cuSetVector(Na,sizeofpointer,self.P_R_asii,1,self.R_asii,1)
            
        status, self.Ni_a = _gpaw.cuMalloc(Na*sizeofint)
        _gpaw.cuSetVector(Na,sizeofint,Ni_a,1,self.Ni_a,1)
        
        status, self.P1_ani = _gpaw.cuMalloc(Na*sizeofpointer)
        status, self.P1_ai = _gpaw.cuMalloc(Na*sizeofpointer)
        status, self.P2_ani = _gpaw.cuMalloc(Na*sizeofpointer)
        status, self.P2_ai = _gpaw.cuMalloc(Na*sizeofpointer)

        _gpaw.cuSetVector(self.Na,sizeofpointer,self.P_P1_ani,1,self.P1_ani,1)
        _gpaw.cuSetVector(self.Na,sizeofpointer,self.P_P1_ai,1,self.P1_ai,1)
        _gpaw.cuSetVector(self.Na,sizeofpointer,self.P_P2_ani,1,self.P2_ani,1)
        _gpaw.cuSetVector(self.Na,sizeofpointer,self.P_P2_ai,1,self.P2_ai,1)


        if chi is not None:
            self.P_phi_aGp = []
            for a, id in enumerate(wfs.setups.id_a):
                phi_Gp = chi.phi_aGp[a]
                npw, npair = phi_Gp.shape
                status,GPU_phi_Gp = _gpaw.cuMalloc(npw*npair*sizeofdata)
                status = _gpaw.cuSetMatrix(npair,npw,sizeofdata,phi_Gp,npair,GPU_phi_Gp,npair)
                self.P_phi_aGp.append(GPU_phi_Gp)
    
        self.P_P_ap = []
        for a, id in enumerate(wfs.setups.id_a):
            status,GPU_P_p = _gpaw.cuMalloc(Ni_a[a]*Ni_a[a]*sizeofdata)
            self.P_P_ap.append(GPU_P_p)

        return


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
        status, self.Q1_G = _gpaw.cuMalloc(self.ncoef*sizeofint)
        status, self.Q2_G = _gpaw.cuMalloc(self.ncoef2*sizeofint)
        _gpaw.cuSetVector(self.ncoef,sizeofint,Q1_G,1,self.Q1_G,1)
        _gpaw.cuSetVector(self.ncoef2,sizeofint,Q2_G,1,self.Q2_G,1)
           
        self.s1 = s1 = kd.sym_k[k]
        self.op1_cc = op1_cc = kd.symmetry.op_scc[s1]
        self.trans1 = trans1 = not ( (np.abs(op1_cc - np.eye(3, dtype=int)) < 1e-10).all() )
        self.s2 = s2 = kd.sym_k[kq_k[k]]
        self.op2_cc = op2_cc = kd.symmetry.op_scc[s2]
        self.trans2 = trans2 = not ( (np.abs(op2_cc - np.eye(3, dtype=int)) < 1e-10).all() )
        self.time_rev1 = time_rev1 = kd.time_reversal_k[k]
        self.time_rev2 = time_rev2 = kd.time_reversal_k[kq_k[k]]
        

        self.ind.kspecific_init(nx*ny*nz, index1_g, index2_g)

        deltak_c = (np.dot(op1_cc, kd.ibzk_kc[ibzkpt1]) -
                    np.dot(op2_cc, kd.ibzk_kc[ibzkpt2]) + chi.q_c)

        deltaeikr_R = np.exp(- 2j * pi * np.dot(np.indices(N_c).T, deltak_c / N_c).T)
        _gpaw.cuSetVector(nx*ny*nz,sizeofdata,deltaeikr_R.ravel(),1,self.expqr_R,1)

        for a in range(self.Na):
            Ni = self.host_Ni_a[a]
            # there is no need to copy the whole host_P_ani while using only one n !! 
            _gpaw.cuSetVector(chi.nbands*Ni,sizeofdata,calc.wfs.kpt_u[u1].P_ani[a].ravel(),1,self.P_P1_ani[a],1)
            _gpaw.cuSetVector(chi.nbands*Ni,sizeofdata,calc.wfs.kpt_u[u2].P_ani[a].ravel(),1,self.P_P2_ani[a],1)


        if chi.optical_limit:
#            self.ncoef = len(Q1_G) # the length of Q1_G and Q2_G are different for non optical transition
            self.vol = chi.vol
            self.qq_v = chi.qq_v
            deltak_c = np.dot(op1_cc, kd.ibzk_kc[ibzkpt1]) - kd.bzk_kc[k] 
            deltaeikr_R = np.exp(2j * pi * np.dot(np.indices(N_c).T, deltak_c / N_c).T)
            _gpaw.cuSetVector(nx*ny*nz,sizeofdata,deltaeikr_R.ravel(),1,self.opteikr_R,1)
            status, self.tmp_G = _gpaw.cuMalloc(self.ncoef*sizeofdata)
            status, self.tmp2_G = _gpaw.cuMalloc(self.ncoef*sizeofdata)
            status, self.dir_c = _gpaw.cuMalloc(3*sizeofdata)

            pd=calc.wfs.pd
            G_Gc = (pd.G_Qv[pd.Q_qG[ibzkpt1]] + np.dot(kd.bzk_kc[k], chi.bcell_cv)) * 1j
            status, self.G_cG = _gpaw.cuMalloc(3*self.ncoef*sizeofdata)

 
            for i in range(3):
                _gpaw.cuSetVector(self.ncoef,sizeofdata,G_Gc[:,i].copy(),1,
                                  self.G_cG+i*self.ncoef*sizeofdata,1)
        
        return 
    

    def exx_init(self, wfs):

        from gpaw.utilities import unpack
        self.nG0 = nG0 = wfs.gd.N_c[0] * wfs.gd.N_c[1] * wfs.gd.N_c[2]
        self.nG = nG = wfs.gd.N_c
        status, self.u1_R = _gpaw.cuMalloc(nG0*sizeofdata)
        status, self.u2_R = _gpaw.cuMalloc(nG0*sizeofdata)
        status, self.tmp_R = _gpaw.cuMalloc(nG0*sizeofdata)
        status, self.op_cc = _gpaw.cuMalloc(9*sizeofdouble)
        status, self.dk_c = _gpaw.cuMalloc(3*sizeofdouble)

#        self.P_Delta_apL = np.zeros(self.Na, dtype=np.int64) # is a pointer
#        self.P_Q_aL = np.zeros(self.Na, dtype=np.int64)
#
#        Delta_apL = {}
#        self.host_nL_a = np.zeros(self.Na, dtype=np.int32)
#        for a in range(self.Na):
#            Ni = self.host_Ni_a[a]
#            Delta_pL = wfs.setups[a].Delta_pL
#            nL = Delta_pL.shape[1]
#            self.host_nL_a[a] = nL
#
#            Delta_apL[a] = np.zeros((Ni*Ni, nL), complex) # has to be double other zgemv wont work 
#            for ii in range(Delta_pL.shape[1]):
#                Delta_apL[a][:,ii] = unpack(Delta_pL[:,ii].copy()).ravel()
#            status, dev_Delta_pL = _gpaw.cuMalloc(Ni*Ni*nL*sizeofdata)
#            status = _gpaw.cuSetMatrix(nL,Ni*Ni,sizeofdata,Delta_apL[a],nL,dev_Delta_pL,nL)
#            self.P_Delta_apL[a] = dev_Delta_pL
#
#            status, dev_Q_L = _gpaw.cuMalloc(nL*sizeofdata)
#            self.P_Q_aL[a] = dev_Q_L
#
#
#
#        status, self.nL_a = _gpaw.cuMalloc(self.Na*sizeofint)
#        _gpaw.cuSetVector(self.Na,sizeofint,self.host_nL_a,1,self.nL_a,1)
#
#        status, self.Delta_apL = _gpaw.cuMalloc(self.Na*sizeofpointer)
#        _gpaw.cuSetVector(self.Na,sizeofpointer,self.P_Delta_apL,1,self.Delta_apL,1)


    def get_wfs(self, host_psit_uG, Q_G, ncoef,nmultix):
                     
        nx,ny,nz = self.nG
        tmp_uQ = self.tmp_uQ

        status, dev_uG = _gpaw.cuMalloc(ncoef*nmultix*sizeofdata)
        _gpaw.cuSetVector(ncoef*nmultix,sizeofdata,host_psit_uG,1,dev_uG,1)
        _gpaw.cuMemset(tmp_uQ, 0, sizeofdata*self.nG0*nmultix)  # dev_Q has to be zeor
        _gpaw.cuMap_G2Q( dev_uG, tmp_uQ, Q_G, ncoef, self.nG0, nmultix )
        if nmultix == 1:
            _gpaw.cufft_execZ2Z(self.cufftplan, tmp_uQ, tmp_uQ, 1)
        else:
            _gpaw.cufft_execZ2Z(self.cufftplanmany, tmp_uQ, tmp_uQ, 1)
        _gpaw.cuZscal(self.handle, self.nG0*nmultix, 1./self.nG0, tmp_uQ, 1)
        
        _gpaw.cuFree(dev_uG)

        


#        status, dev_G = _gpaw.cuMalloc(ncoef*sizeofdata)
#        _gpaw.cuMemset(self.tmp_Q, 0, sizeofdata*self.nG0)  # dev_Q has to be zeor
#    
#        _gpaw.cuMap_G2Q( dev_G, self.tmp_Q, Q_G, ncoef )
#        _gpaw.cufft_execZ2Z(self.cufftplan, self.tmp_Q, self.tmp_Q,1)
#        _gpaw.cuZscal(self.handle, self.nG0, 1./self.nG0, self.tmp_Q, 1)
#        _gpaw.cuFree(dev_G)


        return

    def trans_wfs(self, psi_R, index_Q, trans, time_reversal):

        nx,ny,nz = self.nG
        if trans:
            # transform wavefunction here
            _gpaw.cuMemset(psi_R, 0, sizeofdata*self.nG0)  # dev_Q has to be zero
            _gpaw.cuTrans_wfs(self.tmp_Q, psi_R, index_Q, self.nG0)
        else:
            # Identity
            _gpaw.cuCopy_vector(self.tmp_Q, psi_R, self.nG0)
            
        if time_reversal:
            _gpaw.cuConj_vector(psi_R, self.nG0)

        return 


    def get_P_ai(self, P_P_ani, P_P_ai, P_ani, P_ai, host_P_ani, time_rev, s, ibzkpt, n):

        nband = len(host_P_ani[0])
        for a in range(self.Na):
            Ni = self.host_Ni_a[a]
            # there is no need to copy the whole host_P_ani while using only one n !! 
#            _gpaw.cuSetVector(nband*Ni,sizeofdata,host_P_ani[a].ravel(),1,P_P_ani[a],1)
            _gpaw.cuMemset(P_P_ai[a], 0, sizeofdata*Ni)
    
#        _gpaw.cuSetVector(self.Na,sizeofpointer,P_P_ani,1,P_ani,1)
#        _gpaw.cuSetVector(self.Na,sizeofpointer,P_P_ai,1,P_ai,1)
        
        _gpaw.cuGet_P_ai(self.spos_ac, self.kpoints.ibzk_kc, self.op_scc, self.a_sa, self.R_asii, 
                         P_ani, P_ai, self.Ni_a, time_rev, self.Na, s, ibzkpt, n)

        return 


    def calculate_Q_anL(self, host_P1_ani, n1_n, P_P1_ami, P2_ai, Delta_apL, Q_amL, mband):
        for a in range(self.Na):
            Ni = self.host_Ni_a[a]
            for im in range(mband):
                _gpaw.cuSetVector(Ni,sizeofdata,host_P1_ani[a][n1_n[im]].ravel(),1,
                                  P_P1_ami[a]+im*Ni*sizeofdata,1)

        _gpaw.cuGet_Q_anL(self.P1_ami, P2_ai, Delta_apL, Q_amL, mband, self.Na, self.Ni_a, self.nL_a)


    def calculate_opt(self, psit1_R):

        # multiply the phase to get U_mk(r) since U_mk0(r) has different plw coef 
        _gpaw.cuMul(self.optpsit2_R, self.opteikr_R, self.optpsit2_R, self.nG0)
        _gpaw.cufft_execZ2Z(self.cufftplan, self.optpsit2_R, 
                        self.optpsit2_R, -1)  # R -> Q
        _gpaw.cuMap_Q2G(self.optpsit2_R, self.tmp_G, # reduce planewaves
                        self.Q1_G, self.ncoef) # for optical limit self.dev_Q1_G = self.dev_Q2_G

        for ix in range(3):   # multiple the plw coef by [ 1j (k+G) ]
            _gpaw.cuMul(self.tmp_G, self.G_cG+ix*self.ncoef*sizeofdata,
                         self.tmp2_G, self.ncoef)
#            _gpaw.cuDevSynch()
            _gpaw.cuMemset(self.optpsit2_R, 0, sizeofdata*self.nG0)
            _gpaw.cuMap_G2Q(self.tmp2_G, self.optpsit2_R, 
                            self.Q1_G, self.ncoef, self.nG0, 1 )
            _gpaw.cufft_execZ2Z(self.cufftplan, self.optpsit2_R, 
                            self.optpsit2_R, 1)  # Q -> R
            # has to take into account fft (1/self.nG0) and gd.integrate(self.vol/self.nG0)
            _gpaw.cuZscal(self.handle, self.nG0, self.vol/(self.nG0*self.nG0),
                          self.optpsit2_R, 1)
            _gpaw.cuMulc(psit1_R, self.optpsit2_R, # (psit1_R*opteikr_R).conj() * optpsit2_R
                         self.optpsit2_R, self.nG0)
    
            alpha = -1j * self.qq_v[ix]                    
            _gpaw.cugemm(self.handle,alpha, self.optpsit2_R, self.opteikr_R, 0.0,
                         self.dir_c+ix*sizeofdata,
                         1, 1, self.nG0, 1, 1, 1, 2, 0) # transb(Hermitian), transa

        tmpp = np.zeros(3, dtype=complex)               # similar to np.dot(-1j*qq_v*tmp)
        status = _gpaw.cuGetVector(3,sizeofdata,self.dir_c,1, tmpp, 1)
        rhoG0 = np.array([np.sum(tmpp),])

        return rhoG0


    def chi_free(self, chi):

        for iw in range(chi.Nw_local):
            _gpaw.cuFree(self.chi0_w[iw])
        _gpaw.cuFree(self.rho_uG)
        _gpaw.cuFree(self.spos_ac)
        _gpaw.cuFree(self.a_sa)
        _gpaw.cuFree(self.op_scc)
        self.kpoints.free()
        _gpaw.cuFree(self.tmp_Q)
        _gpaw.cuFree(self.psit1_R)
        _gpaw.cuFree(self.psit2_R)
        _gpaw.cuFree(self.expqr_R)
        self.ind.free()
        if chi.optical_limit:
            _gpaw.cuFree(self.opteikr_R)
            _gpaw.cuFree(self.optpsit2_R)

        return

    
    def kspecific_free(self, chi):
        _gpaw.cuFree(self.Q1_G)
        _gpaw.cuFree(self.Q2_G)

        if chi.optical_limit:
            _gpaw.cuFree(self.tmp_G)
            _gpaw.cuFree(self.tmp2_G)
            _gpaw.cuFree(self.dir_c)
            _gpaw.cuFree(self.G_cG)

        return 



    def paw_free(self):

        _gpaw.cuFree(self.P1_ani) # they are pointer array
        _gpaw.cuFree(self.P1_ai)
        _gpaw.cuFree(self.P2_ani)
        _gpaw.cuFree(self.P2_ai)
        _gpaw.cuFree(self.R_asii)

        for a in range(self.Na):
            _gpaw.cuFree(self.P_phi_aGp[a])                    
            _gpaw.cuFree(self.P_P_ap[a])
            _gpaw.cuFree(self.P_R_asii[a])
            _gpaw.cuFree(self.P_P1_ani[a])
            _gpaw.cuFree(self.P_P1_ai[a])
            _gpaw.cuFree(self.P_P2_ani[a])
            _gpaw.cuFree(self.P_P2_ai[a])
            
        _gpaw.cuFree(self.Ni_a)

