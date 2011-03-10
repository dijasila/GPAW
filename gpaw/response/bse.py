from time import time, ctime
import numpy as np
import pickle
from math import pi
from ase.units import Hartree
from ase.io import write
from gpaw.mpi import world, size, rank, serial_comm
from gpaw.utilities import devnull
from gpaw.response.base import BASECHI
from gpaw.response.parallel import parallel_partition
from gpaw.response.df import DF

class BSE(BASECHI):
    """This class defines Belth-Selpether equations."""

    def __init__(self,
                 calc=None,
                 nbands=None,
                 nc=None,
                 nv=None,
                 w=None,
                 q=None,
                 eshift=None,
                 ecut=10.,
                 eta=0.2,
                 ftol=1e-5,
                 txt=None,
                 optical_limit=False,
                 positive_w=False, # True : use Tamm-Dancoff Approx
                 use_W=True): # True: include screened interaction kernel

        BASECHI.__init__(self, calc, nbands, w, q, ecut,
                     eta, ftol, txt, optical_limit)


        self.epsilon_w = None
        self.positive_w = positive_w
        self.nc = nc # conduction band index
        self.nv = nv # valence band index
        self.use_W = use_W
        self.eshift = eshift

    def initialize(self):

        self.printtxt('')
        self.printtxt('-----------------------------------------------')
        self.printtxt('Bethe Salpeter Equation calculation started at:')
        self.printtxt(ctime())

        BASECHI.initialize(self)

        if self.eshift is not None:
            self.add_discontinuity(self.eshift)
        
        calc = self.calc
        self.kd = kd = calc.wfs.kd

        # frequency points init
        self.dw = self.w_w[1] - self.w_w[0]
        assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
        assert self.w_w.max() == self.w_w[-1]

        self.dw /= Hartree
        self.w_w  /= Hartree
        self.wmax = self.w_w[-1] 
        self.Nw  = int(self.wmax / self.dw) + 1

        # band init
        if self.nc is None and self.positive_w is True: # applied only to semiconductor
            nv = self.nvalence / 2 - 1
            self.nv = np.array([nv, nv+1]) # conduction band start / end
            self.nc = np.array([nv+1, nv+2]) # valence band start / end
            self.printtxt('Number of electrons: %d' %(self.nvalence))
            self.printtxt('Valence band included        : (band %d to band %d)' %(self.nv[0],self.nv[1]-1))
            self.printtxt('Conduction band included     : (band %d to band %d)' %(self.nc[0],self.nc[1]-1))
        elif self.nc == 'all' or self.positive_w is False: # applied to metals
            self.nv = np.array([0, self.nbands])
            self.nc = np.array([0, self.nbands])
            self.printtxt('All the bands are included')
        else:
            self.printtxt('User defined bands for BSE.')
            self.printtxt('Valence band included: (band %d to band %d)' %(self.nv[0],self.nv[1]-1))
            self.printtxt('Conduction band included: (band %d to band %d)' %(self.nc[0],self.nc[1]-1))            

        # find the pair index and initialized pair energy (e_i - e_j) and occupation(f_i-f_j)
        self.e_S = {}
        focc_s = {}
        self.Sindex_S3 = {}
        iS = 0
        kq_k = self.kq_k
        for k1 in range(self.nkpt):
            ibzkpt1 = kd.kibz_k[k1]
            ibzkpt2 = kd.kibz_k[kq_k[k1]]
            for n1 in range(self.nv[0], self.nv[1]): 
                for m1 in range(self.nc[0], self.nc[1]): 
                    focc = self.f_kn[ibzkpt1,n1] - self.f_kn[ibzkpt2,m1]
                    if not self.positive_w: # Dont use Tamm-Dancoff Approx.
                        check_ftol = np.abs(focc) > self.ftol
                    else:
                        check_ftol = focc > self.ftol
                    if check_ftol:           
                        self.e_S[iS] =self.e_kn[ibzkpt2,m1] - self.e_kn[ibzkpt1,n1]
                        focc_s[iS] = focc
                        self.Sindex_S3[iS] = (k1, n1, m1)
                        iS += 1
        self.nS = iS
        self.focc_S = np.zeros(self.nS)
        for iS in range(self.nS):
            self.focc_S[iS] = focc_s[iS]

        # parallel init
        self.Scomm = world
        # kcomm and wScomm is only to be used when wavefunctions r parallelly distributed.
        self.kcomm = world
        self.wScomm = serial_comm
        
        self.nS, self.nS_local, self.nS_start, self.nS_end = parallel_partition(
                               self.nS, world.rank, world.size, reshape=False)
        self.nq, self.nq_local, self.q_start, self.q_end = parallel_partition(
                               self.nkpt, world.rank, world.size, reshape=False)
        self.print_bse()

        self.get_phi_aGp()

        # Coulomb kernel init
        self.kc_G = np.zeros(self.npw)
        for iG in range(self.npw):
            index = self.Gindex_G[iG]
            qG = np.dot(self.q_c + self.Gvec_Gc[iG], self.bcell_cv)
            self.kc_G[iG] = 1. / np.inner(qG, qG)
        if self.optical_limit:
            self.kc_G[0] = 0.
        self.printtxt('')
        
        return


    def calculate(self):

        calc = self.calc
        f_kn = self.f_kn
        e_kn = self.e_kn
        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        focc_S = self.focc_S
        e_S = self.e_S

        if self.use_W:
            self.printtxt('Calculating screening interaction kernel.')
            W_qGG = self.screened_interaction_kernel()
            
        # calculate kernel
        K_SS = np.zeros((self.nS, self.nS), dtype=complex)
        W_SS = np.zeros_like(K_SS)
        self.rhoG0_S = np.zeros((self.nS), dtype=complex)

        t0 = time()
        self.printtxt('Calculating BSE matrix elements.')
        
        for iS in range(self.nS_start, self.nS_end):
            k1, n1, m1 = self.Sindex_S3[iS]
            rho1_G = self.density_matrix(n1,m1,k1)
            self.rhoG0_S[iS] = rho1_G[0]

            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                rho2_G = self.density_matrix(n2,m2,k2)
                K_SS[iS, jS] = np.sum(rho1_G.conj() * rho2_G * self.kc_G)

                if self.use_W:
                    rho3_G = self.density_matrix(n1,n2,k1,k2)
                    rho4_G = self.density_matrix(m1,m2,k1,k2)
                    q_c = bzk_kc[k2] - bzk_kc[k1]
                    iq = self.kd.where_is_q(q_c)
                    W_GG = W_qGG[iq].copy()
    
                    if k1 == k2:
                        ik = self.kd.kibz_k[k1]
                        deg_bands1 = np.abs(self.e_kn[ik, n1] - self.e_kn[ik, n2]) < 1e-4
                        deg_bands2 = np.abs(self.e_kn[ik, m1] - self.e_kn[ik, m2]) < 1e-4
                    
                        if (deg_bands1 or deg_bands2):
                            tmp_G = np.zeros(self.npw)
                            const = 2.*self.vol*(6*pi**2/self.vol)**(2./3.)*self.dfinv_q0
                            q = np.array([0.0001,0,0])
                            for jG in range(self.npw):
                                qG = np.dot(q+self.Gvec_Gc[jG], self.bcell_cv)
                                tmp_G[jG] = 1./np.sqrt(np.inner(qG,qG))
                            tmp_G *= const
                            if deg_bands1 and not deg_bands2:
                                W_GG[0,:] = tmp_G
                            elif not deg_bands1 and deg_bands2:
                                W_GG[:,0] = tmp_G
                            elif deg_bands1 and deg_bands2:
                                W_GG[:,0] = tmp_G
                                W_GG[0,:] = tmp_G
                                W_GG[0,0] =  2./pi*(6*pi**2/self.vol)**(1./3.) * self.dfinv_q0*self.vol
    
                    tmp_GG = np.outer(rho3_G.conj(), rho4_G) * W_GG
                    W_SS[iS, jS] = np.sum(tmp_GG)

            self.timing(iS, t0, self.nS_local, 'pair orbital') 

        K_SS *= 4 * pi / self.vol
        if self.use_W:
            K_SS -= 0.5 * W_SS / self.vol
        world.sum(K_SS)
        world.sum(self.rhoG0_S)

        # get and solve hamiltonian
        H_SS = np.zeros_like(K_SS)
        for iS in range(self.nS):
            H_SS[iS,iS] = e_S[iS]
            for jS in range(self.nS):
                H_SS[iS,jS] += focc_S[iS] * K_SS[iS,jS]

        self.w_S, self.v_SS = np.linalg.eig(H_SS)
        
        return 


    def screened_interaction_kernel(self):
        """Calcuate W_GG(q)"""

        self.kd.get_bz_q_points()
        dfinv_qGG = np.zeros((self.nkpt, self.npw, self.npw))
        kc_qGG = np.zeros((self.nkpt, self.npw, self.npw))
        dfinv_q0 = np.zeros(1)

        t0 = time()
        for iq in range(self.q_start, self.q_end):
            q = self.kd.bzq_kc[iq]
            optical_limit=False
            if (np.abs(q) < self.ftol).all():
                optical_limit=True
                q = np.array([0.00001, 0, 0])
            df = DF(calc=self.calc, q=q, w=(0.,), nbands=self.nbands,
                    optical_limit=optical_limit,
                    hilbert_trans=False, xc='RPA',
                    eta=0., ecut=self.ecut*Hartree, txt='no_output', comm=serial_comm)
            dfinv_qGG[iq] = df.get_inverse_dielectric_matrix(xc='RPA')[0]

            for iG in range(self.npw):
                for jG in range(self.npw):
                    qG1 = np.dot(q + self.Gvec_Gc[iG], self.bcell_cv)
                    qG2 = np.dot(q + self.Gvec_Gc[jG], self.bcell_cv)
                    kc_qGG[iq,iG,jG] = 1. / np.sqrt(np.dot(qG1, qG1) * np.dot(qG2,qG2))

            self.timing(iq, t0, self.nq_local, 'iq')

            assert df.npw == self.npw
            if optical_limit:
                dfinv_q0[0] = dfinv_qGG[iq, 0,0]
                
        W_qGG = 4 * pi * dfinv_qGG * kc_qGG
        world.sum(W_qGG)
        world.broadcast(dfinv_q0, 0)
        self.dfinv_q0 = dfinv_q0[0]

        return W_qGG
                                          
    def print_bse(self):

        printtxt = self.printtxt

        printtxt('Number of frequency points   : %d' %(self.Nw) )
        printtxt('Number of pair orbitals      : %d' %(self.nS) )
        printtxt('Parallelization scheme:')
        printtxt('   Total cpus         : %d' %(world.size))
        printtxt('   pair orb parsize   : %d' %(self.Scomm.size))        
        
        return


    def get_dielectric_function(self, filename='df.dat', overlap=True):

        if self.epsilon_w is None:
            self.initialize()
            self.calculate()
            self.printtxt('Calculating dielectric function.')

            w_S = self.w_S
            v_SS = self.v_SS # v_SS[:,lamda]
            rhoG0_S = self.rhoG0_S
            focc_S = self.focc_S

            # get overlap matrix
            tmp = np.dot(v_SS.conj().T, v_SS )
            overlap_SS = np.linalg.inv(tmp)
    
            # get chi
            epsilon_w = np.zeros(self.Nw, dtype=complex)
            t0 = time()

            A_S = np.dot(rhoG0_S, v_SS)
            B_S = np.dot(rhoG0_S*focc_S, v_SS)
            C_S = np.dot(B_S.conj(), overlap_SS.T) * A_S

            for iw in range(self.Nw):
                tmp_S = 1. / (iw*self.dw - w_S + 1j*self.eta)
                epsilon_w[iw] += np.dot(tmp_S, C_S)
    
            epsilon_w *=  - 4 * pi / np.inner(self.qq_v, self.qq_v) / self.vol
            epsilon_w += 1        

            self.epsilon_w = epsilon_w
    
        if rank == 0:
            f = open(filename,'w')
            for iw in range(self.Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, np.real(epsilon_w[iw]), np.imag(epsilon_w[iw])
            f.close()
        # Wait for I/O to finish
        world.barrier()

        return epsilon_w


    def timing(self, i, t0, n_local, txt):

        if i == 0:
            dt = time() - t0
            self.totaltime = dt * n_local
            self.printtxt('  Finished %s 0 in %f seconds, estimated %f seconds left.' %(txt, dt, self.totaltime))
            
        if rank == 0 and n_local // 5 > 0:            
            if i > 0 and i % (n_local // 5) == 0:
                dt =  time() - t0
                self.printtxt('  Finished %s %d in %f seconds, estimated %f seconds left.  '%(txt, i, dt, self.totaltime - dt) )

        return


    def add_discontinuity(self, shift):

        eFermi = self.calc.occupations.get_fermi_level()
        for i in range(self.e_kn.shape[1]):
            for k in range(self.e_kn.shape[0]):
                if self.e_kn[k,i] > eFermi:
                    self.e_kn[k,i] += shift / Hartree

        return
    

    def get_e_h_density(self, lamda=None, filename=None):

        if filename is not None:
            self.load(filename)
            self.initialize()
            
        gd = self.gd
        w_S = self.w_S
        v_SS = self.v_SS
        A_S = v_SS[:, lamda]
        kq_k = self.kq_k
        kd = self.kd

        # Electron density
        nte_R = gd.zeros()
        
        for iS in range(self.nS_start, self.nS_end):
            print 'electron density:', iS
            k1, n1, m1 = self.Sindex_S3[iS]
            ibzkpt1 = kd.kibz_k[k1]
            psitold_g = self.get_wavefunction(ibzkpt1, n1)
            psit1_g = kd.transform_wave_function(psitold_g, k1)

            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                if m1 == m2 and k1 == k2:
                    psitold_g = self.get_wavefunction(ibzkpt1, n2)
                    psit2_g = kd.transform_wave_function(psitold_g, k1)

                    nte_R += A_S[iS] * A_S[jS].conj() * psit1_g.conj() * psit2_g

        # Hole density
        nth_R = gd.zeros()
        
        for iS in range(self.nS_start, self.nS_end):
            print 'hole density:', iS
            k1, n1, m1 = self.Sindex_S3[iS]
            ibzkpt1 = kd.kibz_k[kq_k[k1]]
            psitold_g = self.get_wavefunction(ibzkpt1, m1)
            psit1_g = kd.transform_wave_function(psitold_g, kq_k[k1])

            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                if n1 == n2 and k1 == k2:
                    psitold_g = self.get_wavefunction(ibzkpt1, m2)
                    psit2_g = kd.transform_wave_function(psitold_g, kq_k[k1])

                    nth_R += A_S[iS] * A_S[jS].conj() * psit1_g * psit2_g.conj()
                    
        self.Scomm.sum(nte_R)
        self.Scomm.sum(nth_R)


        if rank == 0:
            write('rho_e.cube',self.calc.atoms, format='cube', data=nte_R)
            write('rho_h.cube',self.calc.atoms, format='cube', data=nth_R)
            
        world.barrier()
        
        return 

    def get_excitation_wavefunction(self, lamda=None,filename=None, re_c=None, rh_c=None):
        """ garbage at the moment. come back later"""
        if filename is not None:
            self.load(filename)
            self.initialize()
            
        gd = self.gd
        w_S = self.w_S
        v_SS = self.v_SS
        A_S = v_SS[:, lamda]
        kq_k = self.kq_k
        kd = self.kd

        nx, ny, nz = self.nG[0], self.nG[1], self.nG[2]
        nR = 9
        nR2 = (nR - 1 ) // 2
        if re_c is not None:
            psith_R = gd.zeros(dtype=complex)
            psith2_R = np.zeros((nR*nx, nR*ny, nz), dtype=complex)
            
        elif rh_c is not None:
            psite_R = gd.zeros(dtype=complex)
            psite2_R = np.zeros((nR*nx, ny, nR*nz), dtype=complex)
        else:
            self.printtxt('No wavefunction output !')
            return
            
        for iS in range(self.nS_start, self.nS_end):

            k, n, m = self.Sindex_S3[iS]
            ibzkpt1 = kd.kibz_k[k]
            ibzkpt2 = kd.kibz_k[kq_k[k]]
            print 'hole wavefunction', iS, (k,n,m),A_S[iS]
            
            psitold_g = self.get_wavefunction(ibzkpt1, n)
            psit1_g = kd.transform_wave_function(psitold_g, k)

            psitold_g = self.get_wavefunction(ibzkpt2, m)
            psit2_g = kd.transform_wave_function(psitold_g, kq_k[k])

            if re_c is not None:
                # given electron position, plot hole wavefunction
                tmp = A_S[iS] * psit1_g[re_c].conj() * psit2_g
                psith_R += tmp

                k_c = self.bzk_kc[k] + self.q_c
                for i in range(nR):
                    for j in range(nR):
                        R_c = np.array([i-nR2, j-nR2, 0])
                        psith2_R[i*nx:(i+1)*nx, j*ny:(j+1)*ny, 0:nz] += \
                                                tmp * np.exp(1j*2*pi*np.dot(k_c,R_c))
                
            elif rh_c is not None:
                # given hole position, plot electron wavefunction
                tmp = A_S[iS] * psit1_g.conj() * psit2_g[rh_c] * self.expqr_g
                psite_R += tmp

                k_c = self.bzk_kc[k]
                k_v = np.dot(k_c, self.bcell_cv)
                for i in range(nR):
                    for j in range(nR):
                        R_c = np.array([i-nR2, 0, j-nR2])
                        R_v = np.dot(R_c, self.acell_cv)
                        assert np.abs(np.dot(k_v, R_v) - np.dot(k_c, R_c) * 2*pi).sum() < 1e-5
                        psite2_R[i*nx:(i+1)*nx, 0:ny, j*nz:(j+1)*nz] += \
                                                tmp * np.exp(-1j*np.dot(k_v,R_v))
                
            else:
                pass

        if re_c is not None:
            self.Scomm.sum(psith_R)
            self.Scomm.sum(psith2_R)
            if rank == 0:
                write('psit_h.cube',self.calc.atoms, format='cube', data=psith_R)

                atoms = self.calc.atoms
                shift = atoms.cell[0:2].copy()
                positions = atoms.positions
                atoms.cell[0:2] *= nR2
                atoms.positions += shift * (nR2 - 1)
                
                write('psit_bigcell_h.cube',atoms, format='cube', data=psith2_R)
        elif rh_c is not None:
            self.Scomm.sum(psite_R)
            self.Scomm.sum(psite2_R)
            if rank == 0:
                write('psit_e.cube',self.calc.atoms, format='cube', data=psite_R)

                atoms = self.calc.atoms
#                shift = atoms.cell[0:2].copy()
                positions = atoms.positions
                atoms.cell[0:2] *= nR2
#                atoms.positions += shift * (nR2 - 1)
                
                write('psit_bigcell_e.cube',atoms, format='cube', data=psite2_R)
                
        else:
            pass

        world.barrier()
            
        return
    

    def load(self, filename):

        data = pickle.load(open(filename))
        self.w_S  = data['w_S']
        self.v_SS = data['v_SS']

        self.printtxt('Read succesfully !')
        

    def save(self, filename):
        """Dump essential data"""

        data = {'w_S'  : self.w_S,
                'v_SS' : self.v_SS}
        
        if rank == 0:
            pickle.dump(data, open(filename, 'w'), -1)

        world.barrier()

