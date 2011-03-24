import numpy as np

from gpaw.overlap import Overlap
from gpaw.fd_operators import Laplace
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.utilities import unpack
from gpaw.io.tar import TarFileReference
from gpaw.lfc import BasisFunctions
from gpaw.utilities.blas import axpy
from gpaw.transformers import Transformer
from gpaw.fd_operators import Gradient
from gpaw.band_descriptor import BandDescriptor
from gpaw import extra_parameters
from gpaw.wavefunctions.fdpw import FDPWWaveFunctions
from gpaw.hs_operators import MatrixOperator
from gpaw.preconditioner import Preconditioner


class FDWaveFunctions(FDPWWaveFunctions):
    def __init__(self, stencil, diagksl, orthoksl, initksl,
                 gd, nvalence, setups, bd,
                 dtype, world, kd, timer=None):
        FDPWWaveFunctions.__init__(self, diagksl, orthoksl, initksl,
                                   gd, nvalence, setups, bd,
                                   dtype, world, kd, timer)

        self.wd = self.gd  # wave function descriptor
        
        # Kinetic energy operator:
        self.kin = Laplace(self.gd, -0.5, stencil, self.dtype, allocate=False)

        self.matrixoperator = MatrixOperator(orthoksl)

    def set_setups(self, setups):
        self.pt = LFC(self.gd, [setup.pt_j for setup in setups],
                      self.kpt_comm, dtype=self.dtype, forces=True)
        FDPWWaveFunctions.set_setups(self, setups)

    def set_positions(self, spos_ac):
        if not self.kin.is_allocated():
            self.kin.allocate()
        FDPWWaveFunctions.set_positions(self, spos_ac)

    def summary(self, fd):
        fd.write('Mode: Finite-difference\n')
        
    def make_preconditioner(self, block=1):
        return Preconditioner(self.gd, self.kin, self.dtype, block)
    
    def apply_pseudo_hamiltonian(self, kpt, hamiltonian, psit_xG, Htpsit_xG):
        self.kin.apply(psit_xG, Htpsit_xG, kpt.phase_cd)
        hamiltonian.apply_local_potential(psit_xG, Htpsit_xG, kpt.s)

    def add_orbital_density(self, nt_G, kpt, n):
        if self.dtype == float:
            axpy(1.0, kpt.psit_nG[n]**2, nt_G)
        else:
            axpy(1.0, kpt.psit_nG[n].real**2, nt_G)
            axpy(1.0, kpt.psit_nG[n].imag**2, nt_G)

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        # Used in calculation of response part of GLLB-potential
        nt_G = nt_sG[kpt.s]
        if self.dtype == float:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G**2, nt_G)
        else:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G.real**2, nt_G)
                axpy(f, psit_G.imag**2, nt_G)

        # Hack used in delta-scf calculations:
        if hasattr(kpt, 'c_on'):
            assert self.bd.comm.size == 1
            d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                            dtype=complex)
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn += ne * np.outer(c_n.conj(), c_n)
            for d_n, psi0_G in zip(d_nn, kpt.psit_nG):
                for d, psi_G in zip(d_n, kpt.psit_nG):
                    if abs(d) > 1.e-12:
                        nt_G += (psi0_G.conj() * d * psi_G).real

    def calculate_kinetic_energy_density(self, tauct, grad_v):
        assert not hasattr(self.kpt_u[0], 'c_on')
        if isinstance(self.kpt_u[0].psit_nG, TarFileReference):
            raise RuntimeError('Wavefunctions have not been initialized.')

        taut_sG = self.gd.zeros(self.nspins)
        dpsit_G = self.gd.empty(dtype=self.dtype)
        for kpt in self.kpt_u:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                for v in range(3):
                    grad_v[v](psit_G, dpsit_G, kpt.phase_cd)
                    axpy(0.5 * f, abs(dpsit_G)**2, taut_sG[kpt.s])

        self.kpt_comm.sum(taut_sG)
        self.band_comm.sum(taut_sG)
        return taut_sG
        
    def calculate_forces(self, hamiltonian, F_av, td_correction=None):
        """ Calculate wavefunction forces with optional corrections for
            Ehrenfest dynamics
        """  

            
        #If td_correction is not none, we replace the overlap part of the
        #force, sum_n f_n eps_n < psit_n | dO / dR_a | psit_n>, with
        #sum_n f_n <psit_n | H S^-1 D^a + c.c. | psit_n >, with D^a
        #defined as D^a = sum_{i1,i2} | pt_i1^a > [O_{i1,i2} < d pt_i2^a / dR_a |
        #+ (< phi_i1^a | d phi_i2^a / dR_a > - < phit_i1^a | d phit_i2^a / dR_a >) < pt_i1^a|].
        #This is required in order to conserve the total energy also when electronic
        #excitations start to play a significant role.

        #TODO: move the corrections into the tddft directory
        
        # Calculate force-contribution from k-points:
        F_av.fill(0.0)
        F_aniv = self.pt.dict(self.bd.mynbands, derivative=True)
        #print 'self.dtype =', self.dtype
        for kpt in self.kpt_u:
            #self.overlap.update_k_point_projections(kpt)
            self.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            hpsit = self.gd.zeros(len(kpt.psit_nG), dtype=self.dtype)
            #eps_psit = self.gd.zeros(len(kpt.psit_nG), dtype=self.dtype)
            sinvhpsit = self.gd.zeros(len(kpt.psit_nG), dtype=self.dtype)
            hamiltonian.apply(kpt.psit_nG, hpsit, self, kpt, calculate_P_ani = True)
            if(td_correction is 'sinvcg'):
                self.overlap.apply_inverse_cg(hpsit, sinvhpsit, self, kpt, calculate_P_ani=True)
            elif(td_correction is 'sinvapr'):
                self.overlap.apply_inverse(hpsit, sinvhpsit, self, kpt, calculate_P_ani=True)
            #print 'sinvhpsit_0_cg - epspsit_0.max', abs(sinvhpsit[0]-eps_psit[0]).max()
            #print 'sinvhpsit_0 - epspsit_0.max', abs(sinvhpsit2[0]-eps_psit[0]).max()
            self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)
            if(td_correction is not None):
                G_axi = self.pt.dict(self.bd.mynbands)
                self.pt.integrate(sinvhpsit, G_axi, kpt.q)
            
            for a, F_niv in F_aniv.items():
                F_niv = F_niv.conj()
                F_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
                FdH1_niv = F_niv.copy()
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                P_ni = kpt.P_ani[a]
                dO_ii = hamiltonian.setups[a].dO_ii
                F_vii = np.dot(np.dot(F_niv.transpose(), P_ni), dH_ii)
                if(td_correction is not None):
                    fP_ni = P_ni * kpt.f_n[:,np.newaxis]
                    G_ni = G_axi[a]
                    nabla_iiv = hamiltonian.setups[a].nabla_iiv
                    F_vii_sinvh_dpt = -np.dot(np.dot(FdH1_niv.transpose(), G_ni), dO_ii)
                    F_vii_sinvh_dphi = -np.dot(nabla_iiv.transpose(2,0,1), np.dot(fP_ni.conj().transpose(), G_ni))
                    F_vii += F_vii_sinvh_dpt + F_vii_sinvh_dphi
                else:
                    F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                    F_vii -= np.dot(np.dot(F_niv.transpose(), P_ni), dO_ii)
                                    
                #F_av_dO[a] += 2 * F_vii_dO.real.trace(0,1,2)
                #F_av_dH[a] += 2 * F_vii_dH.real.trace(0,1,2)
                F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

            # Hack used in delta-scf calculations:
            if hasattr(kpt, 'c_on'):
                assert self.bd.comm.size == 1
                self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q) #XXX again
                d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                dtype=complex)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    d_nn += ne * np.outer(c_n.conj(), c_n)
                for a, F_niv in F_aniv.items():
                    F_niv = F_niv.conj()
                    dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                    Q_ni = np.dot(d_nn, kpt.P_ani[a])
                    F_vii = np.dot(np.dot(F_niv.transpose(), Q_ni), dH_ii)
                    F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                    dO_ii = hamiltonian.setups[a].dO_ii
                    F_vii -= np.dot(np.dot(F_niv.transpose(), Q_ni), dO_ii)
                    F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

        self.bd.comm.sum(F_av, 0)

        if self.bd.comm.rank == 0:
            self.kpt_comm.sum(F_av, 0)

    def estimate_memory(self, mem):
        FDPWWaveFunctions.estimate_memory(self, mem)
        self.kin.estimate_memory(mem.subnode('Kinetic operator'))
