from __future__ import print_function

import sys
import pickle

import numpy as np
from ase.units import Hartree, Bohr
from ase.parallel import paropen
import pylab as p


class Heterostructure:
    def __init__(self, q_points_abs, frequencies, interlayer_distances,
                 chi_monopole, z, drho_monopole, chi_dipole = None, drho_dipole= None):
        
        self.n_layers = len(interlayer_distances) + 1     
        self.q_points_abs = q_points_abs # * Bohr
        self.frequencies = frequencies # / Hartree        
        
        self.chi_monopole = chi_monopole # * Bohr  
        self.drho_monopole = np.array(drho_monopole)
        self.chi_dipole = chi_dipole  #/ Bohr         
        self.drho_dipole = np.array(drho_dipole)
        # grids for induced density
        self.z = z 
        self.interlayer_distances = interlayer_distances / Bohr  
   

    def solve_poisson_1D(self, drho, q, z, Nz=5000, monopole = True, delta = None):
        from scipy.integrate import cumtrapz

        def poisson_integral(drho, q, z, sign = 1):
            temp = cumtrapz(np.exp(- sign * q * z) * drho, z, initial=0)
            return temp 
        
        drho = np.append(np.insert(drho, 0, np.zeros([Nz])), np.zeros([Nz]))
        dz = z[1]-z[0]
        z -= z[len(z)/2] # center arround 0
        z_grid = np.append(np.insert(z,0,np.linspace(z[0]-(Nz+1)*dz,z[0]-dz, Nz)), 
                           np.linspace(z[-1]+dz,z[-1]+(Nz+1)*dz, Nz))
                           
        dphi =  np.exp(q * z_grid) / (2 * q) * poisson_integral(-4 * np.pi * drho, q, z_grid) \
            - np.exp(-q * z_grid) / (2 * q) * poisson_integral(-4 * np.pi * drho, q, z_grid, 
                                                                sign = -1)

        """
        Boundary conditions:
        c1 * e^{q z_inf} + c2 + dphi(z_inf) = 0 
        If dipole: c1 + c2 + dphi(z=0) = 0 
        """
        i_z0 = len(z_grid)/2
        
        if not monopole:
            phi_model = np.pi / (q * delta) * (-np.exp(-q*np.abs(z_grid+delta)) + \
                                                np.exp(-q*np.abs(z_grid-delta)))
            c1 = (dphi[-1] - phi_model[-1])/ np.exp(q * z_grid[-1])
            c2 =  dphi[i_z0]- c1
        else:
            c1 =  (dphi[-1] - 2 * np.pi / q * np.exp(-q * z_grid[-1]))/ np.exp(q * z_grid[-1])
            c2 = 0
        dphi -= c1 *  np.exp(q * z_grid) - c2
        
        return z_grid, dphi
    
    def overlap(self, z_rho, z_phi, drho, dphi, d, plot = False):
        from scipy import interpolate
        z_rho -= z_rho[len(z_rho)/2]
       
        tck = interpolate.splrep(z_phi - d, dphi)
        dphi = interpolate.splev(z_rho, tck, der=0)
        I = np.trapz(drho * dphi, z_rho)
        return I
       
    def CoulombKernel(self, iq, full=True):
        #---------------------------------
        # Different Types of Interaction 
        #---------------------------------
        from scipy.integrate import cumtrapz
        q = self.q_points_abs[iq]
        z = self.z 
        # Monopole generates a monopole
        def v_mm(i, j, d):
            drho_i = self.drho_monopole[i, iq].copy() / self.chi_monopole[i, iq, 0]
            z_grid, dphi_i = self.solve_poisson_1D(drho_i, q, z[i])
            drho_j = self.drho_monopole[j, iq].copy() / self.chi_monopole[j, iq, 0]
            temp = self.overlap(z[j], z_grid, drho_j, dphi_i, d)
            return temp
        
        def v_md(i, j, d):
            drho_i = self.drho_monopole[i, iq].copy() / self.chi_monopole[i, iq, 0]
            z_grid, dphi_i = self.solve_poisson_1D(drho_i, q, z[i])
            drho_j = self.drho_dipole[j, iq].copy() / self.chi_dipole[j, iq, 0]
            temp = self.overlap(z[j], z_grid, drho_j, dphi_i, d)
            return temp

        def v_dm(i, j, d):
            drho_i = self.drho_dipole[i, iq].copy() / self.chi_dipole[i, iq, 0]            
            delta = np.abs(z[i][np.argmax(drho_i)] - z[i][np.argmin(drho_i)])/2.
            z_grid, dphi_i = self.solve_poisson_1D(drho_i, q, z[i], 
                                                   monopole = False, delta = delta)
            drho_j = self.drho_monopole[j, iq].copy() / self.chi_monopole[j, iq, 0]
            temp = self. overlap(z[j], z_grid, drho_j, dphi_i, d)
            return temp

        def v_dd(i, j, d):
            drho_i = self.drho_dipole[i, iq].copy() / self.chi_dipole[i, iq, 0]
            delta = np.abs(z[i][np.argmax(drho_i)] - z[i][np.argmin(drho_i)])/2.
            z_grid, dphi_i = self.solve_poisson_1D(drho_i, q, z[i], 
                                                   monopole = False, delta = delta)
            drho_j = self.drho_dipole[j, iq].copy() / self.chi_dipole[j, iq, 0]
            temp = self.overlap(z[j], z_grid, drho_j, dphi_i, d)
            return temp
           
      
        #---------------------------------
        # Building Distances Matrix
        #---------------------------------

        Nls = self.n_layers
        d_ij = np.zeros((Nls, Nls))

        for i in range(0, Nls):
            for j in range(i + 1, Nls):
                for l in range(i, j):
                    t = j - i - 1
                    d_ij[i, j] = d_ij[i, j] + self.interlayer_distances[t]
                    d_ij[j, i] = d_ij[j, i] + self.interlayer_distances[t]

        #---------------------------------
        # Calculating the Kernel
        #---------------------------------
        if self.chi_dipole is not None:
            kernel_ij = np.zeros((2 * Nls, 2 * Nls), dtype=complex)
            for i in range(0, Nls):
                if full:
                    kernel_ij[2*i, 2*i] = v_mm(i, i, 0)
                    kernel_ij[2*i+1, 2*i+1] = v_dd(i, i, 0)
                if self.n_layers > 1:
                    for j in np.delete(range(0, Nls), i):
                        kernel_ij[2*i, 2*j] = v_mm(i, j, d_ij[i, j])
                        kernel_ij[2*i, 2*j+1] = v_md(i, j, np.sign(j-i) * d_ij[i, j])
                        kernel_ij[2*i+1, 2*j] = v_dm(i, j, np.sign(j-i) * d_ij[i, j])
                        kernel_ij[2*i+1, 2*j+1] = v_dd(i, j, np.sign(j-i) * d_ij[i, j])
        else:
            kernel_ij = np.zeros((Nls, Nls), dtype=complex)
            for i in range(0, Nls):
                if full:
                    kernel_ij[i, i] = v_mm(i, i, 0)
                for j in np.delete(range(0, Nls), i):
                    kernel_ij[i, j] = v_mm(i, j, d_ij[i, j])
       
        return np.real(kernel_ij)

    def get_chi_matrix(self):
        """
        Dyson like equation;
        chi_full = chi_intra + chi_intra V_inter chi_full
            
        """
        Nls = self.n_layers
        q_points_abs = self.q_points_abs
        chi_m_iqw = self.chi_monopole
        chi_d_iqw = self.chi_dipole

        if self.chi_dipole is not None:
            chi_qwij = np.zeros((len(self.q_points_abs),
                                 len(self.frequencies),
                                 2 * Nls, 2 * Nls), dtype=complex)
            for iq in range(len(q_points_abs)):
                kernel_ij = self.CoulombKernel(iq,
                                               full=False) # Diagonal is set to zero
                for iw in range(0, len(self.frequencies)):
                    chi_intra_i = np.insert(chi_d_iqw[:, iq, iw], 
                                            np.arange(len(chi_m_iqw[:, iq, iw])),
                                            chi_m_iqw[:, iq, iw])
                    chi_intra_ij = np.diag(chi_intra_i)
                    chi_qwij[iq, iw, :, :] = np.dot(np.linalg.inv(
                            np.eye(2 * Nls) - np.dot(chi_intra_ij, kernel_ij)), 
                                                    chi_intra_ij)
        else:
            chi_qwij = np.zeros((len(self.q_points_abs),
                                 len(self.frequencies),
                                 Nls, Nls), dtype=complex)
            for iq in range(len(q_points_abs)):
                kernel_ij = self.CoulombKernel(iq, 
                                               full=False) # Diagonal is set to zero
                for iw in range(len(self.frequencies)):
                    chi_intra_i = chi_m_iqw[:, iq, iw]
                    chi_intra_ij = np.diag(chi_intra_i)
                    chi_qwij[iq, iw, :, :] = np.dot(np.linalg.inv(
                            np.eye(Nls) - np.dot(chi_intra_ij, kernel_ij)), 
                                                    chi_intra_ij)
        
        return chi_qwij

    def get_eps_matrix(self):
        Nls = self.n_layers
        chi_qwij = self.get_chi_matrix()
        if self.chi_dipole is not None:
            eps_qwij = np.zeros((len(self.q_points_abs), len(self.frequencies), 
                                 2 * Nls, 2 * Nls), dtype=complex)
        else: 
            eps_qwij = np.zeros((len(self.q_points_abs), len(self.frequencies), 
                                 Nls, Nls), dtype=complex)
        for iq in range(len(self.q_points_abs)):
            kernel_ij = self.CoulombKernel(iq)
            for iw in range(0, len(self.frequencies)):
                eps_qwij[iq, iw, :, :] = np.linalg.inv(\
                    np.eye(kernel_ij.shape[0]) + np.dot(kernel_ij,
                                                        chi_qwij[iq, iw, :, :]))
      
        return eps_qwij
    
    def get_exciton_screened_potential(self, e_distr, h_distr):  
        v_screened_qw = np.zeros((len(self.q_points_abs), len(self.frequencies)))    
        eps_qwij = self.get_eps_matrix() 
        h_distr = h_distr.transpose()

        for iq in range(0, len(self.q_points_abs)):
            kernel_ij = self.CoulombKernel(iq)
            ext_pot = np.dot(kernel_ij, h_distr)
            for iw in range(0, len(self.frequencies)):
                v_screened_qw[iq, iw] = self.q_points_abs[iq] / 2. / np.pi * np.dot(e_distr, np.dot(np.linalg.inv(eps_qwij[iq, iw, :, :]), ext_pot))   
                        
        return 1. / (v_screened_qw)
    
    def get_plasmon_eigenmodes(self):
        eps_qwij = self.get_eps_matrix()
        Nw = len(self.frequencies)
        Nq = len(self.q_points_abs)
        w_w = self.frequencies
        if self.chi_dipole is not None:
            Nd = self.n_layers * 2
            eig = np.zeros([Nq, Nw, self.n_layers * 2], dtype=complex)
            vec = np.zeros([Nq, Nw, self.n_layers * 2, self.n_layers * 2], 
                           dtype=complex)
        else: 
            Nd = self.n_layers
            eig = np.zeros([Nq, Nw, self.n_layers], dtype=complex)
            vec = np.zeros([Nq, Nw, self.n_layers, self.n_layers], 
                           dtype=complex)
        omega0 = [[] for i in range(Nq)]
        eigen0 = np.zeros([Nq, 100])
        for iq in range(Nq):
            m = 0
            eig[iq, 0], vec[iq, 0] = np.linalg.eig(eps_qwij[iq, 0])
            vec_dual = np.linalg.inv(vec[iq, 0])
            for iw in range(1, Nw):
                eig[iq, iw], vec_p = np.linalg.eig(eps_qwij[iq, iw])
                vec_dual_p = np.linalg.inv(vec_p)
                overlap = np.abs(np.dot(vec_dual, vec_p))
                index = list(np.argsort(overlap)[:, -1])
                vec[iq, iw] = vec_p[:, index]
                vec_dual = vec_dual_p[index, :]                 
                eig[iq, iw, :] = eig[iq, iw, index]
                klist = [k for k in range(Nd) if (eig[iq, iw - 1, k] < 0 
                                                  and eig[iq, iw, k] > 0)]
                for k in klist:# Eigenvalue crossing
                    a = np.real((eig[iq, iw, k]-eig[iq, iw-1, k]) / \
                                (w_w[iw]-w_w[iw-1]))
                    # linear interp for crossing point
                    w0 = np.real(-eig[iq, iw-1, k]) / a + w_w[iw-1] 
                    eig0 = a * (w0 - w_w[iw-1]) + eig[iq, iw-1, k]
                    #print('crossing found at w = %1.2f eV'%w0)
                    omega0[iq].append(w0)
                    m += 1
        return eig, vec, np.array(omega0)

"""TOOLS"""

def get_chi_2D(filenames, name=None):
    """Calculate the monopole and dipole contribution to the
    2D susceptibillity \chi_2D, defined as: 
    \chi^M_2D(q, \omega) = \int\int dr dr' \chi(q, \omega, r,r') \\
                        = L \chi_{G=G'=0}(q, \omega)
    \chi^D_2D(q, \omega) = \int\int dr dr' z \chi(q, \omega, r,r') z'
                         = 1/L sum_{G_z,G_z'} z_factor(G_z) chi_{G_z,G_z'} z_factor(G_z'),
    Where z_factor(G_z) =  +/- i e^{+/- i*G_z*z0} (L G_z cos(G_z L/2)-2 sin(G_z L/2))/G_z^2

    input parameters: 
    
    filenames: list of chi_wGG.pckl files for different q 
    name: 'str' for writing output files  
    """

    nq = len(filenames)
    omega_w, pd, chi_wGG = pickle.load(open(filenames[0]))
    r = pd.gd.get_grid_point_coordinates()
    z = r[2,0,0,:]
    L = pd.gd.cell_cv[2, 2] # Length of cell in Bohr
    z0 = L/2. # position of layer
    npw = chi_wGG.shape[1]
    nw = chi_wGG.shape[0]
    q_points_abs = []
    Gvec = pd.G_Qv[pd.Q_qG[0]]
    Glist = []   
    for iG in range(npw): # List of G with Gx,Gy = 0
        if Gvec[iG, 0] == 0 and Gvec[iG, 1] == 0:
            Glist.append(iG)
    chiM_2D_qw = np.zeros([nq, nw], dtype=complex)
    chiD_2D_qw = np.zeros([nq, nw], dtype=complex)
    drho_M_qz = np.zeros([nq, len(z)], dtype=complex) # induced density
    drho_D_qz = np.zeros([nq, len(z)], dtype=complex) # induced dipole density
    for iq in range(nq):
        if not iq == 0:
            omega_w, pd, chi_wGG = pickle.load(open(filenames[iq]))  
        chi_wGG = np.array(chi_wGG)
        q = pd.K_qv
        q_abs = np.linalg.norm(q)        
        q_points_abs.append(q_abs)          
        chiM_2D = np.zeros_like(chi_wGG[:, 0, 0], dtype=complex) 
        chiD_2D = np.zeros_like(chi_wGG[:, 0, 0], dtype=complex)
        drho_M_qz[iq] +=  chi_wGG[0,0,0]
        for iG in Glist[1:]: 
            G_z = Gvec[iG, 2] 
            qGr_R = np.inner(G_z, z.T).T
            # Fourier transform to get induced density, 
            # use static limit so far
            drho_M_qz[iq] += np.exp(1j * qGr_R) * chi_wGG[0,iG,0]
            for iG1 in Glist[1:]:
                G_z1 = Gvec[iG1, 2]
                # integrate with z along both coordinates 
                factor = z_factor(z0, L, G_z)
                factor1 = z_factor(z0, L, G_z1, sign=-1)
                chiD_2D += 1. / L * factor * chi_wGG[:, iG, iG1] * factor1 
                # induced dipole density due to V_ext = z
                drho_D_qz[iq] += 1. / L * np.exp(1j * qGr_R) * chi_wGG[0,iG,iG1] * factor1 
        chiM_2D_qw[iq, :] = L * chi_wGG[:, 0, 0] 
        chiD_2D_qw[iq, :] = chiD_2D

    """ Returns q array, frequency array, chi2D monopole, chi, (Now returned in Bohr)
    """
    pickle.dump((np.array(q_points_abs), omega_w, chiM_2D_qw, chiD_2D_qw, \
                     z, drho_M_qz, drho_D_qz ), open(name + '-chi.pckl', 'w'))     
    return np.array(q_points_abs)/Bohr, omega_w * Hartree, chiM_2D_qw, \
        chiD_2D_qw, z, drho_M_qz, drho_D_qz


# Temporary 
# Should be rewritten!!!
def get_chiM_2D_from_old_DF(filenames_eps, read, qpoints, d=None, write_chi0 = False, name = None):
    #rec_cell = reciprocal_cell*Bohr
    #q_points = np.loadtxt(filename_qpoints) 
    #q_points = np.dot(q_points,rec_cell)
    #Gvec = pickle.load(open(filename_Gvec %0))
    #Gvec = np.dot(Gvec,rec_cell) # the cell has to be in bohr
    from gpaw.response.df0 import DF
    df = DF()
    df.read(read + str(qpoints[0]))
    cell = df.acell_cv
    Gvec = np.dot(df.Gvec_Gc,df.bcell_cv)
    nq = len(filenames_eps)#len(q_points[:,0])
    L = cell[2,2] # Length of cell in Bohr
    d /= Bohr # d in Bohr
    z0 = L/2. # position of layer
    npw = Gvec.shape[0]
    nw = df.Nw
    omega_w = df.w_w#[0.]
    q_points_abs = []
    Glist = []   

    for iG in range(npw): # List of G with Gx,Gy = 0
        if Gvec[iG, 0] == 0 and Gvec[iG, 1] == 0:
            Glist.append(iG)
    epsM_2D_qw = np.zeros([nq, nw], dtype=complex)
    epsD_2D_qw = np.zeros([nq, nw], dtype=complex)
    chiM_2D_qw = np.zeros([nq, nw], dtype=complex)
    chiD_2D_qw = np.zeros([nq, nw], dtype=complex)
    VM_eff_qw = np.zeros([nq, nw], dtype=complex)
    for iq in range(nq):
        df.read(read + str(qpoints[iq]))
        la,la,la,eps_wGG, chi_wGG = pickle.load(open(filenames_eps[iq])) 
        #chi_wGG = pickle.load(open(filenames_chi %iq))  
        #chi_wGG = np.array(chi_wGG)
        eps_inv_wGG = np.zeros_like(eps_wGG, dtype = complex) 
        for iw in range(nw):
            eps_inv_wGG[iw] = np.linalg.inv(eps_wGG[iw])
            eps_inv_wGG[iw] = np.identity(npw)
        del eps_wGG
        q = df.q_c#q_points[iq]
        q_abs = np.linalg.norm(q)        
        q_points_abs.append(q_abs) # return q in Ang            
        epsM_2D_inv = eps_inv_wGG[:, 0, 0]
        epsD_2D_inv = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype = complex)
        chiM_2D = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype = complex) #chi_wGG[:, 0, 0]#
        chiD_2D = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype = complex)
        for iG in Glist[1:]: 
            G_z = Gvec[iG, 2] 
            epsM_2D_inv += 2./d * np.exp(1j*G_z*z0) * np.sin(G_z*d/2.) / G_z * eps_inv_wGG[:, iG, 0]
            
            for iG1 in Glist[1:]:
                G_z1 = Gvec[iG1, 2]
                # intregrate over entire cell for z and z'
                factor1 = z_factor(z0, L, G_z)
                factor2 = z_factor(z0, L, G_z1, sign=-1)
                chiD_2D += 1./L * factor1 * factor2 * chi_wGG[:, iG, iG1]
                # intregrate z over d for epsilon^-1
                #factor1 =  z_factor2(z0, d, G_z) 
                #epsD_2D_inv += 2j / d / L * factor1 * factor2 * eps_inv_wGG[:, iG, iG1]  #average
                #epsD_2D_inv += 1j * G_z * np.exp(1j*G_z*z0) * factor2 * eps_inv_wGG[:, iG, iG1]  #atz0
                factor1 =  z_factor(z0, d, G_z) 
                epsD_2D_inv += 12. / d**3 / L * factor1 * factor2 * eps_inv_wGG[:, iG, iG1]  #kristian
            
        epsM_2D_qw[iq, :] = 1. / epsM_2D_inv
        epsD_2D_qw[iq, :] = 1. / epsD_2D_inv
        chiM_2D_qw[iq, :] = L * chi_wGG[:, 0, 0] #chiM_2D#
        chiD_2D_qw[iq, :] = chiD_2D
        del chi_wGG,  eps_inv_wGG

    # Effective Coulomb interaction in 2D from eps_{2D}^{-1} = 1 + V_{eff} \chi_{2D}
    VM_eff_qw = (1. /epsM_2D_qw - 1) / chiM_2D_qw 
    VD_eff_qw = (1. /epsD_2D_qw - 1) / chiD_2D_qw
    chi0M_2D_qw = (1 - epsM_2D_qw) * 1. / VM_eff_qw  # Chi0 from effective Coulomb 
    chi0D_2D_qw = (1 - epsD_2D_qw) * 1. / VD_eff_qw
    pickle.dump((np.array(q_points_abs), omega_w, VM_eff_qw, VD_eff_qw, 
                 chiM_2D_qw, chiD_2D_qw), open(name + '-chi.pckl', 'w')) 
    pickle.dump((np.array(q_points_abs), omega_w, VM_eff_qw, VD_eff_qw, 
                 chi0M_2D_qw, chi0D_2D_qw, chiM_2D_qw, chiD_2D_qw, 
                 epsM_2D_qw, epsD_2D_qw), open(name + '-2D.pckl', 'w')) 
        
    return np.array(q_points_abs), omega_w, chiM_2D_qw, chiD_2D_qw, VM_eff_qw, VD_eff_qw, epsM_2D_qw, epsD_2D_qw


def z_factor(z0, d, G, sign = 1):
    factor= -1j * sign * np.exp(1j*sign*G*z0)*(d*G*np.cos(G*d/2.) - 2. * np.sin(G*d/2.))/G**2
    return factor

def z_factor2(z0, d, G, sign = 1):
    factor= sign * np.exp(1j*sign*G*z0) * np.sin(G * d / 2.)
    return factor

