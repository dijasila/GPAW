from __future__ import print_function

import os
import sys
import pickle
from math import pi

import numpy as np
from ase.utils import prnt
from ase.units import Hartree, Bohr


class Heterostructure:
    def __init__(self, q_points_abs, frequencies, interlayer_distances,
                 chi_monopole, chi_dipole=None, v_monopole=None,
                 v_dipole=None):
        
        self.n_layers = len(interlayer_distances) + 1     # Number of Layers in the heterostructure
        self.q_points_abs = q_points_abs # * Bohr
        self.frequencies = frequencies # / Hartree        # List of frequencies: they have to be same as the ones used for calculating chi_monopole and chi_dipole
        
        self.chi_monopole = chi_monopole# * Bohr         # List of monopole chi0 in each layer       
        self.chi_dipole = chi_dipole  #/ Bohr             # List of dipole chi0 in each layer
        self.v_monopole = v_monopole #/ Bohr             # Effective 2D coulomb 
        self.v_dipole = v_dipole # / Bohr
        self.interlayer_distances = interlayer_distances / Bohr   # Distances array: element i has to contain the distance between the layer i and i+1
        nq = len(q_points_abs)
        nw = len(frequencies)

    def CoulombKernel(self, iq, full=True):
        
        #---------------------------------
        # Different Types of Interaction 
        #---------------------------------

        q_abs = self.q_points_abs[iq]

        def delta_factor(q, delta=2, sign=1):
            """ 
            delta is a distance parameter to describe the intra-layer surface separation.
            2*delta ~ thickness of layer in Ang. 
            sign is positive/negative for monopole/dipole potential.
            """
            temp = (np.exp(q * delta) + sign * np.exp(-q * delta)) / 2.
            return temp
        
        # Monopole generates a monopole
        def v_mm(i, iq, d, delta=3):
            q = self.q_points_abs[iq]
            delta /= Bohr
            if d == 0: # intra-layer
                if self.v_monopole is not None:
                    temp = self.v_monopole[i, iq, 0]
                else:
                    temp = 2 * np.pi / q * np.exp(-q * delta)
            else:
                temp = 2 * np.pi / q * np.exp(-q * np.abs(d)) * delta_factor(q, delta)
            return temp
        
        # Monopole generates a dipole
        def v_md(i, iq, d, delta=3):
            q = self.q_points_abs[iq]
            delta /= Bohr
            temp = -np.sign(d) * 2 * np.pi * np.exp(-q * np.abs(d)) * \
                delta_factor(q, delta)
            return temp

        # Dipole generates a monopole
        def v_dm(i, iq, d, delta=3):
            q = self.q_points_abs[iq]
            delta /= Bohr  # intra-layer dipole d = 6
            temp = np.sign(d) * 2 * np.pi / q * np.exp(-q * np.abs(d)) * \
                delta_factor(q, delta, sign=-1) / delta # wrong normalization??
            return temp

        # Dipole generates a dipole
        def v_dd(i, iq, d, delta=3): 
            q = self.q_points_abs[iq]
            delta /= Bohr
            if d == 0: 
                if self.v_dipole is not None:
                    temp = self.v_dipole[i, iq, 0]
                else:
                    temp = 2 * np.pi * np.exp(-q * delta) / delta # wrong normalization??
            else: 
                # opposite sign as intra-layer v_dd!
                temp = - 2 * np.pi * np.exp(-q * np.abs(d)) * \
                    delta_factor(q, delta, sign=-1) / delta
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
                    kernel_ij[2*i, 2*i] = v_mm(i, iq, 0)
                    kernel_ij[2*i+1, 2*i+1] = v_dd(i, iq, 0)

                for j in np.delete(range(0, Nls), i):
                    kernel_ij[2*i, 2*j] = v_mm(i, iq, d_ij[i, j])
                    kernel_ij[2*i+1, 2*j] = v_dm(i, iq, np.sign(j-i) * d_ij[i, j])
                    kernel_ij[2*i, 2*j+1] = v_md(i, iq, np.sign(j-i) * d_ij[i, j])
                    kernel_ij[2*i+1, 2*j+1] = v_dd(i, iq, np.sign(j-i) * d_ij[i, j])
        else:
            kernel_ij = np.zeros((Nls, Nls), dtype=complex)
            for i in range(0, Nls):
                if full:
                    kernel_ij[i, i] = v_mm(i, iq, 0)
                for j in np.delete(range(0, Nls), i):
                    kernel_ij[i, j] = v_mm(i, iq, d_ij[i, j])

        return kernel_ij

    def get_chi_matrix(self):
        """Dyson like equation.
        
        ::
            
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
                    print('crossing found at w = %1.2f eV'%w0)
                    omega0[iq].append(w0)
                    #omega0[iq, m] = w0
                    m += 1
                    #eigen0 = np.append(eigen0, eig0)
        return eig, vec, np.array(omega0)

"""TOOLS"""

def get_chiM_2D(filenames, filenames_chi, d=5, name=None):
    nq = len(filenames)
    omega_w, pd, eps_wGG = pickle.load(open(filenames[0])) 
    omega_w, pd, chi_wGG = pickle.load(open(filenames_chi[0]))
    L= pd.gd.cell_cv[2, 2] # Length of cell in Bohr
    d /= Bohr # d in Bohr
    z0 = L/2. # position of layer
    npw = eps_wGG.shape[1]
    nw = eps_wGG.shape[0]
    q_points_abs = []
    Gvec = pd.get_reciprocal_vectors()
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
        if not iq == 0:
            omega_w, pd, eps_wGG = pickle.load(open(filenames[iq]))  
            omega_w, pd, chi_wGG = pickle.load(open(filenames_chi[iq]))  
        chi_wGG = np.array(chi_wGG)
        eps_inv_wGG = np.zeros_like(eps_wGG, dtype=complex)
        for iw in range(nw):
            eps_inv_wGG[iw] = np.linalg.inv(eps_wGG[iw])
        q = pd.K_qv
        q_abs = np.linalg.norm(q)        
        q_points_abs.append(q_abs)          
        epsM_2D_inv = eps_inv_wGG[:, 0, 0]
        epsD_2D_inv = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype=complex)
        chiM_2D = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype=complex) #chi_wGG[:, 0, 0]#
        chiD_2D = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype=complex)
        for iG in Glist[1:]: 
            G_z = Gvec[iG, 2] 
            epsM_2D_inv += 2./d * np.exp(1j*G_z*z0) * np.sin(G_z*d/2.) / G_z * \
            eps_inv_wGG[:, iG, 0]
            for iG1 in Glist[1:]:
                G_z1 = Gvec[iG1, 2]
                # intregrate over entire cell for z and z'
                factor1 = z_factor(z0, L, G_z)
                factor2 = z_factor(z0, L, G_z1, sign=-1)
                chiD_2D += 1./L * factor1 * factor2 * chi_wGG[:, iG, iG1]
                # intregrate z over d for epsilon^-1
                #factor1 =  z_factor2(z0, d, G_z) 
                #epsD_2D_inv += 2j / d / L * factor1 * factor2 * eps_inv_wGG[:, iG, iG1] #average
                #epsD_2D_inv += 1j * G_z * np.exp(1j*G_z*z0) * factor2 * eps_inv_wGG[:, iG, iG1]  #atz0
                factor1 = z_factor(z0, d, G_z)
                epsD_2D_inv += 12. / d**3 / L * factor1 * factor2 * eps_inv_wGG[:, iG, iG1]  #kristian
        epsM_2D_qw[iq, :] = 1. / epsM_2D_inv
        epsD_2D_qw[iq, :] = 1. / epsD_2D_inv
        chiM_2D_qw[iq, :] = L * chi_wGG[:, 0, 0] #chiM_2D#
        chiD_2D_qw[iq, :] = chiD_2D
        
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
    
    return np.array(q_points_abs)/Bohr, omega_w * Hartree, VM_eff_qw*Bohr, \
        VD_eff_qw, chi0M_2D_qw/Bohr, chi0D_2D_qw, chiM_2D_qw/Bohr, chiD_2D_qw, \
        epsM_2D_qw, epsD_2D_qw


# Temporary
def get_chiM_2D_from_old_DF(filenames_eps, filenames_chi, filename_qpoints, filename_Gvec,reciprocal_cell, d=None, write_chi0 = False, name = None):
    rec_cell = reciprocal_cell*Bohr
    q_points = np.loadtxt(filename_qpoints) 
    q_points = np.dot(q_points,rec_cell)
    Gvec = pickle.load(open(filename_Gvec %0))
    Gvec = np.dot(Gvec,rec_cell) # the cell has to be in bohr
    nq = len(q_points[:,0])
    L= 2*np.pi/rec_cell[2,2] # Length of cell in Bohr
    d /= Bohr # d in Bohr
    z0 = L/2. # position of layer
    npw = Gvec.shape[0]
    nw = 1
    omega_w = [0.]
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
        eps_wGG = pickle.load(open(filenames_eps %iq))  
        chi_wGG = pickle.load(open(filenames_chi %iq))  
        chi_wGG = np.array(chi_wGG)
        eps_inv_wGG = np.zeros_like(eps_wGG, dtype = complex) 
        for iw in range(nw):
            eps_inv_wGG[iw] = np.linalg.inv(eps_wGG[iw])
#            eps_inv_wGG[iw] = np.identity(npw)
        q = q_points[iq]
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
                factor1 =  z_factor2(z0, d, G_z) 
                epsD_2D_inv += 2j / d / L * factor1 * factor2 * eps_inv_wGG[:, iG, iG1]  #average
                #epsD_2D_inv += 1j * G_z * np.exp(1j*G_z*z0) * factor2 * eps_inv_wGG[:, iG, iG1]  #atz0
                #factor1 =  z_factor(z0, d, G_z) 
                #epsD_2D_inv += 12. / d**3 / L * factor1 * factor2 * eps_inv_wGG[:, iG, iG1]  #kristian
        epsM_2D_qw[iq, :] = 1. / epsM_2D_inv
        epsD_2D_qw[iq, :] = 1. / epsD_2D_inv
        chiM_2D_qw[iq, :] = L * chi_wGG[:, 0, 0] #chiM_2D#
        chiD_2D_qw[iq, :] = chiD_2D

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
        
    return np.array(q_points_abs), chiM_2D_qw, chiD_2D_qw, VM_eff_qw, VD_eff_qw, epsM_2D_qw, epsD_2D_qw


def z_factor(z0, d, G, sign = 1):
    factor= -1j*sign*np.exp(1j*sign*G*z0)*(d*G*np.cos(G*d/2.)-2.*np.sin(G*d/2.))/G**2
    return factor

def z_factor2(z0, d, G, sign = 1):
    factor= sign * np.exp(1j*sign*G*z0) * np.sin(G * d / 2.)
    return factor
