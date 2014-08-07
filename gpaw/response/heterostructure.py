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
                 chi_monopole, chi_dipole=None):
        
        """ should we keep in units of Ang and eV ?  """
        self.n_layers = len(interlayer_distances) + 1     # Number of Layers in the heterostructure
        self.q_points_abs = q_points_abs # * Bohr 
        self.frequencies = frequencies # / Hartree        # List of frequencies: they have to be same as the ones used for calculating chi_monopole and chi_dipole
        self.chi_monopole = chi_monopole # * Bohr         # List of monopole chi0 in each layer
        if chi_dipole is not None:
            self.chi_dipole = chi_dipole # / Bohr         # List of dipole chi0 in each layer
        else: 
            self.chi_dipole = chi_dipole
        self.interlayer_distances = interlayer_distances # / Bohr   # Distances array: element i has to contain the distance between the layer i and i+1

    def CoulombKernel(self, q_abs):
 
        #---------------------------------
        # Different Types of Interaction 
        #---------------------------------
        
        # Monopole generates a monopole
        def v_mm(q, d):                               
            temp = 2 * np.pi * np.exp(-q * np.abs(d)) / (q + 1e-08)
            return temp

        # Dipole generates a monopole
        def v_dm(q, d):
            temp = -2. * np.pi * np.sign(d) * np.exp(-q * np.abs(d))
            return temp

        # Monopole generates a dipole
        def v_md(q, d):
            temp = 2. * np.pi * np.sign(d) * np.exp(-q * np.abs(d))
            return temp

        # Dipole generates a dipole
        def v_dd(q, d):
            temp = 2. * np.pi * q * np.exp(-q * np.abs(d))
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
            kernel_ij = np.zeros((2 * Nls, 2 * Nls))

            for i in range(0, Nls):
                for j in range(0, Nls):
                    if i == j:
                        kernel_ij[2*i, 2*j] = v_mm(q_abs, d_ij[i, j])
                        kernel_ij[2*i+1, 2*i+1] = v_dd(q_abs, d_ij[i, j])
                    else:
                        kernel_ij[2*i, 2*j] = v_mm(q_abs, d_ij[i, j])
                        kernel_ij[2*i+1, 2*j] = v_dm(q_abs, np.sign(j-i) * d_ij[i, j])
                        kernel_ij[2*i, 2*j+1] = v_md(q_abs, np.sign(j-i) * d_ij[i, j])
                        kernel_ij[2*i+1, 2*j+1] = v_dd(q_abs, d_ij[i, j])
       
        else:
            kernel_ij = np.zeros((Nls, Nls))
            for i in range(0, Nls):
                for j in range(0, Nls):
                    kernel_ij[i, j] = v_mm(q_abs, d_ij[i, j])
        return kernel_ij
    
    def get_eps_matrix(self):
        Nls = self.n_layers
        q_points_abs = self.q_points_abs
        chi_m_iqw = self.chi_monopole   # changed to chi_m_iqw from chi_m_qwi (Easier to make from seperate calculations)
        if self.chi_dipole is not None:
            eps_qwij = np.zeros((len(self.q_points_abs), len(self.frequencies), 2*Nls, 2*Nls), dtype = 'complex')
            chi_d_iqw = self.chi_dipole
       
            for iq in range(len(q_points_abs)):
                kernel_ij = self.CoulombKernel(q_points_abs[iq])
                for iw in range(0, len(self.frequencies)):
                    chi_tot_i = np.insert(chi_d_iqw[:, iq, iw], np.arange(len(chi_m_iqw[:, iq, iw])), chi_m_iqw[:, iq, iw])
                    chi_tot_ij = np.diag(chi_tot_i)
                    eps_qwij[iq, iw, :, :] = np.eye(2*Nls, 2*Nls) - np.dot(kernel_ij, chi_tot_ij)

        else:
            eps_qwij = np.zeros((len(q_points_abs), len(self.frequencies), Nls, Nls), dtype = 'complex')
            
            for iq in range(len(q_points_abs)):
                kernel_ij = self.CoulombKernel(q_points_abs[iq])
                for iw in range(len(self.frequencies)):
                    chi_m_ij = np.diag(chi_m_iqw[:, iq, iw])
                    eps_qwij[iq, iw, :, :] = np.eye(Nls, Nls) - np.dot(kernel_ij, chi_m_ij)

        return eps_qwij
    
    def get_exciton_screened_potential(self, e_distr, h_distr):  
        v_screened_qw = np.zeros((len(self.q_points_abs), len(self.frequencies)))    
        eps_qwij = self.get_eps_matrix() 
        h_distr = h_distr.transpose()

        for iq in range(0, len(self.q_points_abs)):
            kernel_ij = self.CoulombKernel(self.q_points_abs[iq])
            ext_pot = np.dot(kernel_ij, h_distr)
            for iw in range(0, len(self.frequencies)):
                v_screened_qw[iq, iw] = self.q_points_abs[iq] / 2. / np.pi * 
                np.dot(e_distr, np.dot(np.linalg.inv(eps_qwij[iq, iw, :, :]), ext_pot))   
                        
        return 1. / (v_screened_qw)
    
    def get_plasmon_eigenmodes(self):
        eps_qwij = self.get_eps_matrix()
        Nw = len(self.frequencies)
        Nq = len(self.q_points_abs)
        w_w = self.frequencies
        if self.chi_dipole is not None:
            Nd = self.n_layers * 2
            eig = np.zeros([Nq, Nw, self.n_layers * 2], dtype = 'complex')
            vec = np.zeros([Nq, Nw, self.n_layers * 2, self.n_layers * 2], dtype = 'complex')
        else: 
            Nd = self.n_layers
            eig = np.zeros([Nq, Nw, self.n_layers], dtype = 'complex')
            vec = np.zeros([Nq, Nw, self.n_layers, self.n_layers], dtype = 'complex')
        omega0 = np.zeros([Nq, Nd])
        eigen0 = np.zeros([Nq, Nd])
        for iq in range(Nq):
            m = 0
            eig[iq, 0], vec[iq, 0] = np.linalg.eig(eps_qwij[iq, 0])
            vec_dual = np.linalg.inv(vec[iq, 0])
            for iw in range(1, Nw):
                eig[iq, iw], vec_p = np.linalg.eig(eps_qwij[iq, iw])
                vec_dual_p = np.linalg.inv(vec_p)
                overlap = np.abs(np.dot(vec_dual, vec_p))
                index = list(np.argsort(overlap)[:, -1])
                """
                if len(np.unique(index)) < Nd: # add missing indices
                    print('ERROR')
                    addlist = []
                    removelist = []
                    for j in range(Nd):
                        if index.count(j) < 1:
                            addlist.append(j)
                        if index.count(j) > 1:
                            for l in range(1,index.count(j)): 
                                removelist.append(np.argwhere(np.array(index) == j)[l]) 
                    for j in range(len(addlist)):
                        index[removelist[j]]=addlist[j]
                """
                vec[iq, iw] = vec_p[:, index]
                vec_dual = vec_dual_p[index, :]                 
                eig[iq, iw, :] = eig[iq, iw, index]
                
                for k in [k for k in range(Nd) if (eig[iq, iw - 1, k] < 0 and eig[iq, iw, k] > 0)]:# Eigenvalue crossing
                    a = np.real((eig[iq, iw, k]-eig[iq, iw-1, k]) / (w_w[iw]-w_w[iw-1]))
                    w0 = np.real(-eig[iq, iw-1, k]) / a + w_w[iw-1]  # linear interp for crossing point
                    eig0 = a * (w0 - w_w[iw-1]) + eig[iq, iw-1, k]
                    print('crossing found at w = %1.2f eV'%w0)
                    omega0[iq, m] = w0
                    m += 1
                    #eigen0 = np.append(eigen0, eig0)
        return eig, vec, omega0

"""TOOLS"""


def get_epsilonM_2D(filenames, d=None, write_chi0 = False, name = None):
    nq = len(filenames)
    omega_w, pd, eps_wGG = pickle.load(open(filenames[0]))  
    L= pd.gd.cell_cv[2, 2] # Length of cell in Bohr
    d /= Bohr
    z0 = L/2.
    npw = eps_wGG.shape[1]
    nw = eps_wGG.shape[0]
    q_points_abs = []
    Gvec = pd.G_Qv[pd.Q_qG[0]]
    Glist = []   
    for iG in range(npw):
        if Gvec[iG, 0] == 0 and Gvec[iG, 1] == 0:
            Glist.append(iG)
    
    epsM_2D_qw = np.zeros([nq, nw], dtype=complex)
    epsD_2D_qw = np.zeros([nq, nw], dtype=complex)
    for iq in range(nq):
        if not iq == 0:
            omega_w, pd, eps_wGG = pickle.load(open(filenames[iq]))  
        q = pd.K_qv
        q_abs = np.linalg.norm(q)        
        q_points_abs.append(q_abs / Bohr) # return q in Ang          
        eps_inv_wGG = np.zeros_like(eps_wGG, dtype = complex)
        for iw in range(nw):
            eps_inv_wGG[iw] = np.linalg.inv(eps_wGG[iw])
        epsM_2D_inv = eps_inv_wGG[:, 0, 0]
        epsD_2D_inv = np.zeros_like(eps_inv_wGG[:, 0, 0], dtype = 'complex')
        for iG in Glist[1:]: 
            G_z = Gvec[iG, 2] 
            epsM_2D_inv += 2./d * np.exp(1j*G_z*z0) * np.sin(G_z*d/2.) / G_z *eps_inv_wGG[:, iG, 0]
            for iG1 in Glist[1:]:
                G_z1 = Gvec[iG1, 2]  
                # epsilon dipole - intregrate from -d/2 to d/2 along z, and over the entire cell for z
                factor1 = 2.*(z0+1j/G_z) * np.exp(1j*G_z*z0) * np.sin(G_z*d/2.) / G_z -
                2j * d/2. * np.exp(1j*G_z*z0) * np.cos(G_z*d/2.) / G_z
                factor2 = 2.*(z0-1j/G_z1) * np.exp(-1j*G_z1*z0) * np.sin(G_z1*z0) / G_z1 + 
                2j * z0 * np.exp(-1j*G_z1*z0) * np.cos(G_z1*z0) / G_z1
                #epsD_2D_inv += 1./(L**2*d**2)*factor1*factor2*eps_inv_wGG[:,iG,iG1]
                epsD_2D_inv += 12. / (d**3 * L) * factor1 * factor2 * eps_inv_wGG[:, iG, iG1]

        epsM_2D_qw[iq, :] = 1. / epsM_2D_inv
        epsD_2D_qw[iq, :] = 1. / epsD_2D_inv
    
    V_2D = np.ones([nq, nw])
    V_2D_d = np.ones([nq, nw]) # dipole coulomb
    for n in range(nq):
        V_2D[n, :] = 2. * np.pi / q_points_abs[n]
        V_2D_d[n, :] = 2. * np.pi * q_points_abs[n]  # Dipole kernel
    chi0M_2D_qw = 1. / V_2D * (1 - epsM_2D_qw) # chi0 in 1/Ang
    chi0D_2D_qw = 1. / V_2D_d * (1 - epsD_2D_qw)
    if write_chi0:
        pickle.dump((q_points_abs, omega_w * Hartree, chi0M_2D_qw, chi0D_2D_qw), open(name + '-chi0.pckl', 'w'))    
    return q_points_abs, omega_w * Hartree, epsM_2D_qw, epsD_2D_qw, chi0M_2D_qw, chi0D_2D_qw
