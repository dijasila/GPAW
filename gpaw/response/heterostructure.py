from __future__ import print_function

import os
import sys
import pickle
from math import pi

import numpy as np
from ase.utils import prnt
from ase.units import Hartree, Bohr


class Heterostructure:

    def __init__(self, q_points_abs, frequencies, interlayer_distances, chi_monopole, chi_dipole=None):
        
        self.n_layers = len(interlayer_distances)+1                       # Number of Layers in the heterostructure
        self.q_points_abs = q_points_abs * Bohr                           # List of the absolute values of q_points: they have to be same as the ones used for calculating chi_monopole and chi_dipole
        self.frequencies = frequencies / Hartree                          # List of frequencies: they have to be same as the ones used for calculating chi_monopole and chi_dipole
        self.chi_monopole = chi_monopole * Bohr    #check units          # List of monopole chi0 in each layer
        if chi_dipole is not None:
            self.chi_dipole = chi_dipole / Bohr        #check units          # List of dipole chi0 in each layer
        else: 
            self.chi_dipole = chi_dipole 
        self.interlayer_distances = interlayer_distances / Bohr          # Distances array: element i has to contain the distance between the layer i and i+1



    def CoulombKernel(self, q_abs):
 
        #---------------------------------
        # Different Types of Interaction      
        #---------------------------------

        def v_mm(q,d):                                                        # Monopole generates a monopole
            temp = 2*np.pi*np.exp(-q*d)/(q+1e-08)
            return temp

        def v_dm(q,d):                                                        # Dipole generates a monopole
            temp = -2.*np.pi*np.sign(d)*np.exp(-q*np.abs(d))
            return temp

        def v_md(q,d):                                                        # Monopole generates a dipole
            temp = -2.*np.pi*np.sign(d)*np.exp(-q*np.abs(d))
            return temp

        def v_dd(q,d):                                                        # Dipole generates a dipole
            temp = 2.*np.pi*q*np.exp(-q*np.abs(d))
            return temp

        #---------------------------------
        # Building Distances Matrix
        #---------------------------------

        Nls = self.n_layers
        d_ij = np.zeros((Nls,Nls))

        for i in range(0,Nls):
            for j in range(i+1,Nls):
                for l in range(i,j):
                    t = j-i-1
                    d_ij[i,j] = d_ij[i,j]+self.interlayer_distances[t]
                    d_ij[j,i] = d_ij[j,i]+self.interlayer_distances[t]

        #---------------------------------
        # Calculating the Kernel
        #---------------------------------

        if self.chi_dipole is not None:
            kernel_ij = np.zeros((2*Nls,2*Nls))

            for i in range(0,Nls):
                for j in range(0,Nls):
                    if i==j:
                        kernel_ij[2*i,2*j] = v_mm(q_abs,d_ij[i,j])
                        kernel_ij[2*i+1,2*i+1] = v_dd(q_abs,d_ij[i,j])
                    else:
                        kernel_ij[2*i,2*j] = v_mm(q_abs,d_ij[i,j])
                        kernel_ij[2*i+1,2*j] =  v_dm(q_abs,np.sign(j-i)*d_ij[i,j])
                        kernel_ij[2*i,2*j+1] = v_md(q_abs,np.sign(j-i)*d_ij[i,j])
                        kernel_ij[2*i+1,2*j+1] = v_dd(q_abs,d_ij[i,j])

        else:
            kernel_ij = np.zeros((Nls,Nls))
            for i in range(0,Nls):
                for j in range(0,Nls):
                    kernel_ij[i,j] = v_mm(q_abs,d_ij[i,j])

        return kernel_ij
    
    def get_eps_matrix(self):
        Nls = self.n_layers
        q_points_abs = self.q_points_abs
        chi_m_iqw = self.chi_monopole   # changed to chi_m_iqw from chi_m_qwi (Easier to make from seperate calculations)
        if self.chi_dipole is not None:       
            eps_qwij = np.zeros((len(self.q_points_abs),len(self.frequencies),2*Nls,2*Nls), dtype = 'complex')
            chi_d_iqw = self.chi_dipole
       
            for iq in range(len(q_points_abs)):
                kernel_ij = self.CoulombKernel(q_points_abs[iq])
                for iw in range(0,len(self.frequencies)):
                    chi_tot_i = np.insert(chi_d_iqw[:,iq,iw],np.arange(len(chi_m_iqw[:,iq,iw])),chi_m_iqw[:,iq,iw])
                    chi_tot_ij = np.diag(chi_tot_i)
                    eps_qwij[iq,iw,:,:] = np.eye(2*Nls,2*Nls)-np.dot(kernel_ij,chi_tot_ij)

        else:
            eps_qwij = np.zeros((len(q_points_abs),len(self.frequencies),Nls,Nls), dtype = 'complex')
            
            for iq in range(len(q_points_abs)):
                kernel_ij = self.CoulombKernel(q_points_abs[iq])
                for iw in range(len(self.frequencies)):
                    chi_m_ij = np.diag(chi_m_iqw[:,iq,iw])
                    eps_qwij[iq,iw,:,:] = np.eye(Nls,Nls)-np.dot(kernel_ij,chi_m_ij)

        return eps_qwij
            
    

    def get_exciton_screened_potential(self,e_distr,h_distr):  
        v_screened_qw = np.zeros((len(self.q_points_abs),len(self.frequencies)))    
        eps_qwij = self.get_eps_matrix() 
        h_distr = h_distr.transpose()

        for iq in range(0,len(self.q_points_abs)):
            kernel_ij = self.CoulombKernel(self.q_points_abs[iq])
            ext_pot = np.dot(kernel_ij,h_distr)
            for iw in range(0,len(self.frequencies)):
                v_screened_qw[iq,iw] = self.q_points_abs[iq]/2./np.pi*np.dot(e_distr,np.dot(np.linalg.inv(eps_qwij[iq,iw,:,:]),ext_pot))   
                        
        return 1./(v_screened_qw)



"""TOOLS"""

def get_chi0M_multipole(filenames, static_limit = False, dipole = True, name = 'chi0M.pckl'):
    omega_w, pd, chi0_wGG, chi0_wxvG, chi0_wvv = pickle.load(open(filenames[0]))
    nq = len(filenames)
    nw = len(omega_w)
    q_points_abs = []
    chi0M_monopole_qw = np.zeros([nq, nw], dtype=complex)
    chi0M_dipole_qw = np.zeros([nq,nw], dtype=complex)
      
    L= pd.gd.cell_cv[2,2]
    npw =  chi0_wGG.shape[1]
    
    Gvec = pd.G_Qv[pd.Q_qG[0]]
    Glist = []
    
    for iG in range(npw):
        if Gvec[iG, 2] == 0:
            Glist.append(iG)
    npw2D = len(Glist)

    for iq in range(nq):
        if not iq == 0:
            omega_w, pd, chi0_wGG, chi0_wxvG, chi0_wvv = pickle.load(open(filenames[iq]))    
        q = pd.K_qv
        q_abs = np.linalg.norm(q)
        q_points_abs.append(q_abs)
        qG2D = Gvec[Glist] + q 
        
        # ---------------------------------------
        # Coulomb kernel 2D Monopole and Dipole
        # ---------------------------------------
        kernel2D_monopole_GG = np.zeros((npw2D, npw2D), dtype=complex)
        kernel2D_dipole_GG = np.zeros((npw2D, npw2D), dtype=complex)
        
        for iG in range(npw2D):
            qG = qG2D[iG] #= np.dot(q_points[iq] + Gvec2D_Gc[iG], rec_cell)
            qG_abs = np.linalg.norm(qG)  
            kernel2D_monopole_GG[iG,iG] = 2. * np.pi / qG_abs
            kernel2D_dipole_GG[iG,iG] = 2. * np.pi * qG_abs           
       
        # ---- Chi0_monopole ----#
        chi02D_monopole_wGG = np.zeros((nw, npw2D,npw2D), dtype=complex)
        iGtemp=0
        for iG in Glist:
            iGtemp1=0
            for iG1 in Glist:
                chi02D_monopole_wGG[:,iGtemp,iGtemp1] = L*chi0_wGG[:,iG,iG1]
                iGtemp1 += 1
            iGtemp += 1
     
        # Including local field effects
        for iw in range(nw):
            eps2D_monopole_GG = np.eye(npw2D,npw2D) - np.dot(kernel2D_monopole_GG,chi02D_monopole_wGG[iw])
            eps2D_monopole_inv_GG = np.linalg.inv(eps2D_monopole_GG)
            chi0M_monopole_qw[iq,iw] = q_abs/2./np.pi*(1.-1./eps2D_monopole_inv_GG[0,0])    

        # ---------------------------------------
        # Chi0M Dipole
        # ---------------------------------------
        chi02D_dipole_wGG = np.zeros((nw,npw2D,npw2D), dtype=complex)

        iGtemp = 0
        for iG in range(0,npw):  
            iGtemp1 = 0        
            for iG1 in range(0,npw):
                if not (Gvec[iG,2] ==0 or Gvec[iG1,2] ==0): # exlude G_perp = 0 in sum??? 
                    G_perp = 2*np.pi*Gvec[iG,2]/L#+10e-8  Divergence for G_perp=0 will dominate chi_dipole!
                    G1_perp = 2*np.pi*Gvec[iG1,2]/L#+10e-8
                    factor1 = 2.*(L/2.+1j/G_perp)*np.exp(1j*G_perp*L/2.)*np.sin(G_perp*L/2.)/G_perp - 2j*L/2.*np.exp(1j*G_perp*L/2.)*np.cos(G_perp*L/2.)/G_perp
                    factor2 = 2.*(L/2.-1j/G1_perp)*np.exp(-1j*G1_perp*L/2.)*np.sin(G1_perp*L/2.)/G1_perp + 2j*L/2.*np.exp(-1j*G1_perp*L/2.)*np.cos(G_perp*L/2.)/G_perp
                    chi02D_dipole_wGG[:,iGtemp,iGtemp1] += 1./L*factor1*factor2*chi0_wGG[:,iG,iG1]          
                if iG1 != npw-1 and Gvec[iG1+1,2] == 0:
                    iGtemp1 += 1
            if iG != npw-1 and Gvec[iG+1,2] == 0:
                iGtemp += 1
    
        for iw in range(nw):
            eps2D_dipole_GG = np.eye(npw2D,npw2D) - np.dot(kernel2D_dipole_GG,chi02D_dipole_wGG[iw])
            eps2D_dipole_inv_GG = np.linalg.inv(eps2D_dipole_GG)
            chi0M_dipole_qw[iq,iw] = 1./q_abs/2./np.pi*(1.-1./eps2D_dipole_inv_GG[0,0])
    
    
    pickle.dump((q_points_abs, omega_w*Hartree, chi0M_monopole_qw, chi0M_monopole_qw), open(name, 'w'))
    return q_points_abs, omega_w*Hartree, chi0M_monopole_qw, chi0M_dipole_qw













































        
