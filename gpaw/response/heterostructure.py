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
        if chi_dipole:
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
                    d_ij[i,j] = d_ij[i,j]+inter_dist[t]
                    d_ij[j,i] = d_ij[j,i]+inter_dist[t]

        #---------------------------------
        # Calculating the Kernel
        #---------------------------------

        if self.chi_dipole:
            kernel_ij = np.zeros((2*Nls,2*Nls))

            for i in range(0,Nls):
                for j in range(0,Nls):
                    if i==j:
                        kernel_ij[2*i,2*j] = v_mm(q_abs,d_ij[i,j])
                        kernel_ij[2*i+1,2*i+1] = v_dd(q_abs,d_ij[i,j])
                    else:
                        kernel_ij[2*i,2*j] = v_mm(q_abs,d_ij[i,j])
                        kernel_ij[2*i+1,2*j] = v_dm(q_abs,np.sign(j-i)*d_ij[i,j])
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
        chi_m_qwi = self.chi_monopole   
    
        if self.chi_dipole:       
            eps_qwij = np.zeros((len(self.q_points_abs),len(self.frequencies),2*Nls,2*Nls))
            chi_d_qwi = self.chi_dipole
       
            for iq in range(0,len(q_points_abs)):
                kernel_ij = self.CoulombKernel(q_points_abs[iq])
                for iw in range(0,len(self.frequencies)):
                    chi_tot_qwi = np.insert(chi_d_qwi[iq,iw,:],np.arange(len(chi_m_qwi[iq,iw,:])))
                    chi_tot_qwij = np.diag(chi_tot_qwi)
                    eps_qwij[iq,iw,:,:] = np.eye(Nls,Nls)-np.dot(kernel_ij,chi_tot_ij)

        else:
            eps_qwij = np.zeros((len(q_points_abs),len(self.frequencies),Nls,Nls))
            
            for iq in range(0,len(q_points_abs)):
                kernel_ij = self.CoulombKernel(q_points_abs[iq])
                for iw in range(0,len(self.frequencies)):
                    chi_m_ij = np.diag(chi_m_qwi[iq,iw,:])
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
                v_screened_qw[iq,iw] = np.dot(e_distr,np.dot(np.linalg.inv(eps_qwij[iq,iw,:,:]),ext_pot))   
                        
        return 1./v_screened_qw




























        
