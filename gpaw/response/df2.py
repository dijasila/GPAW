import os
import sys
import pickle

import numpy as np
from ase.utils import prnt
from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw.response.chi0 import Chi0


class DielectricFunction:
    """This class defines dielectric function related physical quantities."""
    def __init__(self, calc=None, name=None, omega_w=None, ecut=50,
                 eta=0.2, ftol=1e-6,
                 world=mpi.world, txt=sys.stdout):
        if omega_w is None:
            omega_w = np.linspace(0, 20, 51, end=True)
            
        self.chi0 = Chi0(calc, omega_w, ecut=ecut, eta=eta, ftol=ftol,
                         world=world, txt=txt, real_space_derivatives=not True)
        
        self.name = name

    def calculate_chi0(self, q_c):
        if self.name:
            kd = self.chi0.calc.wfs.kd
            name = self.name + '%+d%+d%+d.pckl' % tuple((q_c * kd.N_c).round())
            if os.path.isfile(name):
                try:
                    G_G, chi0_wGG, chi0_wxvG = pickle.load(open(name))
                except EOFError:
                    pass
                else:
                    return G_G, chi0_wGG, chi0_wxvG
        pd, chi0_wGG, chi0_wxvG = self.chi0.calculate(q_c)
        G_G = pd.G2_qG[0]**0.5  # |G+q|

        if self.name:
            pickle.dump((G_G, chi0_wGG, chi0_wxvG), open(name, 'wb'),
                        pickle.HIGHEST_PROTOCOL)
        return G_G, chi0_wGG, chi0_wxvG
        
    def get_dielectric_matrix(self, xc='RPA', q_c=[0, 0, 0],
                                    direction='x'):
        G_G, chi0_wGG, chi0_wxvG = self.calculate_chi0(q_c)

        if G_G[0] == 0.0:
            G_G[0] = 1.0
            if isinstance(direction, str):
                d_v = {'x': [1, 0, 0],
                       'y': [0, 1, 0],
                       'z': [0, 0, 1]}[direction]
            else:
                d_v = direction
            d_v = np.asarray(d_v) / np.linalg.norm(d_v)
            chi0_wGG[:, 0, 0] = np.dot(chi0_wxvG[:, 0, :, 0], d_v**2)
            chi0_wGG[:, 0] = np.dot(d_v, chi0_wxvG[:, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0_wxvG[:, 1])
            print d_v
        
        print chi0_wGG[0,:2,:2]
        assert xc == 'RPA'

        nG = len(G_G)
        for chi0_GG in chi0_wGG:
            e_GG = np.eye(nG) - 4 * np.pi * chi0_GG / G_G / G_G[:, np.newaxis]
            chi0_GG[:] = e_GG
        return chi0_wGG

    def get_dielectric_function(self, xc='RPA', q_c=[0, 0, 0],
                                      direction='x', filename='df.csv'):
        """Calculate the dielectric function.

        Returns dielectric function without and with local field correction:
 
        df_NLFC_w, df_LFC_w = DielectricFunction.get_dielectric_function()
        """
        e_wGG = self.get_dielectric_matrix(xc, q_c, direction)
        df_NLFC_w = np.zeros(len(e_wGG), dtype=complex)
        df_LFC_w = np.zeros(len(e_wGG), dtype=complex)
        if filename is not None:
            fd = open(filename, 'w')
        for w, e_GG in enumerate(e_wGG):
            df_NLFC_w[w] = e_GG[0, 0]
            df_LFC_w[w] = 1 / np.linalg.inv(e_GG)[0, 0]
            if filename is not None:
                prnt('%.3f, %.3f, %.3f, %.3f, %.3f' %
                     (self.chi0.omega_w[w]*Hartree,
                      df_NLFC_w[w].real, df_NLFC_w[w].imag,
                      df_LFC_w[w].real, df_LFC_w[w].imag), file=fd)
        return df_NLFC_w, df_LFC_w


    def get_macroscopic_dielectric_constant(self, xc='RPA'):
        """Calculate macroscopic dielectric constant. Returns eM_NLFC and eM_LFC

        Macroscopic dielectric constant is defined as the real part of dielectric function at w=0.
        
        Parameters:

        eM_LFC: float
            Dielectric constant without local field correction. (RPA, ALDA)
        eM2_NLFC: float
            Dielectric constant with local field correction.

        """

        #assert self.optical_limit
        fd = self.chi0.fd
        prnt('', file=fd)
        prnt('%s Macroscopic Dielectric Constant:' % xc, file=fd)
        dirstr = ['x', 'y', 'z']
        tempdir = np.array([1,-1,1])

        if 1:  #for dir in distr:
        
            eM = np.zeros(2)
            df_NLFC_w, df_LFC_w = self.get_dielectric_function(xc=xc, direction=tempdir)
            eps0 = np.real(df_NLFC_w[0])
            eps = np.real(df_LFC_w[0])
            prnt('  %s direction' %dir, file=fd)
            prnt('    Without local field: %f' % eps0, file=fd)
            prnt('    Include local field: %f' % eps, file=fd)     
            
        return eps0, eps



    def get_eels_spectrum(self, xc='RPA', q_c=[0, 0, 0], 
                          direction='x', filename='eels.csv'):
        """Calculate EELS spectrum. By default, generate a file 'eels.csv'.

        EELS spectrum is obtained from the imaginary part of the inverse 
        of dielectric function. Returns EELS spectrum without and with 
        local field corrections:

        df_NLFC_w, df_LFC_w = DielectricFunction.get_eels_spectrum()
        """

        # Calculate dielectric function
        df_NLFC_w, df_LFC_w = self.get_dielectric_function(xc=xc, q_c=q_c,
                                                           direction=direction,
                                                           filename=None)
        Nw = df_NLFC_w.shape[0]
        
        # Calculate eels
        eels_NLFC_w  = -(1 / df_NLFC_w).imag
        eels_LFC_w = -(1 / df_LFC_w).imag        

        # Write to file
  #      if rank == 0:
        fd = open(filename, 'w')
        prnt('# energy, eels_NLFC_w, eels_LFC_w', file=fd)
        for iw in range(Nw):
            prnt('%.3f, %.3f, %.3f' %
                 (self.chi0.omega_w[iw]*Hartree, eels_NLFC_w[iw], eels_LFC_w[iw]), file=fd)
        fd.close()

        return eels_NLFC_w, eels_LFC_w
        # Wait for I/O to finish
        self.comm.barrier()

    def get_absorption_spectrum(self, xc='RPA', filename='absorption.csv'):
        """Calculate optical absorption spectrum. 
           
           By default, generate a file 'absorption.csv'. Optical absorption
           spectrum is obtained from the imaginary part of dielectric function.
        """

        for dir in ['x', 'y', 'z']:
            df_NLFC_w, df_LFC_w = self.get_dielectric_function(xc=xc, q_c = [0,0,0], dir=dir)
            
            Nw = df_NLFC_w.shape[0]

            f = open('%s_%s' % ('xyz'[dir], filename), 'w')
            #f = open(filename+'.%s'%(dirstr[dir]),'w') # ????
            for iw in range(Nw):
                prnt(self.chi0.omega_w[iw]*Hartree, df_NLFC_w[iw].real,
                     df_NLFC_w[iw].imag, df_LFC_w[iw].real, df_LFC_imag[iw].imag)
            f.close()

            # Wait for I/O to finish
            self.comm.barrier()
