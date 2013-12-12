import os
import sys
import pickle

import numpy as np
from ase.utils import prnt
from ase.units import Hartree, Bohr

import gpaw.mpi as mpi
from gpaw.response.chi0 import Chi0
from gpaw.response.kernel2 import calculate_Kxc
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb


class DielectricFunction:
    """This class defines dielectric function related physical quantities."""
    def __init__(self, calc, name=None, frequencies=None, ecut=50,
                 eta=0.2, ftol=1e-6,
                 world=mpi.world, txt=sys.stdout):
        if frequencies is None:
            frequencies = np.linspace(0, 20, 51, end=True)
            
        self.chi0 = Chi0(calc, frequencies, ecut=ecut, eta=eta, ftol=ftol,
                         world=world, txt=txt)
        
        self.name = name

    def calculate_chi0(self, q_c):
        if self.name:
            kd = self.chi0.calc.wfs.kd
            name = self.name + '%+d%+d%+d.pckl' % tuple((q_c * kd.N_c).round())
            if os.path.isfile(name):
                try:
                    pd, chi0_wGG, chi0_wxvG = pickle.load(open(name))
                except EOFError:
                    pass
                else:
                    return pd, chi0_wGG, chi0_wxvG
        pd, chi0_wGG, chi0_wxvG = self.chi0.calculate(q_c)

        if self.name:
            pickle.dump((pd, chi0_wGG, chi0_wxvG), open(name, 'wb'),
                        pickle.HIGHEST_PROTOCOL)
        return pd, chi0_wGG, chi0_wxvG


    def get_chi(self, xc='RPA', q_c=[0, 0, 0], direction='x',
                wigner_seitz_truncation=False):
        pd, chi0_wGG, chi0_wxvG = self.calculate_chi0(q_c)
        G_G = pd.G2_qG[0]**0.5

        if G_G[0] == 0.0:
            G_G[0] = 1.0
            if isinstance(direction, str):
                d_v = {'x': [1, 0, 0],
                       'y': [0, 1, 0],
                       'z': [0, 0, 1]}[direction]
            else:
                d_v = direction
            d_v = np.asarray(d_v) / np.linalg.norm(d_v)
            chi0_wGG[:, 0] = np.dot(d_v, chi0_wxvG[:, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0_wxvG[:, 1])
            chi0_wGG[:, 0, 0] = np.dot(chi0_wxvG[:, 0, :, 0], d_v**2)
        

        chi_wGG = []
        G_G /= (4 * np.pi)**0.5
        nG = len(G_G)

        if xc == 'RPA':
            for chi0_GG in chi0_wGG:
                chi0_GG[:] = chi0_GG / G_G / G_G[:, np.newaxis] 
                chi_wGG.append(np.dot(np.linalg.inv(np.eye(nG) - chi0_GG), chi0_GG))

        elif xc == 'ALDA':
            R_av = self.chi0.calc.atoms.positions / Bohr
            nt_sG = self.chi0.calc.density.nt_sG
            
            Kxc_sGG = calculate_Kxc(pd, nt_sG, R_av, self.chi0.calc.wfs.setups,
                                    self.chi0.calc.density.D_asp,
                                    functional=xc)
            
            nG = len(G_G)
            for chi0_GG in chi0_wGG:
                e_GG = (np.dot(chi0_GG, np.linalg.inv(np.eye(nG) - 
                        np.dot(Kxc_sGG[0], chi0_GG)))) / G_G / G_G[:, np.newaxis]
                chi0_GG[:] = e_GG

        return chi0_wGG, np.array(chi_wGG)

    def get_dielectric_matrix(self, xc='RPA', q_c=[0, 0, 0],
                              direction='x', wigner_seitz_truncation=False):

        pd, chi0_wGG, chi0_wxvG = self.calculate_chi0(q_c)
        G_G = pd.G2_qG[0]**0.5

        if G_G[0] == 0.0:
            G_G[0] = 1.0
            if isinstance(direction, str):
                d_v = {'x': [1, 0, 0],
                       'y': [0, 1, 0],
                       'z': [0, 0, 1]}[direction]
            else:
                d_v = direction

            d_v = np.asarray(d_v) / np.linalg.norm(d_v)
            chi0_wGG[:, 0] = np.dot(d_v, chi0_wxvG[:, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0_wxvG[:, 1])
            chi0_wGG[:, 0, 0] = np.dot(chi0_wxvG[:, 0, :, 0], d_v**2)

        if xc == 'RPA':
            nG = len(G_G)
            for chi0_GG in chi0_wGG:
                e_GG = np.eye(nG) - 4 * np.pi * chi0_GG / G_G / G_G[:, np.newaxis]
                chi0_GG[:] = e_GG

        elif xc == 'ALDA':
            R_av = self.chi0.calc.atoms.positions / Bohr
            nt_sG = self.chi0.calc.density.nt_sG

            Kxc_sGG = calculate_Kxc(pd, nt_sG, R_av, self.chi0.calc.wfs.setups,
                                    self.chi0.calc.density.D_asp,
                                    functional=xc)

            if wigner_seitz_truncation: 
                kernel = WignerSeitzTruncatedCoulomb(pd.gd.cell_cv, 
                                                     self.chi0.calc.wfs.kd.N_c)
                K_G = kernel.get_potential(pd)
            else:
                K_G = 4 * np.pi * 1 / G_G**2.0
            
            Kc_GG = np.sqrt(np.outer(K_G,K_G))

            nG = len(G_G)
            for chi0_GG in chi0_wGG:
                e_GG = (np.eye(nG) - np.dot(Kc_GG*np.eye(nG),np.dot(chi0_GG, np.linalg.inv(np.eye(nG) - np.dot(Kxc_sGG[0], chi0_GG)))))

                chi0_GG[:] = e_GG
        
        # chi0_wGG is now the dielectric matrix
        return chi0_wGG

    def get_dielectric_function(self, xc='RPA', q_c=[0, 0, 0],
                                direction='x', filename='df.csv',
                                wigner_seitz_truncation=False):
        """Calculate the dielectric function.

        Returns dielectric function without and with local field correction:
 
        df_NLFC_w, df_LFC_w = DielectricFunction.get_dielectric_function()
        """
        e_wGG = self.get_dielectric_matrix(xc, q_c, direction, wigner_seitz_truncation)
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


    def get_macroscopic_dielectric_constant(self, xc='RPA', direction='x'):
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
       
        tempdir = np.array([1,0,0])

        #eM = np.zeros(2)
        df_NLFC_w, df_LFC_w = self.get_dielectric_function(xc=xc, direction=direction)
        eps0 = np.real(df_NLFC_w[0])
        eps = np.real(df_LFC_w[0])
        prnt('  %s direction' %direction, file=fd)
        prnt('    Without local field: %f' % eps0, file=fd)
        prnt('    Include local field: %f' % eps, file=fd)     
            
        return eps0, eps



    def get_eels_spectrum(self, xc='RPA', q_c=[0, 0, 0], 
                          direction='x', filename='eels.csv',
                          wigner_seitz_truncation=False):
        """Calculate EELS spectrum. By default, generate a file 'eels.csv'.

        EELS spectrum is obtained from the imaginary part of the inverse 
        of dielectric function. Returns EELS spectrum without and with 
        local field corrections:

        df_NLFC_w, df_LFC_w = DielectricFunction.get_eels_spectrum()
        """

        # Calculate dielectric function
        df_NLFC_w, df_LFC_w = self.get_dielectric_function(xc=xc, q_c=q_c,
                                                           direction=direction,
                                                           filename=None,
                                                           wigner_seitz_truncation=wigner_seitz_truncation)
        Nw = df_NLFC_w.shape[0]
        
        # Calculate eels
        eels_NLFC_w  = -(1 / df_NLFC_w).imag
        eels_LFC_w = -(1 / df_LFC_w).imag        

        # Write to file
  #      if rank == 0:
        fd = open(filename, 'w')
        prnt('# energy, eels_NLFC_w, eels_LFC_w', file=fd)
        for iw in range(Nw):
            prnt('%.6f, %.6f, %.6f' %
                 (self.chi0.omega_w[iw]*Hartree, eels_NLFC_w[iw], eels_LFC_w[iw]), file=fd)
        fd.close()

        return eels_NLFC_w, eels_LFC_w
        # Wait for I/O to finish
        self.comm.barrier()
        
        
        
    def check_sum_rule(self, spectrum=None):
        """Check f-sum rule.
           It takes the y of a spectrum as an entry and it check its integral"""
        
        fd = self.chi0.fd
        
        #prnt('df',df_NLFC_w, file=fd)
        if spectrum is None:
            raise ValueError('No spectrum input ')
        dw = self.chi0.omega_w[1]-self.chi0.omega_w[0]
        N1 = N2 = 0
        for iw in range(len(spectrum)):
            #prnt(iw, file=fd)
            w = iw * dw
            N1 += np.imag(spectrum[iw]) * w
        N1 *= dw * self.chi0.vol / (2 * np.pi**2)

        
        prnt('', file=fd)
        prnt('Sum rule:', file=fd)
        nv = self.chi0.calc.wfs.nvalence
        prnt('Without local field: N1 = %f, %f  %% error' %(N1, (N1 - nv) / nv * 100), file=fd)
        

        


    def get_polarizability(self, xc='RPA', direction='x',
                           wigner_seitz_truncation=False,
                           filename='absorption.csv', pbc=None):
        """Calculate polarizability. The imaginary part gives the absorption spectrum
           
           By default, generate a file 'absorption.csv'. Optical absorption
           spectrum is obtained from the imaginary part of dielectric function.
        """
        cell_cv = self.chi0.calc.wfs.gd.cell_cv
        if not pbc:
            pbc_c = self.chi0.calc.atoms.pbc
        else:
            pbc_c = pbc
        if pbc_c.all():
            V = 1.0
        else:
            V = np.abs(np.linalg.det(cell[~pbc_c][:, ~pbc_c]))

        df_NLFC_w, df_LFC_w = self.get_dielectric_function(xc=xc, q_c = [0,0,0], direction=direction)
            
        Nw = df_NLFC_w.shape[0]
        if not wigner_seitz_truncation:
            df0_w, df_w = self.get_dielectric_function(xc=xc, q_c=[0,0,0],
                                                       direction=direction)
            alpha_w = V * (df_w - 1.0) / (4 * np.pi)
            alpha0_w = V * (df0_w - 1.0) / (4 * np.pi)
        else:
            prnt('Using Wigner-Seitz truncated Coulomb interaction',
                     file=self.chi0.fd)
            chi0_wGG, chi_wGG = self.get_chi(xc=xc, direction=direction)
            epsM = 1.0 / (1.0 + chi_wGG[:, 0, 0])
            eps0M = 1.0 / (1.0 + chi0_wGG[:, 0, 0])
            alpha_w = V * (epsM - 1.0) / (4 * np.pi)
            alpha0_w = V * (eps0M - 1.0) / (4 * np.pi)
        Nw = len(alpha_w)

        if mpi.rank == 0:
            f = open(filename, 'w')
            for iw in range(Nw):
                prnt(self.chi0.omega_w[iw] * Hartree, 
                     alpha0_w[iw].real, alpha0_w[iw].imag, 
                     alpha_w[iw].real, alpha_w[iw].imag, file=f)
            f.close()

        return alpha0_w * Bohr**(sum(~pbc_c)), alpha_w * Bohr**(sum(~pbc_c))
