import os
import sys
import pickle
from math import pi

import numpy as np
from ase.utils import prnt
from ase.units import Hartree, Bohr

import gpaw.mpi as mpi
from gpaw.response.chi0 import Chi0
from gpaw.response.kernel2 import calculate_Kxc
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb


class DielectricFunction:
    """This class defines dielectric function related physical quantities."""
    def __init__(self, calc, name=None, frequencies=None,  domega0=0.1, 
                 omegamax=20, alpha = 3.0, ecut=50, hilbert=False, nbands=None,
                 eta=0.2, ftol=1e-6, world=mpi.world, txt=sys.stdout):
            
        self.chi0 = Chi0(calc, frequencies, domega0=domega0, 
                         omegamax=omegamax, alpha=alpha, ecut=ecut,
                         hilbert=hilbert,
                         nbands=nbands, eta=eta, ftol=ftol, world=world,
                         txt=txt)
        
        self.name = name

    def calculate_chi0(self, q_c):
        if self.name:
            kd = self.chi0.calc.wfs.kd
            name = self.name + '%+d%+d%+d.pckl' % tuple((q_c * kd.N_c).round())
            if os.path.isfile(name):
                try:
                    pd, chi0_wGG, chi0_wxvG, chi0_wvv = pickle.load(open(name))
                except EOFError:
                    pass
                else:
                    return pd, chi0_wGG, chi0_wxvG, chi0_wvv
        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.chi0.calculate(q_c)
        self.chi0.timer.write(self.chi0.fd)
        
        if self.name and mpi.rank == 0:
            with open(name, 'wb') as fd:
                pickle.dump((pd, chi0_wGG, chi0_wxvG, chi0_wvv), fd,
                            pickle.HIGHEST_PROTOCOL)
        return pd, chi0_wGG, chi0_wxvG, chi0_wvv


    def get_chi(self, xc='RPA', q_c=[0, 0, 0], direction='x',
                wigner_seitz_truncation=False):
        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)
        G_G = pd.G2_qG[0]**0.5
        nG = len(G_G)

        if pd.kd.gamma:
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
            chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0_wvv, d_v).T)
        
        G_G /= (4 * pi)**0.5

        if wigner_seitz_truncation:
            kernel = WignerSeitzTruncatedCoulomb(pd.gd.cell_cv,
                                                 self.chi0.calc.wfs.kd.N_c)
            K_G = kernel.get_potential(pd)
            K_G *= G_G**2
            if pd.kd.gamma:
                K_G[0] = 0.0
        else:
            K_G = np.ones(nG)

        K_GG = np.zeros((nG, nG), dtype=complex)
        for i in range(nG):
            K_GG[i, i] = K_G[i]

        if xc != 'RPA':
            R_av = self.chi0.calc.atoms.positions / Bohr
            nt_sG = self.chi0.calc.density.nt_sG
            K_GG += calculate_Kxc(pd, nt_sG, R_av, self.chi0.calc.wfs.setups,
                                  self.chi0.calc.density.D_asp, 
                                  functional=xc) * G_G * G_G[:, np.newaxis]
            
        chi_wGG = []
        for chi0_GG in chi0_wGG:
            chi0_GG[:] = chi0_GG / G_G / G_G[:, np.newaxis] 
            chi_wGG.append(np.dot(np.linalg.inv(np.eye(nG) - 
                                                np.dot(chi0_GG, K_GG)), 
                                  chi0_GG))
        return chi0_wGG, np.array(chi_wGG)

    def get_dielectric_matrix(self, xc='RPA', q_c=[0, 0, 0],
                              direction='x', wigner_seitz_truncation=False, 
                              symmetric=True):
        
        """ Returns the symmetrized dielectric matrix:
        \tilde\epsilon_GG' = v^{-1/2}_G \epsilon_GG' v^{-1/2}_G' where
        epsilon_GG' = 1 - v_G * P_GG' and P_GG' is the polarization
        
        In RPA:   P = chi^0
        In TDDFT: P = (1 - chi^0 * f_xc)^{-1} chi^0
        
        The head of the inverse symmetrized dielectric matrix is equal
        to the head of the inverse dielectric matrix (inverse dielectric 
        function)
        """
        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)
        G_G = pd.G2_qG[0]**0.5
        nG = len(G_G)

        if pd.kd.gamma:
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
            chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0_wvv, d_v).T)

                    
        if wigner_seitz_truncation: 
            kernel = WignerSeitzTruncatedCoulomb(pd.gd.cell_cv, 
                                                 self.chi0.calc.wfs.kd.N_c)
            K_G = kernel.get_potential(pd)**0.5
            if pd.kd.gamma:
                K_G[0] = 0.0
        else:
            K_G = (4 * pi)**0.5 / G_G

        if xc != 'RPA':
            R_av = self.chi0.calc.atoms.positions / Bohr
            nt_sG = self.chi0.calc.density.nt_sG
            Kxc_sGG = calculate_Kxc(pd, nt_sG, R_av, 
                                    self.chi0.calc.wfs.setups,
                                    self.chi0.calc.density.D_asp,
                                    functional=xc)

        for chi0_GG in chi0_wGG:
            if xc == 'RPA':
                P_GG = chi0_GG
            else:
                P_GG = np.dot(np.linalg.inv(np.eye(nG) - 
                                            np.dot(chi0_GG, Kxc_sGG[0])),
                              chi0_GG)
            if symmetric:
                e_GG = np.eye(nG) - P_GG * K_G * K_G[:, np.newaxis]
            else:
                K_GG = (K_G**2* np.ones([nG, nG])).T
                e_GG = np.eye(nG) - P_GG * K_GG
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
        e_wGG = self.get_dielectric_matrix(xc, q_c, direction, 
                                           wigner_seitz_truncation)
        df_NLFC_w = np.zeros(len(e_wGG), dtype=complex)
        df_LFC_w = np.zeros(len(e_wGG), dtype=complex)

        for w, e_GG in enumerate(e_wGG):
            df_NLFC_w[w] = e_GG[0, 0]
            df_LFC_w[w] = 1 / np.linalg.inv(e_GG)[0, 0]
        
        if filename is not None and mpi.rank == 0:
            with open(filename, 'w') as fd:
                for omega, nlfc, lfc in zip(self.chi0.omega_w * Hartree,
                                            df_NLFC_w,
                                            df_LFC_w):
                    prnt('%.6f, %.6f, %.6f, %.6f, %.6f' %
                         (omega, nlfc.real, nlfc.imag, lfc.real, lfc.imag),
                         file=fd)
                
        return df_NLFC_w, df_LFC_w


    def get_macroscopic_dielectric_constant(self, xc='RPA', direction='x',
                                            wigner_seitz_truncation=False):
        """Calculate macroscopic dielectric constant. Returns eM_NLFC and eM_LFC

        Macroscopic dielectric constant is defined as the real part 
        of dielectric function at w=0.
        
        Parameters:

        eM_LFC: float
            Dielectric constant without local field correction. (RPA, ALDA)
        eM2_NLFC: float
            Dielectric constant with local field correction.
        """

        wst = wigner_seitz_truncation
        fd = self.chi0.fd
        prnt('', file=fd)
        prnt('%s Macroscopic Dielectric Constant:' % xc, file=fd)
       
        tempdir = np.array([1,0,0])

        df_NLFC_w, df_LFC_w = self.get_dielectric_function(xc=xc,
                                                           filename=None,
                                                           direction=direction,
                                                           wigner_seitz_truncation=wst)
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
        df_NLFC_w, df_LFC_w = self.get_dielectric_function(
            xc=xc, q_c=q_c,
            direction=direction,
            filename=None,
            wigner_seitz_truncation=wigner_seitz_truncation)
        Nw = df_NLFC_w.shape[0]
        
        # Calculate eels
        eels_NLFC_w  = -(1 / df_NLFC_w).imag
        eels_LFC_w = -(1 / df_LFC_w).imag        

        # Write to file
        if filename is not None and mpi.rank == 0:
            fd = open(filename, 'w')
            prnt('# energy, eels_NLFC_w, eels_LFC_w', file=fd)
            for iw in range(Nw):
                prnt('%.6f, %.6f, %.6f' %
                     (self.chi0.omega_w[iw] * Hartree, 
                      eels_NLFC_w[iw], eels_LFC_w[iw]), file=fd)
            fd.close()

        return eels_NLFC_w, eels_LFC_w
        
    def get_polarizability(self, xc='RPA', direction='x',
                           wigner_seitz_truncation=False,
                           filename='polarizability.csv', pbc=None):
        """Calculate the polarizability alpha. 
        In 3D the imaginary part of the polarizability is related to the
        dielectric function by Im(eps_M) = 4 pi * Im(alpha). In systems
        with reduced dimensionality the converged value of alpha is
        independent of the cell volume. This is not the case for eps_M, 
        which is ill defined. A truncated Coulomb kernel will always give
        eps_M = 1.0, whereas the polarizability maintains its structure.

        By default, generate a file 'polarizability.csv'. The five colomns are:
        frequency (eV), Real(alpha0), Imag(alpha0), Real(alpha), Imag(alpha)
        alpha0 is the result without local field effects and the 
        dimension of alpha is \AA to the power of non-periodic directions
        """

        cell_cv = self.chi0.calc.wfs.gd.cell_cv
        if not pbc:
            pbc_c = self.chi0.calc.atoms.pbc
        else:
            pbc_c = np.array(pbc)
        if pbc_c.all():
            V = 1.0
        else:
            V = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))

        if not wigner_seitz_truncation:
            # Without truncation alpha is simply related to eps_M
            df0_w, df_w = self.get_dielectric_function(xc=xc, q_c=[0, 0, 0],
                                                       filename=None,
                                                       direction=direction)
            alpha_w = V * (df_w - 1.0) / (4 * pi)
            alpha0_w = V * (df0_w - 1.0) / (4 * pi)
        else:
            # With truncation we need to calculate \chit = v^0.5*chi*v^0.5
            prnt('Using Wigner-Seitz truncated Coulomb interaction',
                 file=self.chi0.fd)
            chi0_wGG, chi_wGG = self.get_chi(xc=xc, direction=direction,
                                             wigner_seitz_truncation=True)
            alpha_w = -V * (chi_wGG[:,0,0]) / (4 * pi)
            alpha0_w = -V * (chi0_wGG[:,0,0]) / (4 * pi)

        Nw = len(alpha_w)
        if filename is not None and mpi.rank == 0:
            fd = open(filename, 'w')
            for iw in range(Nw):
                prnt('%.6f, %.6f, %.6f, %.6f, %.6f' %
                     (self.chi0.omega_w[iw] * Hartree,
                      alpha0_w[iw].real * Bohr**(sum(~pbc_c)), 
                      alpha0_w[iw].imag * Bohr**(sum(~pbc_c)), 
                      alpha_w[iw].real * Bohr**(sum(~pbc_c)), 
                      alpha_w[iw].imag * Bohr**(sum(~pbc_c))), file=fd)
            fd.close()

        return alpha0_w * Bohr**(sum(~pbc_c)), alpha_w * Bohr**(sum(~pbc_c))

    def check_sum_rule(self, spectrum=None):
        """Check f-sum rule.
           It takes the y of a spectrum as an entry and it check its integral"""
        
        fd = self.chi0.fd
        
        #prnt('df',df_NLFC_w, file=fd)
        if spectrum is None:
            raise ValueError('No spectrum input ')
        dw = self.chi0.omega_w[1]-self.chi0.omega_w[0]
        N1 = 0
        for iw in range(len(spectrum)):
            #prnt(N1, file=fd)
            w = iw * dw
            N1 += spectrum[iw] * w
        N1 *= dw * self.chi0.vol / (2 * pi**2)

        prnt('', file=fd)
        prnt('Sum rule:', file=fd)
        nv = self.chi0.calc.wfs.nvalence
        prnt('N1 = %f, %f  %% error' %(N1, (N1 - nv) / nv * 100), file=fd)

    def get_eigenmodes(self, q_c = [0, 0, 0], w_max = None,  name = None):
        
        """
        Plasmonic eigenmodes as eigenvectors of the dielectric matrix.  
        
        """ 
        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)
        e_wGG = self.get_dielectric_matrix(xc = 'RPA', q_c = q_c,
                                           wigner_seitz_truncation=True,
                                           symmetric=False)
        
        kd = self.chi0.calc.wfs.kd
        
        """ get real space grid for plasmon modes"""
        from gpaw.utilities.gpts import get_number_of_grid_points
        from gpaw.grid_descriptor import GridDescriptor
        cell_cv = pd.gd.cell_cv
        h = 0.2/Bohr
        pbc = self.chi0.calc.atoms.pbc
        N_c = get_number_of_grid_points(cell_cv, h = h, mode = 'fd', 
                                        realspace = True)
        gd = GridDescriptor(N_c, cell_cv, pbc) 
        r = gd.get_grid_point_coordinates()
              
        w_w = self.chi0.omega_w * Hartree
        if w_max:
            w_w[np.where(w_w < w_max)]
        Nw = len(w_w)
        print(Nw)
        nG =  e_wGG.shape[1]
             
        eig = np.zeros([Nw, nG], dtype = complex)
        vec = np.zeros([Nw, nG, nG], dtype = complex)
        vec_dual = np.zeros([Nw, nG, nG], dtype = complex)
       
        """find eigenvalues and eigenvectors"""
        e_GG = e_wGG[0]  
        eig[0], vec[0] = np.linalg.eig(e_GG)
        vec_dual[0] = np.linalg.inv(vec[0])
        omega0 = np.array([])
        eigen0 = np.array([], dtype = complex)
        v_ind = np.zeros([0, r.shape[1], r.shape[2], r.shape[3]], dtype = complex)
        n_ind = np.zeros([0, r.shape[1], r.shape[2], r.shape[3]], dtype = complex)
        """ 
        loop to find the eigenvalues that crosses zero 
        from negative to positive values: """
        for i in np.array(range(1,Nw)): 
            e_GG = e_wGG[i]  # epsilon_GG'(omega + d-omega)
            eig[i], vec[i] = np.linalg.eig(e_GG)       
            vec_dual[i] = np.linalg.inv(vec[i])
            for k in range(nG):
                for m in range(nG):                   
                    if eig[i-1,k] < 0 and 0 < eig[i,m]:
                        """ check it's the same mode - 
                        Overlap between eigenvectors should be large:""" 
                        if abs(np.inner(vec[i-1,:,k], vec_dual[i,m,:])) > 0.95:
                            a = np.real((eig[i,m]-eig[i-1,k]) / (w_w[i]-w_w[i-1]))
                            w0 = np.real(-eig[i-1,k]) / a + w_w[i-1]
                            eig0 = a*(w0-w_w[i-1])+eig[i-1,k]
                            print('crossing found at w = %1.1f eV'%w0)
                            omega0 = np.append(omega0, w0)
                            eigen0 = np.append(eigen0, eig0)
                            n_dummy = np.zeros([1, r.shape[1], r.shape[2],
                                                r.shape[3]], dtype = complex)
                            v_dummy = np.zeros([1, r.shape[1], r.shape[2],
                                                r.shape[3]], dtype = complex)   
                            
                            for iG in range(nG):  # Fourier transform
                                qG = np.ravel(pd.G_Qv[pd.Q_qG[0]][iG] + pd.K_qv) ### is this iG+q???
                                coef_G = np.dot(qG, qG) / (4 * pi)
                                qGr_R = np.inner(qG, r.T).T
                                v_dummy += vec[i-1, iG, k] * np.exp(1j * qGr_R) 
                                n_dummy += vec[i-1, iG, k] * np.exp(1j * qGr_R) * coef_G
                            v_ind = np.append(v_ind, v_dummy, axis=0)
                            n_ind = np.append(n_ind, n_dummy, axis=0)   
                        
                                          
        if name is None and self.name:          
            name = self.name + '%+d%+d%+d-eigenmodes.pckl' % tuple((q_c * kd.N_c).round())          
        elif name:
            name = name + '%+d%+d%+d-eigenmodes.pckl' % tuple((q_c * kd.N_c).round())
        else:
            return r*Bohr, w_w, eig, omega0, eigen0, v_ind, n_ind

        pickle.dump((r*Bohr, w_w, eig, omega0, eigen0, v_ind, n_ind), open(name, 'wb'), 
                        pickle.HIGHEST_PROTOCOL)
        """Returns: real space grid, frequency grid, all eigenvalues, zero-crossing 
        frequencies + eigenvalues, induced potential + density in real space"""
        return r*Bohr, w_w, eig, omega0, eigen0, v_ind, n_ind    
