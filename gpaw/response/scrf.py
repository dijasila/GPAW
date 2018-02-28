from __future__ import print_function

import os
import sys
import pickle
from math import pi

import numpy as np
from ase.units import Hartree, Bohr
from ase.parallel import parprint

import gpaw.mpi as mpi

from gpaw.response.chi0 import Chi0

from gpaw.response.kernels import get_coulomb_kernel
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
from gpaw.response.fxc import get_xc_kernel, get_xc_spin_kernel





class SpinChargeResponseFunction:
    """This class provides physical quantities related to the spin-charge response function."""
    def __init__(self, calc, name=None, frequencies=None, domega0=0.1,
                 omega2=10.0, omegamax=None, ecut=50, hilbert=True,
                 nbands=None, eta=0.2, ftol=1e-6, threshold=1,
                 intraband=True, nblocks=1, world=mpi.world, txt=sys.stdout,
                 gate_voltage=None, truncation=None, disable_point_group=False,
                 disable_time_reversal=False,
                 integrationmode=None, pbc=None, rate=0.0,
                 omegacutlower=None, omegacutupper=None, eshift=0.0):
        """Creates a SpinChargeResponseFunction object.

        calc: str
            The groundstate calculation file that the linear response
            calculation is based on.
        name: str
            If defined, save the response function to::

                name + '%+d%+d%+d.pckl' % tuple((q_c * kd.N_c).round())

            where q_c is the reduced momentum and N_c is the number of
            kpoints along each direction.
        frequencies: np.ndarray
            Specification of frequency grid. If not set the non-linear
            frequency grid is used.
        domega0: float
            Frequency grid spacing for non-linear frequency grid at omega = 0.
        omega2: float
            Frequency at which the non-linear frequency grid has doubled
            the spacing.
        omegamax: float
            The upper frequency bound for the non-linear frequency grid.
        ecut: float
            Plane-wave cut-off.
        hilbert: bool
            Use hilbert transform.
        nbands: int
            Number of bands from calc.
        eta: float
            Broadening parameter.
        ftol: float
            Threshold for including close to equally occupied orbitals,
            f_ik - f_jk > ftol.
        threshold: float
            Threshold for matrix elements in optical response perturbation
            theory.
        intraband: bool
            Include intraband transitions.
        world: comm
            mpi communicator.
        nblocks: int
            Split matrices in nblocks blocks and distribute them G-vectors or
            frequencies over processes.
        txt: str
            Output file.
        gate_voltage: float
            Shift Fermi level of ground state calculation by the
            specified amount.
        truncation: str
            'wigner-seitz' for Wigner Seitz truncated Coulomb.
            '2D, 1D or 0d for standard analytical truncation schemes.
            Non-periodic directions are determined from k-point grid
        eshift: float
            Shift unoccupied bands
        """

        self.chi0 = Chi0(calc, frequencies=frequencies,
                         domega0=domega0, omega2=omega2, omegamax=omegamax,
                         ecut=ecut, hilbert=hilbert, nbands=nbands,
                         eta=eta, ftol=ftol, threshold=threshold,
                         intraband=intraband, world=world, nblocks=nblocks,
                         txt=txt, gate_voltage=gate_voltage,
                         disable_point_group=disable_point_group,
                         disable_time_reversal=disable_time_reversal,
                         integrationmode=integrationmode,
                         pbc=pbc, rate=rate, eshift=eshift)

        self.name = name

        self.omega_w = self.chi0.omega_w
        if omegacutlower is not None:
            inds_w = np.logical_and(self.omega_w > omegacutlower / Hartree,
                                    self.omega_w < omegacutupper / Hartree)
            self.omega_w = self.omega_w[inds_w]

        nw = len(self.omega_w)

        world = self.chi0.world
        self.mynw = (nw + world.size - 1) // world.size
        self.w1 = min(self.mynw * world.rank, nw)
        self.w2 = min(self.w1 + self.mynw, nw)
        self.truncation = truncation
    
    
    def get_chi_grid_dim(self, q_c):
        """Pass dimensions involved in chi grid, without running calculation."""
        return self.chi0.get_chi_grid_dim(q_c)
    
    
    def calculate_chi0(self, q_c, spin='all'):
        """Calculates the response function.

        Calculate the response function for a specific momentum.

        q_c: [float, float, float]
            The momentum wavevector.
        spin : str or int
            If 'all' then include all spins. 
            If 0 or 1, only include this specific spin.
            If 'pm' calculate chi^{+-}_0
            If 'mp' calculate chi^{-+}_0
        """
    
        response = self.chi0.get_response()
        
        if not response in ['density', 'spin']:
            raise Exception(""" Currently only scalar response functions of type 'density' and 
                                'spin' are implemented.""")
        
        if self.name:
            kd = self.chi0.calc.wfs.kd
            name = self.name + '%+d%+d%+d.pckl' % tuple((q_c * kd.N_c).round())
            if os.path.isfile(name):
                return self.read(name)

        pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv = self.chi0.calculate(q_c, spin)
        chi0_wGG = self.chi0.distribute_frequencies(chi0_wGG)
        self.chi0.timer.write(self.chi0.fd)
        if self.name:
            self.write(name, pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv)

        return pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv

    def write(self, name, pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv):
        nw = len(self.omega_w)
        nG = finepd.ngmax
        world = self.chi0.world

        if world.rank == 0:
            fd = open(name, 'wb')
            pickle.dump((self.omega_w, pd, finepd, None, chi0_wxvG, chi0_wvv),
                        fd, pickle.HIGHEST_PROTOCOL)
            for chi0_GG in chi0_wGG:
                pickle.dump(chi0_GG, fd, pickle.HIGHEST_PROTOCOL)

            tmp_wGG = np.empty((self.mynw, nG, nG), complex)
            w1 = self.mynw
            for rank in range(1, world.size):
                w2 = min(w1 + self.mynw, nw)
                world.receive(tmp_wGG[:w2 - w1], rank)
                for w in range(w2 - w1):
                    pickle.dump(tmp_wGG[w], fd, pickle.HIGHEST_PROTOCOL)
                w1 = w2
            fd.close()
        else:
            world.send(chi0_wGG, 0)

    def read(self, name):
        print('Reading from', name, file=self.chi0.fd)
        fd = open(name, 'rb')
        omega_w, pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv = pickle.load(fd)
        for omega in self.omega_w:
            assert np.any(np.abs(omega - omega_w) < 1e-8)

        wmin = np.argmin(np.abs(np.min(self.omega_w) - omega_w))
        world = self.chi0.world

        nw = len(omega_w)
        nG = finepd.ngmax

        if chi0_wGG is not None:
            # Old file format:
            chi0_wGG = chi0_wGG[wmin + self.w1:self.w2].copy()
        else:
            if world.rank == 0:
                chi0_wGG = np.empty((self.mynw, nG, nG), complex)
                for _ in range(wmin):
                    pickle.load(fd)
                for chi0_GG in chi0_wGG:
                    chi0_GG[:] = pickle.load(fd)
                tmp_wGG = np.empty((self.mynw, nG, nG), complex)
                w1 = self.mynw
                for rank in range(1, world.size):
                    w2 = min(w1 + self.mynw, nw)
                    for w in range(w2 - w1):
                        tmp_wGG[w] = pickle.load(fd)
                    world.send(tmp_wGG[:w2 - w1], rank)
                    w1 = w2
            else:
                chi0_wGG = np.empty((self.w2 - self.w1, nG, nG), complex)
                world.receive(chi0_wGG, 0)

        if chi0_wvv is not None:
            chi0_wxvG = chi0_wxvG[wmin:wmin + nw]
            chi0_wvv = chi0_wvv[wmin:wmin + nw]

        return pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv

    def collect(self, a_w):
        world = self.chi0.world
        b_w = np.zeros(self.mynw, a_w.dtype)
        b_w[:self.w2 - self.w1] = a_w
        nw = len(self.omega_w)
        A_w = np.empty(world.size * self.mynw, a_w.dtype)
        world.all_gather(b_w, A_w)
        return A_w[:nw]

    def get_frequencies(self):
        """ Return frequencies that Chi is evaluated on"""
        return self.omega_w * Hartree

    def get_chi(self, xc='RPA', q_c=[0, 0, 0], spin='all',
                direction='x', return_VchiV=True, q_v=None,
                density_cut=None, xi_cut=None, 
                fxc_scaling=None, Dt=None):
        """ Returns v^1/2 chi v^1/2 for the density response and chi for the
        spin response. The truncated Coulomb interaction is included as 
        v^-1/2 v_t v^-1/2. This is in order to conform with
        the head and wings of chi0, which is treated specially for q=0.
        
        spin : str or int
            If 'all' then include all spins. 
            If 0 or 1, only include this specific spin.
            If 'pm' calculate chi^{+-}_0
            If 'mp' calculate chi^{-+}_0
        xi_cut : float
            cutoff spin polarization below which f_xc is evaluated in 
            unpolarized limit (to make sure divergent terms cancel out correctly)
        density_cut : float
            cutoff density below which f_xc is set to zero
        fxc_scaling : float or str
            Possible scaling of kernel to hit Goldstone mode.
            float input gives a flat scaling factor, 'Goldstone' automatically fulfills the theorem.
        Dt : float
            Response time [1/eV] used in semi-adiabatic approximation
        
        Note: currently only 'RPA', 'ALDA_x', 'ALDA_X' and 'ALDA' are implemented for spin response.
        """
        
        response = self.chi0.get_response()
        
        assert response in ['density', 'spin']
        if response == 'spin':
            assert xc in ['RPA', 'ALDA_x', 'ALDA_X', 'ALDA', 'ALDA_ae1', 'ALDA_ae2'] ### added for error finding ###
        
        pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c, spin)
        
        N_c = self.chi0.calc.wfs.kd.N_c
        
        if response == 'density':
            Kbare_G = get_coulomb_kernel(finepd,
                                         N_c,
                                         truncation=None,
                                         q_v=q_v)
            vsqr_G = Kbare_G**0.5
            nG = len(vsqr_G)
            
            if self.truncation is not None:
                if self.truncation == 'wigner-seitz':
                    self.wstc = WignerSeitzTruncatedCoulomb(finepd.gd.cell_cv, N_c)
                else:
                    self.wstc = None
                    Ktrunc_G = get_coulomb_kernel(finepd,
                                                  N_c,
                                                  truncation=self.truncation,
                                                  wstc=self.wstc,
                                                  q_v=q_v)
                    K_GG = np.diag(Ktrunc_G / Kbare_G)
            else:
                K_GG = np.eye(nG, dtype=complex)
               
        if response == 'density': # With spin response, no special care is needed for the gamma point
          if finepd.kd.gamma:
              if isinstance(direction, str):
                  d_v = {'x': [1, 0, 0],
                         'y': [0, 1, 0],
                         'z': [0, 0, 1]}[direction]
              else:
                  d_v = direction
              d_v = np.asarray(d_v) / np.linalg.norm(d_v)
              W = slice(self.w1, self.w2)
              chi0_wGG[:, 0] = np.dot(d_v, chi0_wxvG[W, 0])
              chi0_wGG[:, :, 0] = np.dot(d_v, chi0_wxvG[W, 1])
              chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0_wvv[W], d_v).T)
            
        
        if response == 'density':
            if xc != 'RPA':
                Kxc_sGG = get_xc_kernel(finepd,
                                        self.chi0,
                                        functional=xc,
                                        chi0_wGG=chi0_wGG,
                                        density_cut=density_cut)
                K_GG += Kxc_sGG[0] / vsqr_G / vsqr_G[:, np.newaxis]
               
            # Invert Dyson eq.
            chi_wGG = []
            for chi0_GG in chi0_wGG:
                """v^1/2 chi0 V^1/2"""
                chi0_GG[:] = chi0_GG * vsqr_G * vsqr_G[:, np.newaxis]
                chi_GG = np.dot(np.linalg.inv(np.eye(nG) -
                                              np.dot(chi0_GG, K_GG)),
                                chi0_GG)
                if not return_VchiV:
                    chi0_GG /= vsqr_G * vsqr_G[:, np.newaxis]
                    chi_GG /= vsqr_G * vsqr_G[:, np.newaxis]
                chi_wGG.append(chi_GG)
        else:
            if xc != 'RPA':
                Kxc_GG = get_xc_spin_kernel(finepd,
                                            self.chi0,
                                            functional=xc,
                                            xi_cut=xi_cut,
                                            density_cut=density_cut)
            else: # RPA is non-interacting for the spin response
                return pd, finepd, chi0_wGG, chi0_wGG
            
            chi_wGG = []
            
            # Find fxc_scaling if automated scaling is specified
            if not fxc_scaling is None:
              if isinstance(fxc_scaling, str) and fxc_scaling == 'Goldstone':
                parprint("Finding rescaling to fulfill the Goldstone theorem")
                
                ## Collect w=0 data XXX not efficient to use redistribute
                world = self.chi0.world
                #try:
                #  print("bf re, chi0_0GG", world.rank, chi0_wGG[0])  ### error finding ###
                #except:
                #  print("failed bf re, chi0_0GG", world.rank)  ### error finding ###
                #tmpchi0_wGG = self.chi0.redistribute(chi0_wGG)
                #try:
                #  print("bf re, chi0_0GG", world.rank, tmpchi0_wGG[0])  ### error finding ###
                #except:
                #  print("failed bf re, chi0_0GG", world.rank)  ### error finding ###
                
                # Only rank 0 has w=0 and finds rescaling
                fxc_sbuf = np.empty(1, dtype=float)
                if world.rank == 0:
                  fxc_scaling = 1.0
                  #chi0_0GG = tmpchi0_wGG[0]  ### error finding ###
                  chi0_0GG = chi0_wGG[0]
                  chi_0GG = np.dot(np.linalg.inv(np.eye(len(chi0_0GG)) -
                                                 np.dot(chi0_0GG, Kxc_GG*fxc_scaling)),
                                   chi0_0GG)
                  kappa_M_0 = (chi0_0GG[0,0]/chi_0GG[0,0]).real
                  #print("chi0_wGG", chi0_wGG)  ### error finding ###
                  #print("chi0_0GG", chi0_0GG)  ### error finding ###
                  #print("kappa_M_0", kappa_M_0)  ### error finding ###
                  scaling_incr = 0.1*np.sign(kappa_M_0)
                  while abs(kappa_M_0) > 1e-5 and abs(scaling_incr) > 1e-5:
                    fxc_scaling += scaling_incr
                    if fxc_scaling <= 0.0 or fxc_scaling >= 10.:
                      raise ValueError('Found a fxc_scaling of %.4f during scaling' % fxc_scaling)
                    chi_0GG = np.dot(np.linalg.inv(np.eye(len(chi0_0GG)) -
                                                   np.dot(chi0_0GG, Kxc_GG*fxc_scaling)),
                                     chi0_0GG)
                    kappa_M_0 = (chi0_0GG[0,0]/chi_0GG[0,0]).real
                    
                    if np.sign(kappa_M_0) != np.sign(scaling_incr):
                      scaling_incr *= -0.2
                    #print(fxc_scaling, kappa_M_0)  ### error finding ###
                  fxc_sbuf[:] = fxc_scaling
                  #print(fxc_sbuf)  ### error finding ###
                # Broadcast found rescaling  
                world.broadcast(fxc_sbuf, 0)
                #print(world.rank, fxc_sbuf)  ### error finding ###
                fxc_scaling = fxc_sbuf[0]
            else:
              fxc_scaling = 1.0
            
            #fxc_scaling = 1.0  ### error finding ###
            #world = self.chi0.world  ### error finding ###
            #if world.rank == 0:  ### error finding ###
            #  print("chi0_0GG", q_c, chi0_wGG[0])  ### error finding ###
            
            # Add factor for semi-adiabatic approximation
            if not Dt is None:
              Dt *= Hartree
              fxc_scaling_w = fxc_scaling * np.exp(-Dt*self.omega_w)
            else:
              fxc_scaling_w = fxc_scaling * np.ones(len(self.omega_w))
            
            # Invert Dyson equation
            for (chi0_GG, fxcs) in zip(chi0_wGG, fxc_scaling_w):
                chi_GG = np.dot(np.linalg.inv(np.eye(len(chi0_GG)) -
                                              np.dot(chi0_GG, Kxc_GG*fxcs)),
                                chi0_GG)
                
                #print(np.dot(chi0_GG, Kxc_GG*re_factor))  ### error finding ###
                
                chi_wGG.append(chi_GG)
            
            #parprint("q_c", q_c)  ### error finding ###
            #parprint("chi0_30_0G", chi0_wGG[30][0,:])  ### error finding ###
            #parprint("chi0_30_G0", chi0_wGG[30][:,0])  ### error finding ###
            #parprint("Kxc_GG", Kxc_GG)  ### error finding ###
            #parprint("chi_30_0G", chi_wGG[30][0,:])  ### error finding ###
            #parprint("chi_30_G0", chi_wGG[30][:,0])  ### error finding ###
        
        if not fxc_scaling is None:
          return pd, finepd, chi0_wGG, np.array(chi_wGG), fxc_scaling
        return pd, finepd, chi0_wGG, np.array(chi_wGG)
    
    
    def get_density_response_function(self, xc='RPA', q_c=[0, 0, 0], q_v=None,
                                      direction='x', density_cut=None,
                                      filename='drf.csv'):
        """Calculate the density response function.
        
        Returns macroscopic density response function:
        drf0_w, drf_xc_w = SpinChargeResponseFunction.get_density_response_function()
        
        """
        
        self.chi0.set_response('density')
        
        pd, finepd, chi0_wGG, chi_wGG = self.get_chi(xc=xc, q_c=q_c, direction=direction, 
                                                     return_VchiV = False, density_cut=density_cut)
        
        drf0_w = np.zeros(len(chi_wGG), dtype=complex)
        drf_xc_w = np.zeros(len(chi_wGG), dtype=complex)

        for w, (chi0_GG, chi_GG) in enumerate(zip(chi0_wGG, chi_wGG)):
            drf0_w[w] = chi0_GG[0, 0]
            drf_xc_w[w] = chi_GG[0, 0]

        drf0_w = self.collect(drf0_w)
        drf_xc_w = self.collect(drf_xc_w)

        if filename is not None and mpi.rank == 0:
            with open(filename, 'w') as fd:
                for omega, drf0, drf_xc in zip(self.omega_w * Hartree,
                                                  drf0_w,
                                                  drf_xc_w):
                    print('%.6f, %.6f, %.6f, %.6f, %.6f' %
                          (omega, drf0.real, drf0.imag, drf_xc.real, drf_xc.imag),
                          file=fd)

        return drf0_w, drf_xc_w    

    def get_spin_response_function(self, xc='ALDA', q_c=[0, 0, 0], q_v=None,
                                   direction='x', flip='pm', density_cut=None, xi_cut=None,
                                   filename='srf.csv', return_VchiV = False, fxc_scaling=None, Dt=None):
        """Calculate the spin response function.
         
        Returns macroscopic spin response function:
        srf0_w, srf_xc_w = SpinChargeResponseFunction.get_spin_response_function()
        """
              
        self.chi0.set_response('spin') 
        assert self.chi0.eta > 0.0
        assert not self.chi0.hilbert
        assert not self.chi0.timeordered
        assert self.chi0.disable_point_group
        assert self.chi0.disable_time_reversal
        
        pd, finepd, chi0_wGG, chi_wGG, fxc_scaling = self.get_chi(xc=xc, q_c=q_c, spin=flip,
                                                                  direction=direction, density_cut=density_cut,
                                                                  xi_cut=xi_cut, fxc_scaling=fxc_scaling, Dt=Dt)
        
        srf0_w = np.zeros(len(chi_wGG), dtype=complex)
        srf_xc_w = np.zeros(len(chi_wGG), dtype=complex)

        for w, (chi0_GG, chi_GG) in enumerate(zip(chi0_wGG, chi_wGG)):
            srf0_w[w] = chi0_GG[0, 0]
            srf_xc_w[w] = chi_GG[0, 0]

        srf0_w = self.collect(srf0_w)
        srf_xc_w = self.collect(srf_xc_w)

        if filename is not None and mpi.rank == 0:
            with open(filename, 'w') as fd:
                for omega, srf0, srf_xc in zip(self.omega_w * Hartree,
                                                  srf0_w,
                                                  srf_xc_w):
                    print('%.6f, %.6f, %.6f, %.6f, %.6f' %
                          (omega, srf0.real, srf0.imag, srf_xc.real, srf_xc.imag),
                          file=fd)

        return srf0_w, srf_xc_w, fxc_scaling
    
    def get_dielectric_matrix(self, xc='RPA', q_c=[0, 0, 0],
                              direction='x', symmetric=True,
                              calculate_chi=False, q_v=None,
                              add_intraband=True):
        """Returns the symmetrized dielectric matrix.

        ::

            \tilde\epsilon_GG' = v^{-1/2}_G \epsilon_GG' v^{1/2}_G',

        where::

            epsilon_GG' = 1 - v_G * P_GG' and P_GG'

        is the polarization.

        ::

            In RPA:   P = chi^0
            In TDDFT: P = (1 - chi^0 * f_xc)^{-1} chi^0

        in addition to RPA one can use the kernels, ALDA, rALDA, rAPBE,
        Bootstrap and LRalpha (long-range kerne), where alpha is a user
        specified parameter (for example xc='LR0.25')

        The head of the inverse symmetrized dielectric matrix is equal
        to the head of the inverse dielectric matrix (inverse dielectric
        function)"""
        
        self.chi0.set_response('density')
        
        pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)

        N_c = self.chi0.calc.wfs.kd.N_c
        if self.truncation == 'wigner-seitz':
            self.wstc = WignerSeitzTruncatedCoulomb(finepd.gd.cell_cv, N_c)
        else:
            self.wstc = None
        K_G = get_coulomb_kernel(finepd,
                                 N_c,
                                 truncation=self.truncation,
                                 wstc=self.wstc,
                                 q_v=q_v)**0.5
        nG = len(K_G)

        if finepd.kd.gamma:
            if isinstance(direction, str):
                d_v = {'x': [1, 0, 0],
                       'y': [0, 1, 0],
                       'z': [0, 0, 1]}[direction]
            else:
                d_v = direction

            d_v = np.asarray(d_v) / np.linalg.norm(d_v)
            W = slice(self.w1, self.w2)
            if add_intraband:
                chi0_wGG[:, 0] = np.dot(d_v, chi0_wxvG[W, 0])
                chi0_wGG[:, :, 0] = np.dot(d_v, chi0_wxvG[W, 1])
                chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0_wvv[W], d_v).T)
            if q_v is not None:
                print('Restoring q dependence of head and wings of chi0')
                chi0_wGG[:, 1:, 0] *= np.dot(q_v, d_v)
                chi0_wGG[:, 0, 1:] *= np.dot(q_v, d_v)
                chi0_wGG[:, 0, 0] *= np.dot(q_v, d_v)**2

        if xc != 'RPA':
            Kxc_sGG = get_xc_kernel(finepd,
                                    self.chi0,
                                    functional=xc,
                                    chi0_wGG=chi0_wGG)

        if calculate_chi:
            chi_wGG = []

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
                K_GG = (K_G**2 * np.ones([nG, nG])).T
                e_GG = np.eye(nG) - P_GG * K_GG
            if calculate_chi:
                K_GG = np.diag(K_G**2)
                if xc != 'RPA':
                    K_GG += Kxc_sGG[0]
                chi_wGG.append(np.dot(np.linalg.inv(np.eye(nG) -
                                                    np.dot(chi0_GG, K_GG)),
                                      chi0_GG))
            chi0_GG[:] = e_GG

        # chi0_wGG is now the dielectric matrix
        if not calculate_chi:
            return chi0_wGG
        else:
            # chi_wGG is the full density response function..
            return pd, finepd, chi0_wGG, np.array(chi_wGG)

    def get_dielectric_function(self, xc='RPA', q_c=[0, 0, 0], q_v=None,
                                direction='x', filename='df.csv'):
        """Calculate the dielectric function.

        Returns dielectric function without and with local field correction:
        df_NLFC_w, df_LFC_w = SpinChargeResponseFunction.get_dielectric_function()
        """
        e_wGG = self.get_dielectric_matrix(xc, q_c, direction, q_v=q_v)
        df_NLFC_w = np.zeros(len(e_wGG), dtype=complex)
        df_LFC_w = np.zeros(len(e_wGG), dtype=complex)

        for w, e_GG in enumerate(e_wGG):
            df_NLFC_w[w] = e_GG[0, 0]
            df_LFC_w[w] = 1 / np.linalg.inv(e_GG)[0, 0]

        df_NLFC_w = self.collect(df_NLFC_w)
        df_LFC_w = self.collect(df_LFC_w)

        if filename is not None and mpi.rank == 0:
            with open(filename, 'w') as fd:
                for omega, nlfc, lfc in zip(self.omega_w * Hartree,
                                            df_NLFC_w,
                                            df_LFC_w):
                    print('%.6f, %.6f, %.6f, %.6f, %.6f' %
                          (omega, nlfc.real, nlfc.imag, lfc.real, lfc.imag),
                          file=fd)

        return df_NLFC_w, df_LFC_w

    def get_macroscopic_dielectric_constant(self, xc='RPA', direction='x', q_v=None):
        """Calculate macroscopic dielectric constant.

        Returns eM_NLFC and eM_LFC.

        Macroscopic dielectric constant is defined as the real part
        of dielectric function at w=0.

        Parameters:

        eM_LFC: float
            Dielectric constant without local field correction. (RPA, ALDA)
        eM2_NLFC: float
            Dielectric constant with local field correction.
        """

        fd = self.chi0.fd
        print('', file=fd)
        print('%s Macroscopic Dielectric Constant:' % xc, file=fd)

        df_NLFC_w, df_LFC_w = self.get_dielectric_function(
            xc=xc,
            filename=None,
            direction=direction,
            q_v=q_v)
        eps0 = np.real(df_NLFC_w[0])
        eps = np.real(df_LFC_w[0])
        print('  %s direction' % direction, file=fd)
        print('    Without local field: %f' % eps0, file=fd)
        print('    Include local field: %f' % eps, file=fd)

        return eps0, eps

    def get_eels_spectrum(self, xc='RPA', q_c=[0, 0, 0],
                          direction='x', filename='eels.csv'):
        """Calculate EELS spectrum. By default, generate a file 'eels.csv'.

        EELS spectrum is obtained from the imaginary part of the
        density response function as, EELS(\omega) = - 4 * \pi / q^2 Im \chi.
        Returns EELS spectrum without and with local field corrections:

        df_NLFC_w, df_LFC_w = SpinChargeResponseFunction.get_eels_spectrum()
        """
        
        self.chi0.set_response('density')
        
        # Calculate V^1/2 \chi V^1/2
        pd, finepd, Vchi0_wGG, Vchi_wGG = self.get_chi(xc=xc, q_c=q_c,
                                                       direction=direction)
        Nw = self.omega_w.shape[0]

        # Calculate eels = -Im 4 \pi / q^2  \chi
        eels_NLFC_w = -(1. / (1. - Vchi0_wGG[:, 0, 0])).imag
        eels_LFC_w = - (Vchi_wGG[:, 0, 0]).imag

        # Collect frequencies
        eels_NLFC_w = self.collect(eels_NLFC_w)
        eels_LFC_w = self.collect(eels_LFC_w)

        # Write to file
        if filename is not None and mpi.rank == 0:
            fd = open(filename, 'w')
            print('# energy, eels_NLFC_w, eels_LFC_w', file=fd)
            for iw in range(Nw):
                print('%.6f, %.6f, %.6f' %
                      (self.chi0.omega_w[iw] * Hartree,
                       eels_NLFC_w[iw], eels_LFC_w[iw]), file=fd)
            fd.close()

        return eels_NLFC_w, eels_LFC_w

    def get_polarizability(self, xc='RPA', direction='x', q_c=[0, 0, 0],
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

        if not self.truncation:
            """Standard expression for the polarizability"""
            df0_w, df_w = self.get_dielectric_function(xc=xc,
                                                       q_c=q_c,
                                                       filename=None,
                                                       direction=direction)
            alpha_w = V * (df_w - 1.0) / (4 * pi)
            alpha0_w = V * (df0_w - 1.0) / (4 * pi)
        else:
            """Since eps_M = 1.0 for a truncated Coulomb interaction, it does
            not make sense to apply it here. Instead one should define the
            polarizability by

                alpha * eps_M^{-1} = -L / (4 * pi) * <v_ind>

            where <v_ind> = 4 * pi * \chi / q^2 is the averaged induced
            potential (relative to the strength of the  external potential).
            With the bare Coulomb potential, this expression is equivalent to
            the standard one. In a 2D system \chi should be calculated with a
            truncated Coulomb potential and eps_M = 1.0"""
            
            self.chi0.set_response('density')
            
            print('Using truncated Coulomb interaction', file=self.chi0.fd)

            pd, finepd, chi0_wGG, chi_wGG = self.get_chi(xc=xc,
                                                         q_c=q_c,
                                                         direction=direction)
            alpha_w = -V * (chi_wGG[:, 0, 0]) / (4 * pi)
            alpha0_w = -V * (chi0_wGG[:, 0, 0]) / (4 * pi)

            alpha_w = self.collect(alpha_w)
            alpha0_w = self.collect(alpha0_w)

        Nw = len(alpha_w)
        if filename is not None and mpi.rank == 0:
            fd = open(filename, 'w')
            for iw in range(Nw):
                print('%.6f, %.6f, %.6f, %.6f, %.6f' %
                      (self.chi0.omega_w[iw] * Hartree,
                       alpha0_w[iw].real * Bohr**(sum(~pbc_c)),
                       alpha0_w[iw].imag * Bohr**(sum(~pbc_c)),
                       alpha_w[iw].real * Bohr**(sum(~pbc_c)),
                       alpha_w[iw].imag * Bohr**(sum(~pbc_c))), file=fd)
            fd.close()

        return alpha0_w * Bohr**(sum(~pbc_c)), alpha_w * Bohr**(sum(~pbc_c))

    def check_sum_rule(self, spectrum=None):
        """Check f-sum rule.

        It takes the y of a spectrum as an entry and it check its integral.

        spectrum: np.ndarray
            Input spectrum

        Note: not tested for spin response
        """

        assert (self.omega_w[1:] - self.omega_w[:-1]).ptp() < 1e-10

        fd = self.chi0.fd

        if spectrum is None:
            raise ValueError('No spectrum input ')
        dw = self.chi0.omega_w[1] - self.chi0.omega_w[0]
        N1 = 0
        for iw in range(len(spectrum)):
            w = iw * dw
            N1 += spectrum[iw] * w
        N1 *= dw * self.chi0.vol / (2 * pi**2)

        print('', file=fd)
        print('Sum rule:', file=fd)
        nv = self.chi0.calc.wfs.nvalence
        print('N1 = %f, %f  %% error' % (N1, (N1 - nv) / nv * 100), file=fd)

    def get_eigenmodes(self, q_c=[0, 0, 0], w_max=None, name=None,
                       eigenvalue_only=False, direction='x',
                       checkphase=True):

        """Plasmon eigenmodes as eigenvectors of the dielectric matrix.

        Note: not implemented for spin response
        """
        
        self.chi0.set_response('density')
        
        assert self.chi0.world.size == 1

        pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)
        e_wGG = self.get_dielectric_matrix(xc='RPA', q_c=q_c,
                                           direction=direction,
                                           symmetric=False)

        kd = finepd.kd

        # Get real space grid for plasmon modes:
        r = finepd.gd.get_grid_point_coordinates()
        w_w = self.omega_w * Hartree
        if w_max:
            w_w = w_w[np.where(w_w < w_max)]
        Nw = len(w_w)
        nG = e_wGG.shape[1]

        eig = np.zeros([Nw, nG], dtype=complex)
        eig_all = np.zeros([Nw, nG], dtype=complex)

        # Find eigenvalues and eigenvectors:
        e_GG = e_wGG[0]
        eig_all[0], vec = np.linalg.eig(e_GG)
        eig[0] = eig_all[0]
        vec_dual = np.linalg.inv(vec)
        omega0 = np.array([])
        eigen0 = np.array([], dtype=complex)
        v_ind = np.zeros([0, r.shape[1], r.shape[2], r.shape[3]],
                         dtype=complex)
        n_ind = np.zeros([0, r.shape[1], r.shape[2], r.shape[3]],
                         dtype=complex)

        # Loop to find the eigenvalues that crosses zero
        # from negative to positive values:
        for i in np.array(range(1, Nw)):
            e_GG = e_wGG[i]  # epsilon_GG'(omega + d-omega)
            eig_all[i], vec_p = np.linalg.eig(e_GG)
            if eigenvalue_only:
                continue
            vec_dual_p = np.linalg.inv(vec_p)
            overlap = np.abs(np.dot(vec_dual, vec_p))
            index = list(np.argsort(overlap)[:, -1])
            if len(np.unique(index)) < nG:  # add missing indices
                addlist = []
                removelist = []
                for j in range(nG):
                    if index.count(j) < 1:
                        addlist.append(j)
                    if index.count(j) > 1:
                        for l in range(1, index.count(j)):
                            removelist.append(
                                np.argwhere(np.array(index) == j)[l])
                for j in range(len(addlist)):
                    index[removelist[j]] = addlist[j]

            vec = vec_p[:, index]
            vec_dual = vec_dual_p[index, :]
            eig[i] = eig_all[i, index]
            for k in [k for k in range(nG)
                      # Eigenvalue crossing:
                      if (eig[i - 1, k] < 0 and eig[i, k] > 0)]:
                a = np.real((eig[i, k] - eig[i - 1, k]) /
                            (w_w[i] - w_w[i - 1]))
                # linear interp for crossing point
                w0 = np.real(-eig[i - 1, k]) / a + w_w[i - 1]
                eig0 = a * (w0 - w_w[i - 1]) + eig[i - 1, k]
                print('crossing found at w = %1.2f eV' % w0)
                omega0 = np.append(omega0, w0)
                eigen0 = np.append(eigen0, eig0)

                # Fourier Transform:
                qG = finepd.get_reciprocal_vectors(add_q=True)
                coef_G = np.diagonal(np.inner(qG, qG)) / (4 * pi)
                qGr_R = np.inner(qG, r.T).T
                factor = np.exp(1j * qGr_R)
                v_temp = np.dot(factor, vec[:, k])
                n_temp = np.dot(factor, vec[:, k] * coef_G)
                if checkphase:  # rotate eigenvectors in complex plane
                    integral = np.zeros([81])
                    phases = np.linspace(0, 2, 81)
                    for ip in range(81):
                        v_int = v_temp * np.exp(1j * pi * phases[ip])
                        integral[ip] = abs(np.imag(v_int)).sum()
                    phase = phases[np.argsort(integral)][0]
                    v_temp *= np.exp(1j * pi * phase)
                    n_temp *= np.exp(1j * pi * phase)
                v_ind = np.append(v_ind, v_temp[np.newaxis, :], axis=0)
                n_ind = np.append(n_ind, n_temp[np.newaxis, :], axis=0)

        kd = self.chi0.calc.wfs.kd
        if name is None and self.name:
            name = (self.name + '%+d%+d%+d-eigenmodes.pckl' %
                    tuple((q_c * kd.N_c).round()))
        elif name:
            name = (name + '%+d%+d%+d-eigenmodes.pckl' %
                    tuple((q_c * kd.N_c).round()))
        else:
            name = '%+d%+d%+d-eigenmodes.pckl' % tuple((q_c * kd.N_c).round())

        # Returns: real space grid, frequency grid,
        # sorted eigenvalues, zero-crossing frequencies + eigenvalues,
        # induced potential + density in real space.
        if eigenvalue_only:
            pickle.dump((r * Bohr, w_w, eig),
                        open(name, 'wb'), pickle.HIGHEST_PROTOCOL)
            return r * Bohr, w_w, eig
        else:
            pickle.dump((r * Bohr, w_w, eig, omega0, eigen0,
                         v_ind, n_ind), open(name, 'wb'),
                        pickle.HIGHEST_PROTOCOL)
            return r * Bohr, w_w, eig, omega0, eigen0, v_ind, n_ind

    def get_spatial_eels(self, q_c=[0, 0, 0], direction='x',
                         w_max=None, filename='eels', r=None, perpdir=None):
        """Spatially resolved loss spectrum.

        The spatially resolved loss spectrum is calculated as the inverse
        fourier transform of ``VChiV = (eps^{-1}-I)V``::

            EELS(w,r) = - Im [sum_{G,G'} e^{iGr} Vchi_{GG'}(w) V_G'e^{-iG'r}]
                          \delta(w-G\dot v_e )

        Input parameters:

        direction: 'x', 'y', or 'z'
            The direction for scanning acroos the structure
            (perpendicular to the electron beam) .
        w_max: float
            maximum frequency
        filename: str
            name of output

        Returns: real space grid, frequency points, EELS(w,r)
        """
        
        self.chi0.set_response('density')
        
        assert self.chi0.world.size == 1

        pd, finepd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)
        e_wGG = self.get_dielectric_matrix(xc='RPA', q_c=q_c,
                                           symmetric=False)

        if r is None:
            r = finepd.gd.get_grid_point_coordinates()
            ix = r.shape[1] // 2 * 0
            iy = r.shape[2] // 2 * 0
            iz = r.shape[3] // 2
            if direction == 'x':
                r = r[:, :, iy, iz]
                perpdir = [1, 2]
            if direction == 'y':
                r = r[:, ix, :, iz]
                perpdir = [0, 2]
            if direction == 'z':
                r = r[:, ix, iy, :]
                perpdir = [0, 1]

        nG = e_wGG.shape[1]
        Gvec = finepd.G_Qv[finepd.Q_qG[0]]
        Glist = []

        # Only use G-vectors that are zero along electron beam
        # due to \delta(w-G\dot v_e )
        q_v = finepd.K_qv[0]
        for iG in range(nG):
            if perpdir is not None:
                if Gvec[iG, perpdir[0]] == 0 and Gvec[iG, perpdir[1]] == 0:
                    Glist.append(iG)
            elif not np.abs(np.dot(q_v, Gvec[iG])) < \
                 np.linalg.norm(q_v) * np.linalg.norm(Gvec[iG]):
                Glist.append(iG)
        qG = Gvec[Glist] + finepd.K_qv

        w_w = self.omega_w * Hartree
        if w_max:
            w_w = w_w[np.where(w_w < w_max)]
        Nw = len(w_w)
        qGr = np.inner(qG, r.T).T
        phase = np.exp(1j * qGr)
        V_G = (4 * pi) / np.diagonal(np.inner(qG, qG))
        phase2 = np.exp(-1j * qGr) * V_G
        E_wrr = np.zeros([Nw, r.shape[1], r.shape[1]])
        E_wr = np.zeros([Nw, r.shape[1]])
        Eavg_w = np.zeros([Nw], complex)
        Ec_wr = np.zeros([Nw, r.shape[1]], complex)
        for i in range(Nw):
            Vchi_GG = (np.linalg.inv(e_wGG[i]) -
                       np.eye(nG))[Glist, :][:, Glist]

            qG_G = np.sum(qG**2, axis=1)**0.5
            Eavg_w[i] = np.trace(Vchi_GG * np.diag(V_G * qG_G))

            # Fourier transform:
            E_wrr[i] = -np.imag(np.dot(np.dot(phase, Vchi_GG), phase2.T))
            E_wr[i] = np.diagonal(E_wrr[i])
            Ec_wr[i] = np.diagonal(np.dot(np.dot(phase, Vchi_GG *
                                                 np.diag(qG_G)), phase2.T))
        pickle.dump((r * Bohr, w_w, E_wr), open('%s.pickle' % filename, 'wb'),
                    pickle.HIGHEST_PROTOCOL)

        return r * Bohr, w_w, E_wr, Ec_wr, Eavg_w
