# -*- coding: utf-8
"""This module calculates XC kernels for the response model.
"""
from __future__ import print_function

import numpy as np
from math import pi

from gpaw.xc import XC
from gpaw.sphere.lebedev import weight_n, R_nv
from gpaw.mpi import world, rank, size
from gpaw.io.tar import Reader
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.utilities.blas import scal
from gpaw.transformers import Transformer

from ase.utils.timing import Timer
from ase.units import Bohr, Ha


def get_xc_spin_kernel(finepd, chi0, functional='ALDA_x', xi_cut=None, density_cut=None):
    """ XC kernels for (collinear) spin polarized calculations
    Currently only ALDA kernels are implemented
    
    xi_cut: float
        cutoff spin polarization below which f_xc is evaluated in unpolarized limit (to make sure divergent terms cancel out correctly)
    density_cut: float
        cutoff density below which f_xc is set to zero
    """
    
    calc = chi0.calc
    fd = chi0.fd
    nspins = len(calc.density.nt_sG)
    assert nspins == 2
    
    if functional in ['ALDA_x', 'ALDA_X', 'ALDA', 'ALDA_ae1', 'ALDA_ae2']:  ### error finding ###
        # Adiabatic kernel
        if functional[-3:] in ['ae1', 'ae2']:
            mode = functional[-3:]
            functional = functional[:-4]
        else:
            mode = 'PAW'
        Kcalc = ALDASpinKernelCalculator(fd, mode, ecut=chi0.ecut, xi_cut=xi_cut, density_cut=density_cut)
    else:
        raise ValueError("%s spin kernel not implemented" % functional)
    
    return Kcalc(finepd, calc, functional)


def get_xc_kernel(pd, chi0, functional='ALDA', chi0_wGG=None, density_cut=None):
    """ XC kernels for spin neutral calculations
    Only density response kernels are implemented
    Factory function that calls the relevant functions below
    """

    calc = chi0.calc
    fd = chi0.fd
    nspins = len(calc.density.nt_sG)
    assert nspins == 1

    if functional[0] == 'A':
        # Standard adiabatic kernel
        print('Calculating %s kernel' % functional, file=fd)
        Kxc_sGG = calculate_Kxc(pd, calc, functional=functional, density_cut=density_cut)
    elif functional[0] == 'r':
        # Renormalized kernel
        print('Calculating %s kernel' % functional, file=fd)
        from gpaw.xc.fxc import KernelDens
        kernel = KernelDens(calc,
                            functional,
                            [pd.kd.bzk_kc[0]],
                            fd,
                            calc.wfs.kd.N_c,
                            None,
                            ecut=pd.ecut * Ha,
                            tag='',
                            timer=Timer(),
                            )
        kernel.calculate_fhxc()
        r = Reader('fhxc_%s_%s_%s_%s.gpw' %
                   ('', functional, pd.ecut * Ha, 0))
        Kxc_sGG = np.array([r.get('fhxc_sGsG')])

        v_G = 4 * np.pi / pd.G2_qG[0]
        Kxc_sGG[0] -= np.diagflat(v_G)

        if pd.kd.gamma:
            Kxc_sGG[:, 0, :] = 0.0
            Kxc_sGG[:, :, 0] = 0.0
    elif functional[:2] == 'LR':
        print('Calculating LR kernel with alpha = %s' % functional[2:],
              file=fd)
        Kxc_sGG = calculate_lr_kernel(pd, calc,
                                      alpha=float(functional[2:]))
    elif functional == 'DM':
        print('Calculating DM kernel', file=fd)
        Kxc_sGG = calculate_dm_kernel(pd, calc)
    elif functional == 'Bootstrap':
        print('Calculating Bootstrap kernel', file=fd)
        if chi0.world.rank == 0:
            chi0_GG = chi0_wGG[0]
            if chi0.world.size > 1:
                # If size == 1, chi0_GG is not contiguous, and broadcast()
                # will fail in debug mode.  So we skip it until someone
                # takes a closer look.
                chi0.world.broadcast(chi0_GG, 0)
        else:
            nG = pd.ngmax
            chi0_GG = np.zeros((nG, nG), complex)
            chi0.world.broadcast(chi0_GG, 0)
        Kxc_sGG = calculate_bootstrap_kernel(pd, chi0_GG, fd)
    else:
        raise ValueError('%s kernel not implemented' % functional)

    return Kxc_sGG


def calculate_Kxc(pd, calc, functional='ALDA', density_cut=None):
    """ALDA kernel"""

    gd = pd.gd
    npw = pd.ngmax
    nG = pd.gd.N_c
    vol = pd.gd.volume
    G_Gv = pd.get_reciprocal_vectors()
    nt_sG = calc.density.nt_sG
    R_av = calc.atoms.positions / Bohr
    setups = calc.wfs.setups
    D_asp = calc.density.D_asp

    # The soft part
    # assert np.abs(nt_sG[0].shape - nG).sum() == 0
    if functional == 'ALDA_X':
        x_only = True
        A_x = -3. / 4. * (3. / np.pi)**(1. / 3.)
        nspins = len(nt_sG)
        assert nspins == 1
        fxc_sg = nspins**(1. / 3.) * 4. / 9. * A_x * nt_sG**(-2. / 3.)
    else:
        assert len(nt_sG) == 1
        x_only = False
        fxc_sg = np.zeros_like(nt_sG)
        xc = XC(functional[1:])
        xc.calculate_fxc(gd, nt_sG, fxc_sg)

    if density_cut is not None:
        fxc_sg[np.where(nt_sG * len(nt_sG) < density_cut)] = 0.0
    
    # FFT fxc(r)
    nG0 = nG[0] * nG[1] * nG[2]
    tmp_sg = [np.fft.fftn(fxc_sg[s]) * vol / nG0 for s in range(len(nt_sG))]

    Kxc_sGG = np.zeros((len(fxc_sg), npw, npw), dtype=complex)
    for s in range(len(fxc_sg)):
        for iG, iQ in enumerate(pd.Q_qG[0]):
            iQ_c = (np.unravel_index(iQ, nG) + nG // 2) % nG - nG // 2
            for jG, jQ in enumerate(pd.Q_qG[0]):
                jQ_c = (np.unravel_index(jQ, nG) + nG // 2) % nG - nG // 2
                ijQ_c = (iQ_c - jQ_c)
                if (abs(ijQ_c) < nG // 2).all():
                    Kxc_sGG[s, iG, jG] = tmp_sg[s][tuple(ijQ_c)]

    # The PAW part
    KxcPAW_sGG = np.zeros_like(Kxc_sGG)
    dG_GGv = np.zeros((npw, npw, 3))
    for v in range(3):
        dG_GGv[:, :, v] = np.subtract.outer(G_Gv[:, v], G_Gv[:, v])

    for a, setup in enumerate(setups):
        if rank == a % size:
            rgd = setup.xc_correction.rgd
            n_qg = setup.xc_correction.n_qg
            nt_qg = setup.xc_correction.nt_qg
            nc_g = setup.xc_correction.nc_g
            nct_g = setup.xc_correction.nct_g
            Y_nL = setup.xc_correction.Y_nL
            dv_g = rgd.dv_g

            D_sp = D_asp[a]
            B_pqL = setup.xc_correction.B_pqL
            D_sLq = np.inner(D_sp, B_pqL.T)
            nspins = len(D_sp)

            f_sg = rgd.empty(nspins)
            ft_sg = rgd.empty(nspins)

            n_sLg = np.dot(D_sLq, n_qg)
            nt_sLg = np.dot(D_sLq, nt_qg)

            # Add core density
            n_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nc_g
            nt_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nct_g

            coefatoms_GG = np.exp(-1j * np.inner(dG_GGv, R_av[a]))
            for n, Y_L in enumerate(Y_nL):
                w = weight_n[n]
                f_sg[:] = 0.0
                n_sg = np.dot(Y_L, n_sLg)
                if x_only:
                    f_sg = nspins * (4 / 9.) * A_x * (nspins * n_sg)**(-2 / 3.)
                else:
                    xc.calculate_fxc(rgd, n_sg, f_sg)

                ft_sg[:] = 0.0
                nt_sg = np.dot(Y_L, nt_sLg)
                if x_only:
                    ft_sg = nspins * (4 / 9.) * (A_x
                                                 * (nspins * nt_sg)**(-2 / 3.))
                else:
                    xc.calculate_fxc(rgd, nt_sg, ft_sg)
                for i in range(len(rgd.r_g)):
                    coef_GG = np.exp(-1j * np.inner(dG_GGv, R_nv[n])
                                     * rgd.r_g[i])
                    for s in range(len(f_sg)):
                        KxcPAW_sGG[s] += w * np.dot(coef_GG,
                                                    (f_sg[s, i] -
                                                     ft_sg[s, i])
                                                    * dv_g[i]) * coefatoms_GG

    world.sum(KxcPAW_sGG)
    Kxc_sGG += KxcPAW_sGG

    if pd.kd.gamma:
        Kxc_sGG[:, 0, :] = 0.0
        Kxc_sGG[:, :, 0] = 0.0

    return Kxc_sGG / vol


class ALDAKernelCalculator:
    """ Adiabatic local density approx. kernel
    
    ALDA_x is an explicit algebraic version
    ALDA_X uses the libxc package
    """
    
    def __init__(self, fd, mode, ecut=None):
        """
          ### Below is not true atm ###
          gridref defines the refinement of the real space grid from which the kernel is calculated.
          for gridref=1 the coarse grid (n_sG) is used, for gridref=2 the fine grid (n_sg) is used.
        """
        self.fd = fd
        
        if mode == 'PAW':
            self.ae = False
            #self.gridref = 1
        else:
            self.ae = True
            #self.gridref = int(mode[-1])
        
        self.ecut = ecut
        self.functional = None
    
    def __call__(self, finepd, calc, functional):
        assert functional in ['ALDA_x', 'ALDA_X', 'ALDA']
        self.functional = functional
        add_fxc = self.add_fxc # class methods are not within the scope of the __call__ method
        
        N_c = finepd.gd.N_c
        vol = finepd.gd.volume
        npw = finepd.ngmax
        
        if self.ae:
            print('\tFinding all-electron density', file=self.fd)
            #n_sG, gd = calc.density.get_all_electron_density(atoms=calc.atoms, gridrefinement=self.gridref)
            n_sG, gd = calc.density.get_all_electron_density(atoms=calc.atoms, gridrefinement=1)
        else:
            n_sG, gd = calc.density.nt_sG, calc.density.gd
            
        nspins = len(n_sG)

        nfinegd = gd.new_descriptor(N_c=N_c)
        interpolator = Transformer(gd, nfinegd, 3)

        n_sg = nfinegd.empty(nspins)
        for s in range(nspins):
          interpolator.apply(n_sG[s], n_sg[s]) 
        
        '''
        qd = finepd.kd

        ecutmax = 0.5 * pi**2 / (nfinegd.h_cv**2).sum(1).max()
        ecut = self.ecut
        if not ecut is None and self.ecut > ecutmax:
            ecut = None

        lpd = PWDescriptor(ecut, nfinegd, complex, qd)
        '''
        
        print('\tCalculating fxc on the real space grid', file=self.fd)
        fxc_g = np.zeros(np.shape(n_sg[0]))
        add_fxc(nfinegd, n_sg, fxc_g)
            
        '''
        else:
            
            gd, lpd = pd.gd, pd
            
            print('\tCalculating fxc on the real space grid', file=self.fd)
            if self.gridref == 1:
                nt_sG = calc.density.nt_sG
                fxc_G = np.zeros(np.shape(nt_sG[0]))
                add_fxc(gd, nt_sG, fxc_G)
                nspins = len(nt_sG)
            else:
                if calc.density.nt_sg is None:
                    calc.density.interpolate_pseudo_density()
                nt_sg = calc.density.nt_sg
                fxc_G = np.zeros(np.shape(nt_sg[0]))
                add_fxc(gd, nt_sg, fxc_G)
                nspins = len(nt_sg)
        
        assert lpd.ngmax >= npw
        '''
        
        print('\tFourier transforming into reciprocal space', file=self.fd)
        nG = nfinegd.N_c
        nG0 = nG[0] * nG[1] * nG[2]
        
        tmp_g = np.fft.fftn(fxc_g) * vol / nG0
        
        Kxc_GG = np.zeros((npw, npw), dtype=complex)
        for iG, iQ in enumerate(finepd.Q_qG[0]):
            iQ_c = (np.unravel_index(iQ, nG) + nG // 2) % nG - nG // 2
            for jG, jQ in enumerate(finepd.Q_qG[0]):
                jQ_c = (np.unravel_index(jQ, nG) + nG // 2) % nG - nG // 2
                ijQ_c = (iQ_c - jQ_c)
                if (abs(ijQ_c) < nG // 2).all():
                    Kxc_GG[iG, jG] = tmp_g[tuple(ijQ_c)]
        
        if not self.ae:
            print('Rank %d reached PAW correcting the kernel' % world.rank)  ### error finding ###
            print('\tCalculating PAW corrections to the kernel', file=self.fd)
            
            G_Gv = finepd.get_reciprocal_vectors()
            R_av = calc.atoms.positions / Bohr
            setups = calc.wfs.setups
            D_asp = calc.density.D_asp
            
            KxcPAW_GG = np.zeros_like(Kxc_GG)
            dG_GGv = np.zeros((npw, npw, 3))
            for v in range(3):
                dG_GGv[:, :, v] = np.subtract.outer(G_Gv[:, v], G_Gv[:, v])
            
            for a, setup in enumerate(setups):
                #if rank == a % size: # useless to parallelize over atoms, when only having a few
                # PAW correction is evaluated on a radial grid
                rgd = setup.xc_correction.rgd
                n_qg = setup.xc_correction.n_qg
                nt_qg = setup.xc_correction.nt_qg
                nc_g = setup.xc_correction.nc_g
                nct_g = setup.xc_correction.nct_g
                Y_nL = setup.xc_correction.Y_nL
                dv_g = rgd.dv_g

                D_sp = D_asp[a]
                B_pqL = setup.xc_correction.B_pqL
                D_sLq = np.inner(D_sp, B_pqL.T)

                f_g = rgd.zeros()
                ft_g = rgd.zeros()

                n_sLg = np.dot(D_sLq, n_qg)
                nt_sLg = np.dot(D_sLq, nt_qg)
                
                # Add core density
                n_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nc_g
                nt_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nct_g
                
                coefatoms_GG = np.exp(-1j * np.inner(dG_GGv, R_av[a]))
                for n, Y_L in enumerate(Y_nL):
                    if rank == n % size:
                        w = weight_n[n]

                        f_g[:] = 0.
                        n_sg = np.dot(Y_L, n_sLg)
                        add_fxc(rgd, n_sg, f_g)
                        
                        ft_g[:] = 0.
                        nt_sg = np.dot(Y_L, nt_sLg)
                        add_fxc(rgd, nt_sg, ft_g)
                        
                        dGR_GG = np.inner(dG_GGv, R_nv[n])
                        '''
                        for i in range(len(rgd.r_g)):
                            #coef_GG = np.exp(-1j * np.inner(dG_GGv, R_nv[n])
                            #                 * rgd.r_g[i])
                            coef_GG = np.exp(-1j * dGR_GG * rgd.r_g[i])
                            
                            KxcPAW_GG += w * np.dot(coef_GG, (f_g[i]-ft_g[i]) * dv_g[i]) * coefatoms_GG
                        '''
                        
                        r_old = 1.
                        for i in range(1, len(rgd.r_g)): # dv_g[0] == 0.
                            scal(rgd.r_g[i]/r_old, dGR_GG)
                            coef_GG = np.exp(-1j * dGR_GG)
                            scal(w * (f_g[i]-ft_g[i]) * dv_g[i], coef_GG)
                            KxcPAW_GG += coef_GG * coefatoms_GG
                            r_old = rgd.r_g[i]
                            
                        print('\tn=%d completed' % n)  ### error finding ###
            
            print('Rank %d finished kernel calculation' % world.rank)  ### error finding ###
            world.sum(KxcPAW_GG)
            Kxc_GG += KxcPAW_GG
            print('Rank %d finished kernel summation' % world.rank)  ### error finding ###
        
        return Kxc_GG / vol
    
    
    def add_fxc(self, nfinegd, n_sg, fxc_g):
        raise NotImplementedError
    
    
class ALDASpinKernelCalculator(ALDAKernelCalculator):
    def __init__(self, fd, mode, ecut=None, xi_cut=None, density_cut=None):
        self.xi_cut = xi_cut
        self.density_cut = density_cut
        
        ALDAKernelCalculator.__init__(self, fd, mode, ecut)
        
        #if not self.ae: # should be evaluated on the fine grid
        #    self.gridref = 2
    
    def add_fxc(self, nfinegd, n_sg, fxc_g):
        """ Calculate fxc, using the cutoffs from input above """
        _calculate_pol_fxc = self._calculate_pol_fxc # class methods are not within the scope of the __call__ method
        _calculate_unpol_fxc = self._calculate_unpol_fxc
        
        xi_cut = self.xi_cut
        density_cut = self.density_cut
        
        # Mask small xi
        n_g, m_g = None, None
        if not xi_cut is None:
            m_g = n_sg[0] - n_sg[1]
            xi_small_g = np.abs(m_g/n_g) < xi_cut
        else:
            xi_small_g = np.full(np.shape(n_sg[0]), False, np.array(False).dtype)
            
        # Mask small n
        if density_cut:
            n_g = n_sg[0] + n_sg[1]
            n_pos_g = n_g > density_cut
        else:
            n_pos_g = np.full(np.shape(n_sg[0]), True, np.array(True).dtype)
        
        # Don't use small sigma limit if n is small
        xi_small_g = np.logical_and(xi_small_g, n_pos_g)
    
        # In small sigma limit, use unpolarized fxc
        if xi_small_g.any():
            if not density_cut: # Calculate it, if not done previously
                n_g = n_sg[0] + n_sg[1]
            fxc_g[xi_small_g] += _calculate_unpol_fxc(nfinegd, n_g)[xi_small_g]
        
        # Set fxc to zero if n is small
        all_fine_g = np.logical_and(np.invert(xi_small_g), n_pos_g)
        
        # Above both xi_cut and density_cut calculate polarized fxc
        if all_fine_g.any():
            if not xi_cut:
                m_g = n_sg[0] - n_sg[1] # Calculate it, if not done previously
            fxc_g[all_fine_g] += _calculate_pol_fxc(nfinegd, n_sg, m_g)[all_fine_g]
        
        return
    
    
    def _calculate_pol_fxc(self, nfinegd, n_sg, m_g):
        """ Calculate polarized fxc """
        
        assert np.shape(m_g) == np.shape(n_sg[0])
        
        if self.functional == 'ALDA_x':
            return - (6./np.pi)**(1./3.) * ( n_sg[0]**(1./3.) - n_sg[1]**(1./3.) ) / m_g
        else:
            v_sg = np.zeros(np.shape(n_sg))
            xc = XC(self.functional[1:])
            xc.calculate(nfinegd, n_sg, v_sg=v_sg)
                
            return (v_sg[0] - v_sg[1]) / m_g
    
    
    def _calculate_unpol_fxc(self, nfinegd, n_g):
        """ Calculate unpolarized fxc """
        if self.functional == 'ALDA_x':
            return - (3./np.pi)**(1./3.) * 2. / 3. * n_g**(-2./3.)
        else:
            raise NotImplementedError
            #n_sg = np.array([n_g]) # Needs spin channel in array
            #fxc_sg = np.zeros(np.shape(n_sg))
            #xc = XC(functional[1:])
            #xc.calculate_fxc(gd, n_sg, fxc_sg)
            #return fxc_sg[0]


'''        
        
def calculate_spin_Kxc(pd, calc, functional='ALDA_x', xi_cut=None, density_cut=None):
    """ Adiabatic kernel
    Currently only ALDA is available
    ALDA_x is an explicit version written in this script
    ALDA_X uses the libxc package
    """
    
    assert functional in ['ALDA_x', 'ALDA_X', 'ALDA', 'ALDA_ae1', 'ALDA_ae2'] ### added ae for error finding ###
    
    
    def _calculate_pol_fxc(gd, n_sg, m_g):
        """ Calculate polarized fxc """
        
        assert np.shape(m_g) == np.shape(n_sg[0])
        
        if functional == 'ALDA_x':
            return - (6./np.pi)**(1./3.) * ( n_sg[0]**(1./3.) - n_sg[1]**(1./3.) ) / m_g
        else:
            v_sg = np.zeros(np.shape(n_sg))
            xc = XC(functional[1:])
            xc.calculate(gd, n_sg, v_sg=v_sg)
                
            return (v_sg[0] - v_sg[1]) / m_g
    
    
    def _calculate_unpol_fxc(gd, n_g):
        """ Calculate unpolarized fxc """
        if functional == 'ALDA_x':
            return - (3./np.pi)**(1./3.) * 2. / 3. * n_g**(-2./3.)
        else:
            raise NotImplementedError
            #n_sg = np.array([n_g]) # Needs spin channel in array
            #fxc_sg = np.zeros(np.shape(n_sg))
            #xc = XC(functional[1:])
            #xc.calculate_fxc(gd, n_sg, fxc_sg)
            #return fxc_sg[0]
    
    
    def add_fxc(gd, n_sg, fxc_g):
        """ Calculate fxc, using the cutoffs from input above """
        
        # Mask small xi
        n_g, m_g = None, None
        if not xi_cut is None:
            m_g = n_sg[0] - n_sg[1]
            xi_small_g = np.abs(m_g/n_g) < xi_cut
        else:
            xi_small_g = np.full(np.shape(n_sg[0]), False, np.array(False).dtype)
            
        # Mask small n
        if density_cut:
            n_g = n_sg[0] + n_sg[1]
            n_pos_g = n_g > density_cut
        else:
            n_pos_g = np.full(np.shape(n_sg[0]), True, np.array(True).dtype)
        
        # Don't use small sigma limit if n is small
        xi_small_g = np.logical_and(xi_small_g, n_pos_g)
    
        # In small sigma limit, use unpolarized fxc
        if xi_small_g.any():
            if not density_cut: # Calculate it, if not done previously
                n_g = n_sg[0] + n_sg[1]
            fxc_g[xi_small_g] += _calculate_unpol_fxc(gd, n_g)[xi_small_g]
        
        # Set fxc to zero if n is small
        all_fine_g = np.logical_and(np.invert(xi_small_g), n_pos_g)
        
        # Above both xi_cut and density_cut calculate polarized fxc
        if all_fine_g.any():
            if not xi_cut:
                m_g = n_sg[0] - n_sg[1] # Calculate it, if not done previously
            fxc_g[all_fine_g] += _calculate_pol_fxc(gd, n_sg, m_g)[all_fine_g]
        
        return
    
    
    if not functional[-4:-1] == '_ae': ### added for error finding ###
        
        # gd holds real space grids == calc.wfs.gd
        gd = pd.gd
        npw = pd.ngmax
        nG = pd.gd.N_c # coarse grid points
        vol = pd.gd.volume
        # we need reciprocal lattice vector indexed by G
        G_Gv = pd.get_reciprocal_vectors()
        
        # grids from calc.density are real space grids (g/G = fine/coarse)
        nt_sG = calc.density.nt_sG
        R_av = calc.atoms.positions / Bohr
        setups = calc.wfs.setups
        D_asp = calc.density.D_asp
        
        nspins = len(nt_sG)
        assert nspins == 2
        
        # The soft part
        fxc_G = np.zeros(np.shape(nt_sG[0]))
        add_fxc(gd, nt_sG, fxc_G)
        
        # FFT fxc(r)
        nG0 = nG[0] * nG[1] * nG[2]
        tmp_g = np.fft.fftn(fxc_G) * vol / nG0
        
        Kxc_GG = np.zeros((npw, npw), dtype=complex)
        for iG, iQ in enumerate(pd.Q_qG[0]):
            iQ_c = (np.unravel_index(iQ, nG) + nG // 2) % nG - nG // 2
            for jG, jQ in enumerate(pd.Q_qG[0]):
                jQ_c = (np.unravel_index(jQ, nG) + nG // 2) % nG - nG // 2
                ijQ_c = (iQ_c - jQ_c)
                if (abs(ijQ_c) < nG // 2).all():
                    Kxc_GG[iG, jG] = tmp_g[tuple(ijQ_c)]

    
        # The PAW part
        KxcPAW_GG = np.zeros_like(Kxc_GG)
        dG_GGv = np.zeros((npw, npw, 3))
        for v in range(3):
            dG_GGv[:, :, v] = np.subtract.outer(G_Gv[:, v], G_Gv[:, v])

        for a, setup in enumerate(setups):
            if rank == a % size:
                # PAW correction is evaluated on a radial grid
                rgd = setup.xc_correction.rgd
                n_qg = setup.xc_correction.n_qg
                nt_qg = setup.xc_correction.nt_qg
                nc_g = setup.xc_correction.nc_g
                nct_g = setup.xc_correction.nct_g
                Y_nL = setup.xc_correction.Y_nL
                dv_g = rgd.dv_g

                D_sp = D_asp[a]
                B_pqL = setup.xc_correction.B_pqL
                D_sLq = np.inner(D_sp, B_pqL.T)
                nspins = len(D_sp)

                f_g = rgd.zeros()
                ft_g = rgd.zeros()

                n_sLg = np.dot(D_sLq, n_qg)
                nt_sLg = np.dot(D_sLq, nt_qg)

                # Add core density
                n_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nc_g
                nt_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nct_g

                coefatoms_GG = np.exp(-1j * np.inner(dG_GGv, R_av[a]))
                for n, Y_L in enumerate(Y_nL):
                    w = weight_n[n]
                    
                    f_g[:] = 0.
                    n_sg = np.dot(Y_L, n_sLg)
                    add_fxc(rgd, n_sg, f_g)
                    
                    ft_g[:] = 0.
                    nt_sg = np.dot(Y_L, nt_sLg)
                    add_fxc(rgd, nt_sg, ft_g)

                    for i in range(len(rgd.r_g)):
                        coef_GG = np.exp(-1j * np.inner(dG_GGv, R_nv[n])
                                         * rgd.r_g[i])

                        KxcPAW_GG += w * np.dot(coef_GG,
                                                    (f_g[i] -
                                                     ft_g[i])
                                                * dv_g[i]) * coefatoms_GG
        world.sum(KxcPAW_GG)
        Kxc_GG += KxcPAW_GG
    
    else:
        ### added for error finding ###
        gridref = int(functional[-1])
        functional = functional[:-4]
        
        if gridref == 1:
        # find all-electron density to calculate kernel directly
        n_sG, Gd = calc.density.get_all_electron_density(atoms=calc.atoms, gridrefinement=1)
        
        nspins = len(n_sG)
        assert nspins == 2
        
        fxc_G = np.zeros(np.shape(n_sG[0]))
        add_fxc(gd, n_sG, fxc_G)
        
        # FFT fxc(r)
        nG0 = nG[0] * nG[1] * nG[2]
        tmp_g = np.fft.fftn(fxc_G) * vol / nG0
        
        Kxc_GG = np.zeros((npw, npw), dtype=complex)
        for iG, iQ in enumerate(pd.Q_qG[0]):
            iQ_c = (np.unravel_index(iQ, nG) + nG // 2) % nG - nG // 2
            for jG, jQ in enumerate(pd.Q_qG[0]):
                jQ_c = (np.unravel_index(jQ, nG) + nG // 2) % nG - nG // 2
                ijQ_c = (iQ_c - jQ_c)
                if (abs(ijQ_c) < nG // 2).all():
                    Kxc_GG[iG, jG] = tmp_g[tuple(ijQ_c)]
        
        
    """  ### error finding ###
    if pd.kd.gamma:
        Kxc_GG[0, :] = 0.0
        Kxc_GG[:, 0] = 0.0
    """    
    
    return Kxc_GG / vol
'''


def calculate_lr_kernel(pd, calc, alpha=0.2):
    """Long range kernel: fxc = \alpha / |q+G|^2"""

    assert pd.kd.gamma

    f_G = np.zeros(len(pd.G2_qG[0]))
    f_G[0] = -alpha
    f_G[1:] = -alpha / pd.G2_qG[0][1:]

    return np.array([np.diag(f_G)])


def calculate_dm_kernel(pd, calc):
    """Density matrix kernel"""

    assert pd.kd.gamma

    nv = calc.wfs.setups.nvalence
    psit_nG = np.array([calc.wfs.kpt_u[0].psit_nG[n]
                        for n in range(4 * nv)])
    vol = np.linalg.det(calc.wfs.gd.cell_cv)
    Ng = np.prod(calc.wfs.gd.N_c)
    rho_GG = np.dot(psit_nG.conj().T, psit_nG) * vol / Ng**2

    maxG2 = np.max(pd.G2_qG[0])
    cut_G = np.arange(calc.wfs.pd.ngmax)[calc.wfs.pd.G2_qG[0] <= maxG2]

    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0

    Kxc_GG = np.diagflat(4 * np.pi / G_G**2)
    Kxc_GG = np.dot(Kxc_GG, rho_GG.take(cut_G, 0).take(cut_G, 1))
    Kxc_GG -= 4 * np.pi * np.diagflat(1.0 / G_G**2)

    return np.array([Kxc_GG])


def calculate_bootstrap_kernel(pd, chi0_GG, fd):
    """Bootstrap kernel PRL 107, 186401"""

    if pd.kd.gamma:
        v_G = np.zeros(len(pd.G2_qG[0]))
        v_G[0] = 4 * np.pi
        v_G[1:] = 4 * np.pi / pd.G2_qG[0][1:]
    else:
        v_G = 4 * np.pi / pd.G2_qG[0]

    nG = len(v_G)
    K_GG = np.diag(v_G)

    fxc_GG = np.zeros((nG, nG), dtype=complex)
    dminv_GG = np.zeros((nG, nG), dtype=complex)

    for iscf in range(120):
        dminvold_GG = dminv_GG.copy()
        Kxc_GG = K_GG + fxc_GG

        chi_GG = np.dot(np.linalg.inv(np.eye(nG, nG)
                                      - np.dot(chi0_GG, Kxc_GG)), chi0_GG)
        dminv_GG = np.eye(nG, nG) + np.dot(K_GG, chi_GG)

        alpha = dminv_GG[0, 0] / (K_GG[0, 0] * chi0_GG[0, 0])
        fxc_GG = alpha * K_GG
        print(iscf, 'alpha =', alpha, file=fd)
        error = np.abs(dminvold_GG - dminv_GG).sum()
        if np.sum(error) < 0.1:
            print('Self consistent fxc finished in %d iterations !' % iscf,
                  file=fd)
            break
        if iscf > 100:
            print('Too many fxc scf steps !', file=fd)

    return np.array([fxc_GG])
