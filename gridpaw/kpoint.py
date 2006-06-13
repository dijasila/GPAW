# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a ``KPoint`` class."""

from math import pi, sqrt, atan2, cos, sin
from cmath import exp

import Numeric as num
import LinearAlgebra as linalg

from gridpaw.utilities.blas import axpy, rk, r2k, gemm
from gridpaw.utilities.complex import cc, real
from gridpaw.utilities.lapack import diagonalize
from gridpaw.utilities import unpack
from gridpaw.utilities.timing import Timer
from gridpaw.operators import Gradient

from RandomArray import random

import sys

class KPoint:
    """Class for a singel **k**-point.

    The ``KPoint`` class takes care of all wave functions for a
    certain **k**-point and a certain spin."""
    
    def __init__(self, gd, weight, s, k, u, k_c, typecode):
        """Construct **k**-point object.

        Parameters:
         ============ =======================================================
         ``gd``       Descriptor for wave-function grid.
         ``weight``   Weight of this **k**-point.
         ``s``        Spin index: up or down (0 or 1).
         ``k``        **k**-point index.
         ``u``        Combined spin and **k**-point index.
         ``k_c``      scaled **k**-point vector (coordinates scaled to
                      [-0.5:0.5] interval).
         ``typecode`` Data type of wave functions (``Float`` or ``Complex``).
         ============ =======================================================

        Attributes:
         ============= =======================================================
         ``phase_cd``  Bloch phase-factors for translations - axis ``c=0,1,2``
                       and direction ``d=0,1``.
         ``eps_n``     Eigenvalues.
         ``f_n``       Occupation numbers.
         ``H_nn``      Hamiltonian matrix.
         ``S_nn``      Overlap matrix.
         ``psit_nG``   Wave functions.
         ``Htpsit_nG`` Pseudo-part of the Hamiltonian applied to the wave
                       functions.
         ``timer``     ``Timer`` object.
         ``nbands``    Number of bands.
         ============= =======================================================

        Parallel stuff:
         ======== =======================================================
         ``comm`` MPI-communicator for domain.
         ``root`` Rank of the CPU that does the matrix diagonalization of
                  ``H_nn`` and the Cholesky decomposition of ``S_nn``.
         ======== =======================================================
        """

        self.gd = gd
        self.weight = weight
        self.typecode = typecode
        
        self.phase_cd = num.ones((3, 2), num.Complex)
        if typecode == num.Float:
            # Gamma-point calculation:
            self.k_c = None
        else:
            sdisp_cd = self.gd.domain.sdisp_cd
            for c in range(3):
                for d in range(2):
                    self.phase_cd[c, d] = exp(2j * pi *
                                              sdisp_cd[c, d] * k_c[c])
            self.k_c = k_c

        self.s = s  # spin index
        self.k = k  # k-point index
        self.u = u  # combined spin and k-point index

        # Which CPU does overlap-matrix Cholesky-decomposition and
        # Hamiltonian-matrix diagonalization?
        self.comm = self.gd.comm
        self.root = u % self.comm.size
        
        self.psit_nG = None
        self.Htpsit_nG = None

        self.timer = Timer()
        
    def allocate(self, nbands):
        """Allocate arrays."""
        self.nbands = nbands
        self.eps_n = num.zeros(nbands, num.Float)
        self.f_n = num.ones(nbands, num.Float) * self.weight
        self.H_nn = num.zeros((nbands, nbands), self.typecode)
        self.S_nn = num.zeros((nbands, nbands), self.typecode)

    def diagonalize(self, kin, vt_sG, my_nuclei, nbands, exx):
        """Subspace diagonalization of wave functions.

        First, the Hamiltonian (defined by ``kin``, ``vt_sG``, and
        ``my_nuclei``) is applied to the wave functions, then the
        ``H_nn`` matrix is calculated and diagonalized, and finally,
        the wave functions are rotated.  Also the projections
        ``P_uni`` (an attribute of the nuclei) are rotated.

        If this is the first iteration and we are starting from atomic
        orbitals, then the desired number of bands (``nbands``) will
        most likely differ from the number of current atomic orbitals
        (``self.nbands``).  If this is the case, then new arrays are
        allocated:

        * Too many bands: The bands with the lowest eigenvalues are
          used.
        * Too few bands: Extra wave functions calculated as the
          derivative of the wave functions with respect to the
          *x*-coordinate.
        """

        kin.apply(self.psit_nG, self.Htpsit_nG, self.phase_cd)
        self.Htpsit_nG += self.psit_nG * vt_sG[self.s]
        if exx is not None:
            exx.adjust_hamiltonian(psit_nG, self.Htpsit_nG, nbands, self.f_n,
                                   self.u, self.s)
        r2k(0.5 * self.gd.dv, self.psit_nG, self.Htpsit_nG, 0.0, self.H_nn)
        # XXX Do EXX here XXX
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            self.H_nn += num.dot(P_ni, num.dot(unpack(nucleus.H_sp[self.s]),
                                               cc(num.transpose(P_ni))))
            if exx is not None:
                exx.adjust_hamitonian_matrix(self.H_nn, P_ni, nucleus, self.s)

        self.comm.sum(self.H_nn, self.root)

        if self.comm.rank == self.root:
            info = diagonalize(self.H_nn, self.eps_n)
            if info != 0:
                raise RuntimeError, 'Very Bad!!'
        
        self.comm.broadcast(self.H_nn, self.root)
        self.comm.broadcast(self.eps_n, self.root)

        # Rotate psit_nG:
        # We should block this so that we can use a smaller temp !!!!!
        temp = num.array(self.psit_nG)
        gemm(1.0, temp, self.H_nn, 0.0, self.psit_nG)
        
        # Rotate Htpsit_nG:
        temp[:] = self.Htpsit_nG
        gemm(1.0, temp, self.H_nn, 0.0, self.Htpsit_nG)
        
        # Rotate P_ani:
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            temp_ni = P_ni.copy()
            gemm(1.0, temp_ni, self.H_nn, 0.0, P_ni)
        
        if nbands != self.nbands:
            nao = self.nbands  # number of atomic orbitals
            nmin = min(nao, nbands)
            
            tmp_nG = self.psit_nG
            self.psit_nG = self.gd.new_array(nbands, self.typecode)
            self.psit_nG[:nmin] = tmp_nG[:nmin]

            tmp_nG = self.Htpsit_nG
            self.Htpsit_nG = self.gd.new_array(nbands, self.typecode)
            self.Htpsit_nG[:nmin] = tmp_nG[:nmin]
            del tmp_nG

            tmp_n = self.eps_n
            self.allocate(nbands)
            self.eps_n[:nmin] = tmp_n[:nmin]

            extra = nbands - nao
            if extra > 0:
                self.eps_n[nao:] = self.eps_n[nao - 1] + 0.5
                slice_nG = self.psit_nG[nao:]
                ddx = Gradient(self.gd, 0, typecode=self.typecode).apply
                ddx(self.psit_nG[:extra], slice_nG, self.phase_cd)
        
    def calculate_residuals(self, pt_nuclei, converge_all=False):
        """Calculate wave function residuals.

        On entry, ``Htpsit_nG`` contains the soft part of the
        Hamiltonian applied to the wave functions.  After this call,
        ``Htpsit_nG`` holds the residuals::

          ^  ~        ^  ~   
          H psi - eps S psi =
                                _ 
              ~  ~         ~   \   ~a    a           a     ~a   ~
              H psi - eps psi + )  p  (dH    - eps dS    )<p  |psi>
                               /_   i1   i1i2        i1i2   i2
                              ai1i2

                                
        The size of the residuals is returned.
        
        Parameters:
        ================ ====================================================
        ``pt_nuclei``    ?
        ``converge_all`` flag to converge all wave functions or just occupied
        ================ ====================================================
        """
        
        R_nG = self.Htpsit_nG
        # optimize XXX 
        for R_G, eps, psit_G in zip(R_nG, self.eps_n, self.psit_nG):
            R_G -= eps * psit_G

        for nucleus in pt_nuclei:
            nucleus.adjust_residual(R_nG, self.eps_n, self.s, self.u, self.k)

        error = 0.0
        for R_G, f in zip(R_nG, self.f_n):
            weight = f
            if converge_all: weight = 1.
            error += weight * real(num.vdot(R_G, R_G))

        return error
        
    def orthonormalize(self, my_nuclei):
        """Orthogonalize wave functions."""
        S_nn = self.S_nn

        # Fill in the lower triangle:
        rk(self.gd.dv, self.psit_nG, 0.0, S_nn)
        
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            S_nn += num.dot(P_ni,
                            cc(num.innerproduct(nucleus.setup.O_ii, P_ni)))
        
        self.comm.sum(S_nn, self.root)

        if self.comm.rank == self.root:
            # inverse returns a non-contigous matrix - grrrr!  That is
            # why there is a copy.  Should be optimized with a
            # different lapack call to invert a triangular matrix XXXXX
            S_nn[:] = linalg.inverse(
                linalg.cholesky_decomposition(S_nn)).copy()

        self.comm.broadcast(S_nn, self.root)

        # This step will overwrite the Htpsit_nG array!
        gemm(1.0, self.psit_nG, S_nn, 0.0, self.Htpsit_nG)
        self.psit_nG, self.Htpsit_nG = self.Htpsit_nG, self.psit_nG  # swap

        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            gemm(1.0, P_ni.copy(), S_nn, 0.0, P_ni)

    def add_to_density(self, nt_G):
        """Add contribution to pseudo electron-density."""
        if self.typecode is num.Float:
            for psit_G, f in zip(self.psit_nG, self.f_n):
                nt_G += f * psit_G**2
        else:
            for psit_G, f in zip(self.psit_nG, self.f_n):
                nt_G += f * (psit_G * num.conjugate(psit_G)).real
                
    def rmm_diis(self, pt_nuclei, preconditioner, kin, vt_sG):
        """Improve the wave functions.

        Take two steps along the preconditioned residuals.  Step
        lengths are optimized for the first step and reused for the
        seconf."""
        
        vt_G = vt_sG[self.s]
        for n in range(self.nbands):
            R_G = self.Htpsit_nG[n]

            dR_G = num.zeros(R_G.shape, self.typecode)

            pR_G = preconditioner(R_G, self.phase_cd, self.psit_nG[n],
                                  self.k_c)
            
            kin.apply(pR_G, dR_G, self.phase_cd)

            dR_G += vt_G * pR_G

            dR_G -= self.eps_n[n] * pR_G

            for nucleus in pt_nuclei:
                nucleus.adjust_residual2(pR_G, dR_G, self.eps_n[n],
                                         self.s, self.k)
            
            RdR = self.comm.sum(real(num.vdot(R_G, dR_G)))
            dRdR = self.comm.sum(real(num.vdot(dR_G, dR_G)))
            lam = -RdR / dRdR

            R_G *= 2.0 * lam
            axpy(lam**2, dR_G, R_G)  # R_G += lam**2 * dR_G
            self.psit_nG[n] += preconditioner(R_G, self.phase_cd,
                                              self.psit_nG[n], self.k_c)

    def rmm_diis2(self, pt_nuclei, preconditioner, kin, vt_sG):
        """This is just to test the apply_h and apply_s routines
           result has to the same as with rmm_diis"""

        Htpsi_G = self.gd.new_array(typecode=self.typecode)
        Spsi_G = self.gd.new_array(typecode=self.typecode)

        vt_G = vt_sG[self.s]
        for n in range(self.nbands):
            self.apply_h(pt_nuclei, kin, vt_sG, self.psit_nG[n], Htpsi_G)
            self.apply_s(pt_nuclei, self.psit_nG[n], Spsi_G)
            R_G = Htpsi_G - self.eps_n[n] * Spsi_G
                
            dR_G = num.zeros(R_G.shape, self.typecode)

            pR_G = preconditioner(R_G, self.phase_cd, self.psit_nG[n],
                                  self.k_c)
            
            kin.apply(pR_G, dR_G, self.phase_cd)

            dR_G += vt_G * pR_G

            dR_G -= self.eps_n[n] * pR_G

            for nucleus in pt_nuclei:
                nucleus.adjust_residual2(pR_G, dR_G, self.eps_n[n],
                                         self.s, self.k)
            
            RdR = self.comm.sum(real(num.vdot(R_G, dR_G)))
            dRdR = self.comm.sum(real(num.vdot(dR_G, dR_G)))
            lam = -RdR / dRdR

            R_G *= 2.0 * lam
            axpy(lam**2, dR_G, R_G)
##            R_G += lam**2 * dR_G
            self.psit_nG[n] += preconditioner(R_G, self.phase_cd,
                                              self.psit_nG[n], self.k_c)

    def cg(self, pt_nuclei, preconditioner, kin, vt_sG):
        """Conjugate gradient optimization of wave functions"""

        """On entering, self.Htpsit_nG contains the residuals.
        As also Htpsit is needed, it will be constructed from residuals"""

        niter = 3
        tol = 1e-14
        phi_G = self.gd.new_array(typecode=self.typecode) #Update vector
        phi_old_G = self.gd.new_array(typecode=self.typecode) #Old update vector
        Htpsi_G = self.gd.new_array(typecode=self.typecode)
        Htphi_G = self.gd.new_array(typecode=self.typecode)

        Spsi_G = self.gd.new_array(typecode=self.typecode)

        for n in range(self.nbands):
            gamma_old = 1.0
            phi_old_G[:] = 0.0
            #construct Htpsit from residual
            R_G = self.Htpsit_nG[n]
            self.apply_s(pt_nuclei, self.psit_nG[n], Spsi_G)
            Htpsi_G = R_G + self.eps_n[n] * Spsi_G
            for nit in range(niter):
                error = self.comm.sum(real(num.vdot(R_G, R_G)))
                if error < tol:
                    break
#                pR_G = R_G[:]
                pR_G = preconditioner(R_G, self.phase_cd, self.psit_nG[n],
                                  self.k_c)
                #orthogonalize pR_G to previous orbitals
                self.apply_s(pt_nuclei, pR_G, Spsi_G)
                for nn in range(n):
                    ov = self.comm.sum(num.vdot(self.psit_nG[nn],Spsi_G)*self.gd.dv)
                    pR_G -= self.psit_nG[nn] * ov
                    
                gamma = self.comm.sum(real(num.vdot(pR_G, R_G)))
                phi_G = -pR_G + gamma/gamma_old * phi_old_G
                gamma_old = gamma

                #orthonorm. phi to current band:
                self.apply_s(pt_nuclei, phi_G, Spsi_G)
                ov = self.comm.sum(num.vdot(self.psit_nG[n],Spsi_G)*self.gd.dv)
#                self.apply_s(pt_nuclei, self.psit_nG[n], Spsi_G)
#                ov = self.comm.sum(num.vdot(phi_G,Spsi_G)*self.gd.dv)
                phi_G = phi_G - self.psit_nG[n] * ov
#                phi_G -= self.psit_nG[n] * ov
# why is phi -= different from phi = phi - ??
                norm2 = self.comm.sum(real(num.vdot(phi_G,phi_G))*self.gd.dv)
                phi_G /= sqrt(norm2)

                phi_old_G = phi_G[:]

                #find optimum lin. comb of psi and phi
                a = self.eps_n[n]
                b = self.comm.sum(real(num.vdot(phi_G,Htpsi_G)))*self.gd.dv
                self.apply_h(pt_nuclei, kin, vt_sG, phi_G, Htphi_G)
                c = self.comm.sum(real(num.vdot(phi_G,Htphi_G)))*self.gd.dv
                theta = 0.5*atan2(2*b, a-c)
                #theta can correspond either to maximum or minimum of e:
                enew = a*cos(theta)**2 + c*sin(theta)**2 + b*sin(2.0*theta) 
                if ( enew - self.eps_n[n] ) > 0.00: #we were at maximum
                    theta += pi/2.0
                    enew = a*cos(theta)**2 + c*sin(theta)**2+b*sin(2.0*theta)

                self.eps_n[n] = enew
                self.psit_nG[n] = cos(theta) * self.psit_nG[n] + sin(theta) * phi_G
                Htpsi_G = cos(theta) * Htpsi_G + sin(theta) * Htphi_G
                if nit < niter - 1:
#                    self.apply_h(pt_nuclei, kin, vt_sG, self.psit_nG[n], Htpsi_G)
                    self.apply_s(pt_nuclei, self.psit_nG[n], Spsi_G)
                    R_G = Htpsi_G - self.eps_n[n] * Spsi_G
                

    def cg_old(self, pt_nuclei, preconditioner, kin, vt_sG):
        """Conjugate gradient optimization of wave functions"""

        niter = 2
        phi_G = self.gd.new_array(typecode=self.typecode) #Update vector
        phi_old_G = self.gd.new_array(typecode=self.typecode) #Old update vector
        Htpsi_G = self.gd.new_array(typecode=self.typecode)
        Spsi_G = self.gd.new_array(typecode=self.typecode)

        for n in range(self.nbands):
            gamma_old = 1.0
            phi_old_G[:] = 0.0

            for nit in range(niter):
                self.apply_h(pt_nuclei, kin, vt_sG, self.psit_nG[n], Htpsi_G)
                self.apply_s(pt_nuclei, self.psit_nG[n], Spsi_G)
                R_G = Htpsi_G - self.eps_n[n] * Spsi_G
#                pR_G = R_G[:]
                pR_G = preconditioner(R_G, self.phase_cd, self.psit_nG[n],
                                  self.k_c)
                #orthonorm ???
                self.apply_s(pt_nuclei, pR_G, Spsi_G)
                for nn in range(n):
                    ov = self.comm.sum(num.vdot(self.psit_nG[nn],Spsi_G)*self.gd.dv)
                    pR_G -= self.psit_nG[n] * ov
                    
                gamma = self.comm.sum(real(num.vdot(pR_G, R_G)))
                phi_G = -pR_G + gamma/gamma_old * phi_old_G
                gamma_old = gamma
                phi_old_G = phi_G[:]

                #orthonorm. phi to current band:
                self.apply_s(pt_nuclei, phi_G, Spsi_G)
                ov = self.comm.sum(num.vdot(self.psit_nG[n],Spsi_G)*self.gd.dv)
#                ov = self.comm.sum(num.vdot(phi_G,Spsi_G)*self.gd.dv)
                phi_G = phi_G - self.psit_nG[n] * ov
# why is phi -= different from phi = phi - ??
                norm2 = self.comm.sum(real(num.vdot(phi_G,phi_G))*self.gd.dv)
                phi_G /= sqrt(norm2)

                #find optimum lin. comb of psi and f
                a = self.eps_n[n]
                b = self.comm.sum(real(num.vdot(phi_G,Htpsi_G)))*self.gd.dv
                self.apply_h(pt_nuclei, kin, vt_sG, phi_G, Htpsi_G)
                c = self.comm.sum(real(num.vdot(phi_G,Htpsi_G)))*self.gd.dv
                theta = 0.5*atan2(2*b, a-c)
                #theta can correspond either to maximum or minimum of e:
                enew = a*cos(theta)**2 + c*sin(theta)**2 + b*sin(2.0*theta) 
                if ( enew - self.eps_n[n] ) > 0.00: #we were at maximum                    
                    theta += pi/2.0
                    enew = a*cos(theta)**2 + c*sin(theta)**2+b*sin(2.0*theta)
#                print "eigs", enew
                self.eps_n[n] = enew
                self.psit_nG[n] = cos(theta) * self.psit_nG[n] + sin(theta) * phi_G


    def davidson_block(self, pt_nuclei, preconditioner, kin, vt_sG):
        """Block davidson optimization of wave functions
           The algorithm is similar to the one in Dacapo, in each iteration
           the eigenvalues are optimized in nbands + nblock subspace consisting
           of current wave functions and nblock preconditioned residuals """

        niter = 2
        nblock = min(self.nbands, 1)
        nsub_max = self.nbands + niter * nblock
        psitemp = self.gd.new_array(nsub_max , self.typecode)
        hpsitemp = self.gd.new_array(nsub_max, self.typecode)
        spsitemp = self.gd.new_array(nsub_max, self.typecode)
        Hf_nn = num.zeros((nsub_max, nsub_max), self.typecode)
        Sf_nn = num.zeros((nsub_max, nsub_max), self.typecode) 
        eps_n = num.zeros(nsub_max, num.Float)

        #initial hpsi and spsi
        psitemp[:self.nbands] = self.psit_nG[:]
        for n in range(self.nbands):
            self.apply_h(pt_nuclei, kin, vt_sG, self.psit_nG[n], hpsitemp[n])
            self.apply_s(pt_nuclei, self.psit_nG[n], spsitemp[n])

        ndone = 0
        while ndone < self.nbands:
            nstart = ndone
            nend = min(nstart + nblock, self.nbands)
            block_size = nend - nstart

            psitemp[self.nbands:]= 0.0
            hpsitemp[self.nbands:] = 0.0
            spsitemp[self.nbands:] = 0.0
            for nit in range(niter):
                for nbl in range(block_size):
                    R_G = hpsitemp[nstart + nbl] - self.eps_n[nstart + nbl] * spsitemp[nstart + nbl]
                    pR_G = preconditioner(R_G, self.phase_cd, self.psit_nG[nstart + nbl],
                                          self.k_c)
                    norm2 = self.comm.sum(real(num.vdot(pR_G,pR_G))*self.gd.dv)
                    pR_G /= sqrt(norm2)
                    psitemp[self.nbands + nit * block_size + nbl] = pR_G
                    self.apply_h(pt_nuclei, kin, vt_sG,pR_G, hpsitemp[self.nbands + nit * block_size + nbl])
                    self.apply_s(pt_nuclei, pR_G, spsitemp[self.nbands + nit * block_size + nbl])

                nsub = self.nbands + (nit + 1) * block_size
#                H_nn = Hf_nn[:nsub,:nsub]
#                S_nn = Sf_nn[:nsub,:nsub]
                H_nn = num.zeros((nsub, nsub), self.typecode)
                S_nn = num.zeros((nsub, nsub), self.typecode) 
                r2k(0.5 * self.gd.dv, psitemp[:nsub], hpsitemp[:nsub], 0.0, H_nn)
                r2k(0.5 * self.gd.dv, psitemp[:nsub], spsitemp[:nsub], 0.0, S_nn)

                self.comm.sum(H_nn, self.root)
                self.comm.sum(S_nn, self.root)

#                fooeig = linalg.eigenvalues(S_nn)
#                print "Seig", fooeig

#                yield None
        
                if self.comm.rank == self.root:
                    info = diagonalize(H_nn, eps_n, S_nn)
                    if info != 0:
                        print "Diagonalize returned", info
                        raise RuntimeError, 'Very Bad!!'
        
#                yield None
        
                self.comm.broadcast(H_nn, self.root)
                self.comm.broadcast(eps_n, self.root)

                self.eps_n[:] = eps_n[:self.nbands]
#                print "eigs", self.eps_n
                temp = num.array(psitemp)
                gemm(1.0, temp, H_nn, 0.0, psitemp)
                self.psit_nG[:] = psitemp[:self.nbands]

                temp[:] = hpsitemp
                gemm(1.0, temp, H_nn, 0.0, hpsitemp)
                temp[:] = spsitemp
                gemm(1.0, temp, H_nn, 0.0, spsitemp)

            ndone += block_size

            yield None


    def davidson2(self, pt_nuclei, preconditioner, kin, vt_sG):

        niter = 3

        nsub_max = self.nbands + niter
        psitemp = self.gd.new_array(nsub_max, self.typecode)
        hpsitemp = self.gd.new_array(nsub_max, self.typecode)
        spsitemp = self.gd.new_array(nsub_max, self.typecode)
        H_nn = num.zeros((nsub_max, nsub_max), self.typecode)
        S_nn = num.zeros((nsub_max, nsub_max), self.typecode) 
        eps_n = num.zeros(nsub_max, num.Float)

        psitemp[:self.nbands] = self.psit_nG[:]

        k = 0        
        while 1:
            nit = 0
            for n in range(self.nbands):
                self.apply_h(pt_nuclei, kin, vt_sG, psitemp[n], hpsitemp[n])
                self.apply_s(pt_nuclei, psitemp[n], spsitemp[n])
            r2k(0.5 * self.gd.dv, psitemp[:self.nbands], hpsitemp[:self.nbands], 0.0, H_nn[:self.nbands,:self.nbands])
            r2k(0.5 * self.gd.dv, psitemp[:self.nbands], spsitemp[:self.nbands], 0.0, S_nn[:self.nbands,:self.nbands]) 

            self.comm.sum(H_nn, self.root)
            self.comm.sum(S_nn, self.root)

            if self.comm.rank == self.root:
                info = diagonalize(H_nn[:self.nbands,:self.nbands], eps_n, S_nn[:self.nbands,:self.nbands])
            


    def davidson(self, pt_nuclei, preconditioner, kin, vt_sG):
        """Simple davidson optimization of wave functions"""


        niter = 3

        psitemp = self.gd.new_array(2 * self.nbands, self.typecode)
        hpsitemp = self.gd.new_array(2 * self.nbands, self.typecode)
        spsitemp = self.gd.new_array(2 * self.nbands, self.typecode)
        H_nn = num.zeros((2 * self.nbands, 2 * self.nbands), self.typecode)
        S_nn = num.zeros((2 * self.nbands, 2 * self.nbands), self.typecode) 
        eps_n = num.zeros(2 * self.nbands, num.Float)

        Spsi_G = self.gd.new_array(typecode=self.typecode)

        for nit in range(niter):
            psitemp[:self.nbands] = self.psit_nG[:]
            for n in range(self.nbands):
                R_G = self.Htpsit_nG[n]
                pR_G = preconditioner(R_G, self.phase_cd, self.psit_nG[n],
                                      self.k_c)
#                pR_G = R_G[:]
#orthogonalize pR_G
                self.apply_s(pt_nuclei, pR_G, Spsi_G)
                for nn in range(self.nbands + n):
                    ov = self.comm.sum(num.vdot(psitemp[nn],Spsi_G)*self.gd.dv)
                    pR_G -= psitemp[nn] * ov

                norm2 = self.comm.sum(real(num.vdot(pR_G,pR_G))*self.gd.dv)
                pR_G /= sqrt(norm2)
                psitemp[n + self.nbands] = pR_G[:]


                self.apply_h(pt_nuclei, kin, vt_sG, self.psit_nG[n], hpsitemp[n])
                self.apply_s(pt_nuclei, self.psit_nG[n], spsitemp[n])
                self.apply_h(pt_nuclei, kin, vt_sG, pR_G, hpsitemp[n + self.nbands])
                self.apply_s(pt_nuclei, pR_G, spsitemp[n + self.nbands])

            r2k(0.5 * self.gd.dv, psitemp, hpsitemp, 0.0, H_nn)
            r2k(0.5 * self.gd.dv, psitemp, spsitemp, 0.0, S_nn)

            self.comm.sum(H_nn, self.root)
            self.comm.sum(S_nn, self.root)

#            yield None
        
            if self.comm.rank == self.root:
#                info = diagonalize(H_nn, eps_n, S_nn)
                info = diagonalize(H_nn, eps_n)
                if info != 0:
                    raise RuntimeError, 'Very Bad!!'
        
#            yield None
        
            self.comm.broadcast(H_nn, self.root)
            self.comm.broadcast(eps_n, self.root)
        
#        print "eigs", self.eps_n, eps_n[:self.nbands]
            self.eps_n[:] = eps_n[:self.nbands]
            temp = num.array(psitemp)
            gemm(1.0, temp, H_nn, 0.0, psitemp)
            self.psit_nG[:] = psitemp[:self.nbands]
            temp[:] = hpsitemp
            gemm(1.0, temp, H_nn, 0.0, hpsitemp)
            temp[:] = spsitemp
            gemm(1.0, temp, H_nn, 0.0, spsitemp)
            for n in range(self.nbands):
                self.Htpsit_nG[n] = hpsitemp[n] - eps_n[n] * spsitemp[n]

#            yield None

    def create_atomic_orbitals(self, nao, nuclei, nbands):
        """Initialize the wave functions from atomic orbitals.

        Create ``nao`` atomic orbitals."""
        
        # Allocate space for wave functions, occupation numbers,
        # eigenvalues and projections:
        self.allocate(nao)
        self.psit_nG = self.gd.new_array(nao, self.typecode)
        self.Htpsit_nG = self.gd.new_array(nao, self.typecode)

        if False:
            self.psit_nG[:] = random(self.psit_nG.shape)
        else:
            # fill in the atomic orbitals:
            nao0 = 0
            for nucleus in nuclei:
                nao1 = nao0 + nucleus.get_number_of_atomic_orbitals()
                nucleus.create_atomic_orbitals(self.psit_nG[nao0:nao1], self.k)
                nao0 = nao1
            assert nao0 == nao
            # Fill remaining bands with random numbers...
            # if nao < nbands:
            #    extra = nbands - nao
            #    shape = (extra,) + self.psit_nG[0].shape
            #    print 'Making random wfs', shape
            #    self.psit_nG[nao:] = random(shape)


    def apply_h(self, pt_nuclei, kin, vt_sG, psit, Htpsi):
        """Applies the Hamiltonian to the wave function psi"""

        Htpsi[:] = 0.0
        kin.apply(psit, Htpsi, self.phase_cd)
        Htpsi += psit * vt_sG[self.s]
        
        for nucleus in pt_nuclei:
            #apply the non-local part
            nucleus.apply_hamiltonian(psit, Htpsi, self.s, self.k)

    def apply_s(self, pt_nuclei, psit, Spsi):
        """Applies the overlap operator to the wave function psi"""

        Spsi[:] = psit[:]
        for nucleus in pt_nuclei:
            #apply the non-local part
            nucleus.apply_overlap(psit, Spsi, self.k)
