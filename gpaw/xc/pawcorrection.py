# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi, sqrt

import numpy as np

from gpaw.utilities.blas import axpy, gemm, gemv, gemmdot
from gpaw import extra_parameters
from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import nablarlYL

# load points and weights for the angular integration
from gpaw.sphere.lebedev import Y_nL, R_nv, weight_n


"""
                           3
             __   dn       __   __    dY
   __  2    \       L  2  \    \        L  2
  (\/n) = (  ) Y  --- )  + ) (  )  n  --- )
            /__ L dr      /__  /__  L dr
                                        v
             L            v=1    L


        dY
          L
  A   = --- r
   Lv   dr
          v

"""
# A_nvL is defined as above, n is an expansion point index (50 Lebedev points).
rnablaY_nLv = np.empty((len(R_nv), 25, 3))
for rnablaY_Lv, Y_L, R_v in zip(rnablaY_nLv, Y_nL, R_nv):
    for l in range(5):
        for L in range(l**2, (l + 1)**2):
            rnablaY_Lv[L] = nablarlYL(L, R_v)  - l * R_v * Y_L[L]


class PAWXCCorrection:
    def __init__(self,
                 w_jg,  # all-lectron partial waves
                 wt_jg, # pseudo partial waves
                 nc_g,  # core density
                 nct_g, # smooth core density
                 rgd,   # radial grid descriptor
                 jl,    # ?
                 lmax,  # maximal angular momentum to consider
                 Exc0,  # xc energy of reference atom
                 phicorehole_g, # ?
                 fcorehole,     # ?
                 xctype,
                 tauc_g,   # kinetic core energy array
                 tauct_g   # pseudo kinetic core energy array
                 ):
        self.nc_g = nc_g
        self.nct_g = nct_g
        self.Exc0 = Exc0
        self.Lmax = (lmax + 1)**2
        self.rgd = rgd
        self.dv_g = rgd.dv_g
        #self.nspins = nspins
        self.Y_nL = Y_nL[:, :self.Lmax]
        self.rnablaY_nLv = rnablaY_nLv[:, :self.Lmax]
        self.ng = ng = len(nc_g)

        jlL = [(j, l, l**2 + m) for j, l in jl for m in range(2 * l + 1)]
        self.ni = ni = len(jlL)
        self.nj = nj = len(jl)
        self.nii = nii = ni * (ni + 1) // 2
        njj = nj * (nj + 1) // 2

        self.tauc_g = tauc_g
        self.tauct_g = tauct_g
        if xctype == 'MGGA':
            self.initialize_kinetic(jlL, jl, nii, phit_jg, phi_jg)
        else:
            self.tau_npg = None
            self.taut_npg = None
        #
        B_Lqp = np.zeros((self.Lmax, njj, nii))
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL:
            for j2, l2, L2 in jlL[i1:]:
                if j1 < j2:
                    q = j2 + j1 * nj - j1 * (j1 + 1) // 2
                else:
                    q = j1 + j2 * nj - j2 * (j2 + 1) // 2
                B_Lqp[:, q, p] = gaunt[L1, L2, :self.Lmax]
                p += 1
            i1 += 1
        self.B_pqL = B_Lqp.T.copy()

        #
        self.n_qg = np.zeros((njj, ng))
        self.nt_qg = np.zeros((njj, ng))
        q = 0
        for j1, l1 in jl:
            for j2, l2 in jl[j1:]:
                #rl1l2 = rgd.r_g**(l1 + l2)
                self.n_qg[q] = w_jg[j1] * w_jg[j2]
                self.nt_qg[q] = wt_jg[j1] * wt_jg[j2]
                q += 1

        #
        self.ncorehole_g = None
        #if nspins == 2 and fcorehole != 0.0:
        if 2 == 2 and fcorehole != 0.0:
            XXXself.ncorehole_g = fcorehole * phicorehole_g**2 / (4 * pi)

    def calculate(self, xc, D_sp, dH_sp=None):
        if dH_sp is None:
            dH_sp = np.zeros_like(D_sp)
            
        if xc.name == 'GLLB':
            # The coefficients for GLLB-functional are evaluated elsewhere
            return self.xc.xcfunc.xc.calculate_energy_and_derivatives(
                D_sp, H_sp, a)

        nspins = len(D_sp)
        de = 0.0
        D_sLq = np.inner(D_sp, self.B_pqL.T)
        v_sg = self.rgd.empty(nspins)
        type = xc.xckernel.type
        xc = xc.calculate_radial

        if type == 'MGGA':
            dedtau_sg = self.rgd.empty(nspins)
            if self.tau_pg is None:
                self.initialize_kinetic()
                print 'TODO: tau_ypg is HUGE!  There must be a better way.'

        sign = 1
        for n_qg, nc_g, tau_npg, tauc_g in [
            (self.n_qg, self.nc_g, self.tau_npg, self.tauc_g),
            (self.nt_qg, self.nct_g, self.taut_npg, self.tauct_g)]:
            n_sLg = np.dot(D_sLq, n_qg)
            n_sLg[:, 0] += sqrt(4 * pi) / nspins * nc_g
            if type != 'LDA':
                dndr_sLg = np.empty_like(n_sLg)
                for s in range(nspins):
                    for n_g, dndr_g in zip(n_sLg[s], dndr_sLg[s]):
                        self.rgd.derivative(n_g, dndr_g)
            for n, Y_L in enumerate(self.Y_nL):
                w = sign * weight_n[n]
                v_sg[:] = 0.0
                if type == 'LDA':
                    e = xc(self.rgd, n_sLg, Y_L, v_sg)
                else:
                    if type == 'GGA':
                        rnablaY_Lv = self.rnablaY_nLv[n]
                        e, rd_vsg, dedsigma_xg = xc(self.rgd, n_sLg, Y_L, v_sg,
                                                    dndr_sLg, rnablaY_Lv)
                    elif type == 'MGGA':
                        tau_sg = np.dot(D_sp, tau_ypg[n]) + tauc_g # XXX / ns?
                        e, rd_vsg, dedsigma_xg = xc(self.rgd, n_sLg, Y_L, v_sg,
                                                    dndr_sLg, rnablaY_Lv,
                                                    tau_sg, dedtau_sg)
                        dH_sp += np.inner(dedtau_sg, tau_pg)
                    else:
                        NOOOOOOOOOOOOOOOOOOOOOOOOOOO
                    B_pqv = np.dot(self.B_pqL, 8 * pi * w * rnablaY_Lv)
                    v_vsg = dedsigma_xg[::2] * rd_vsg
                    if nspins == 2:
                        v_vsg += 0.5 * dedsigma_xg[1] * rd_vsg[:, ::-1]
                    v_qvs = np.inner(n_qg, v_vsg * self.rgd.dr_g)
                    dH_sp += np.dot(B_pqv.reshape((len(B_pqv), -1)),
                                    v_qvs.reshape((-1, nspins))).T
                    
                de += w * e
                dH_sq = w * np.inner(v_sg * self.dv_g, n_qg)
                dH_sp += np.inner(dH_sq, np.dot(self.B_pqL, Y_L))
            sign = -1
            
        return de - self.Exc0

    def create_kinetic(self,jlL,jl,ny,nii,phi_jg,tau_ypg):
        """Short title here.
        
        kinetic expression is::

                                             __         __
          tau_s = 1/2 Sum_{i1,i2} D(s,i1,i2) \/phi_i1 . \/phi_i2 +tauc_s

        here the orbital dependent part is calculated::

          __         __
          \/phi_i1 . \/phi_i2 =
                      __    __
                      \/YL1.\/YL2 phi_j1 phi_j2 +YL1 YL2 dphi_j1 dphi_j2
                                                         ------  ------
                                                           dr     dr
          __    __
          \/YL1.\/YL2 [y] = Sum_c A[L1,c,y] A[L2,c,y] / r**2
          
        """
        ng = self.ng
        Lmax = self.Lmax
        nj = len(jl)
        ni = len(jlL)
        nii = ni * (ni + 1) // 2
        dphidr_jg = np.zeros(np.shape(phi_jg))
        for j in range(nj):
            phi_g = phi_jg[j]
            self.rgd.derivative(phi_g, dphidr_jg[j])
        ##second term
        for y in range(ny):
            i1 = 0
            p = 0
            Y_L = self.Y_nL[y]
            for j1, l1, L1 in jlL:
                for j2, l2, L2 in jlL[i1:]:
                    c = Y_L[L1]*Y_L[L2]
                    temp = c * dphidr_jg[j1] *  dphidr_jg[j2]
                    tau_ypg[y,p,:] += temp
                    p += 1
                i1 +=1
        ##first term
        for y in range(ny):
            i1 = 0
            p = 0
            A_Li = A_Liy[:self.Lmax, :, y]
            A_Lxg = A_Li[:, 0]
            A_Lyg = A_Li[:, 1]
            A_Lzg = A_Li[:, 2]
            for j1, l1, L1 in jlL:
                for j2, l2, L2 in jlL[i1:]:
                    temp = (A_Lxg[L1] * A_Lxg[L2] + A_Lyg[L1] * A_Lyg[L2]
                            + A_Lzg[L1] * A_Lzg[L2])
                    temp *=  phi_jg[j1] * phi_jg[j2]
                    temp[1:] /= self.rgd.r_g[1:]**2
                    temp[0] = temp[1]
                    tau_ypg[y, p, :] += temp
                    p += 1
                i1 +=1
        tau_ypg *= 0.5
                    
        return
        
    def initialize_kinetic(self, jlL, jl, nii, phit_jg, phi_jg):
        ny = len(R_nv)
        ng = self.ng

        self.tau_npg = np.zeros((ny, nii, ng))
        self.taut_npg = np.zeros((ny, nii, ng))
        self.create_kinetic(jlL,jl,ny, nii,phit_jg, self.taut_npg)
        self.create_kinetic(jlL,jl,ny, nii,phi_jg, self.tau_npg)
        tauc_g = data.tauc_g
        tauct_g = data.tauct_g
        self.tauc_g = np.array(tauc_g[:ng].copy())
        self.tauct_g = np.array(tauct_g[:ng].copy())

    def four_phi_integrals(self, D_sp, fxc):
        """Calculate four-phi integrals.

        The density is given by the density matrix ``D_sp`` in packed(pack)
        form, and the resulting rank-four tensor is also returned in
        packed format. ``fxc`` is a radial object???
        """

        ns, nii = D_sp.shape

        assert ns == 1# and not self.xc.get_functional().gga

        D_p = D_sp[0]
        D_Lq = np.dot(self.B_pqL.T, D_p)

        # Expand all-electron density in spherical harmonics:
        n_qg = self.n_qg
        n_Lg = np.dot(D_Lq, n_qg)
        n_Lg[0] += self.nc_g * sqrt(4 * pi)

        # Expand pseudo electron density in spherical harmonics:
        nt_qg = self.nt_qg
        nt_Lg = np.dot(D_Lq, nt_qg)
        nt_Lg[0] += self.nct_g * sqrt(4 * pi)

        # Allocate array for result:
        J_pp = np.zeros((nii, nii))

        # Loop over 50 points on the sphere surface:
        for w, Y_L in zip(weight_n, self.Y_nL):
            B_pq = np.dot(self.B_pqL, Y_L)

            fxcdv = fxc(np.dot(Y_L, n_Lg)) * self.dv_g
            dn2_qq = np.inner(n_qg * fxcdv, n_qg)

            fxctdv = fxc(np.dot(Y_L, nt_Lg)) * self.dv_g
            dn2_qq -= np.inner(nt_qg * fxctdv, nt_qg)

            J_pp += w * np.dot(B_pq, np.inner(dn2_qq, B_pq))

        return J_pp
