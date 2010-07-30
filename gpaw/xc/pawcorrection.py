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
                 xc,    # radial exchange-correlation object
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
                 #nspins,        # Number os spins
                 tauc_g=None,   # kinetic core energy array
                 tauct_g=None,  # pseudo kinetic core energy array
                 ):

        self.nc_g = nc_g
        self.nct_g = nct_g
        self.xc = xc
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

        #
        self.B_Lqp = np.zeros((self.Lmax, njj, nii))
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL:
            for j2, l2, L2 in jlL[i1:]:
                if j1 < j2:
                    q = j2 + j1 * nj - j1 * (j1 + 1) // 2
                else:
                    q = j1 + j2 * nj - j2 * (j2 + 1) // 2
                self.B_Lqp[:, q, p] = gaunt[L1, L2, :self.Lmax]
                p += 1
            i1 += 1
        self.B_pqL = self.B_Lqp.T.copy()

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

    def calculate(self, D_sp, dH_sp):
        if self.xc.name == 'GLLB':
            # The coefficients for GLLB-functional are evaluated elsewhere
            return self.xc.xcfunc.xc.calculate_energy_and_derivatives(
                D_sp, H_sp, a)

        nspins = len(D_sp)
        e = 0.0
        D_sLq = np.inner(D_sp, self.B_Lqp)
        v_sg = self.rgd.empty(nspins)
        type = self.xc.xckernel.type
        xc = self.xc.calculate_radial
        sign = 1
        if type == 'LDA':
            for n_qg, nc_g in [(self.n_qg, self.nc_g),
                               (self.nt_qg, self.nct_g)]:
                n_sLg = np.dot(D_sLq, n_qg)
                n_sLg[:, 0] += sqrt(4 * pi) / nspins * nc_g
                for n, Y_L in enumerate(self.Y_nL):
                    w = sign * weight_n[n]
                    v_sg[:] = 0.0
                    e += w * xc(self.rgd, n_sLg, Y_L, v_sg)
                    dH_sq = w * np.inner(v_sg * self.dv_g, n_qg)
                    dH_sp += np.inner(dH_sq, np.dot(self.B_pqL, Y_L))
                sign = -1

        elif type == 'GGA':
            for n_qg, nc_g in [(self.n_qg, self.nc_g),
                               (self.nt_qg, self.nct_g)]:
                n_sLg = np.dot(D_sLq, n_qg)
                n_sLg[:, 0] += sqrt(4 * pi) / nspins * nc_g
                dndr_sLg = np.empty_like(n_sLg)
                for s in range(nspins):
                    for n_g, dndr_g in zip(n_sLg[s], dndr_sLg[s]):
                        self.rgd.derivative(n_g, dndr_g)
                for n, Y_L in enumerate(self.Y_nL):
                    w = sign * weight_n[n]
                    v_sg[:] = 0.0
                    rnablaY_Lv = self.rnablaY_nLv[n]
                    en, rd_vsg, dedsigma_xg = xc(self.rgd, n_sLg, Y_L, v_sg,
                                                 dndr_sLg, rnablaY_Lv)
                    e += w * en
                    dH_sq = w * np.inner(v_sg * self.dv_g, n_qg)
                    dH_sp += np.inner(dH_sq, np.dot(self.B_pqL, Y_L))
                    B_pqv = np.dot(self.B_pqL, 8 * pi * w * rnablaY_Lv)
                    v_vsg = dedsigma_xg[::2] * rd_vsg
                    if nspins == 2:
                        v_vsg += 0.5 * dedsigma_xg[1] * rd_vsg[:, ::-1]
                    v_qvs = np.inner(n_qg, v_vsg * self.rgd.dr_g)
                    dH_sp += np.dot(B_pqv.reshape((len(B_pqv), -1)),
                                    v_qvs.reshape((-1, nspins))).T
                sign = -1
        else:
            dgf
            
        return e - self.Exc0
