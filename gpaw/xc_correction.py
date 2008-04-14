# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi, sqrt

import numpy as npy
from numpy import dot as dot3  # Avoid dotblas bug!

from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import YL

# load points and weights for the angular integration
from gpaw.sphere import Y_nL, points, weights

from gpaw.gllb import SMALL_NUMBER

"""
                           3
             __   dn       __   __    dY
   __  2    \       L  2  \    \        L  2
  (\/n) = (  ) Y  --- )  + ) (  )  n  --- )
            /__ L dr      /__  /__  L dr
                                        c
             L            c=1    L


        dY
          L
  A   = --- r
   Lc   dr
          c

"""


A_Liy = npy.zeros((25, 3, len(points)))

y = 0
for R in points:
    for l in range(5):
        for m in range(2 * l + 1):
            L = l**2 + m
            for c, n in YL[L]:
                for i in range(3):
                    ni = n[i]
                    if ni > 0:
                        a = ni * c * R[i]**(ni - 1)
                        for ii in range(3):
                            if ii != i:
                                a *= R[ii]**n[ii]
                        A_Liy[L, i, y] += a
            A_Liy[L, :, y] -= l * R * Y_nL[y, L]
    y += 1

class XCCorrection:
    def __init__(self,
                 xc,    # radial exchange-correlation object
                 w_jg,  # all-lectron partial waves
                 wt_jg, # pseudo partial waves
                 nc_g,  # core density
                 nct_g, # smooth core density
                 rgd,   # radial grid edscriptor
                 jl,    # ?
                 lmax,  # maximal angular momentum to consider
                 Exc0,
                 phicorehole_g, fcorehole, nspins, # ?
                 tauc_g=None): # kinetic core energy array
 
        
        self.nc_g = nc_g
        self.nct_g = nct_g
        self.xc = xc
        self.Exc0 = Exc0
        self.Lmax = (lmax + 1)**2
        if lmax == 0:
            self.weights = [1.0]
            self.Y_yL = npy.array([[1.0 / sqrt(4.0 * pi)]])
        else:
            self.weights = weights
            self.Y_yL = Y_nL[:, :self.Lmax].copy()
        jlL = []
        for j, l in jl:
            for m in range(2 * l + 1):
                jlL.append((j, l, l**2 + m))

        ng = len(nc_g)
        self.ng = ng
        ni = len(jlL)
        nj = len(jl)
        np = ni * (ni + 1) // 2
        nq = nj * (nj + 1) // 2
        self.B_Lqp = npy.zeros((self.Lmax, nq, np))
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
        self.B_pqL = npy.transpose(self.B_Lqp).copy()
        self.dv_g = rgd.dv_g
        self.n_qg = npy.zeros((nq, ng))
        self.nt_qg = npy.zeros((nq, ng))
        q = 0
        for j1, l1 in jl:
            for j2, l2 in jl[j1:]:
                rl1l2 = rgd.r_g**(l1 + l2)
                self.n_qg[q] = rl1l2 * w_jg[j1] * w_jg[j2]
                self.nt_qg[q] = rl1l2 * wt_jg[j1] * wt_jg[j2]
                q += 1
        self.rgd = rgd

        self.nspins = nspins
        if nspins == 1:
            self.nc_g = nc_g
        else:
            if fcorehole == 0.0:
                self.nca_g = self.ncb_g = 0.5 * nc_g
            else:
                ncorehole_g = fcorehole * phicorehole_g**2 / (4 * pi)
                self.nca_g = 0.5 * (nc_g - ncorehole_g)
                self.ncb_g = 0.5 * (nc_g + ncorehole_g)
        
        if self.xc.get_functional().mgga:
            ny = len(points)
            self.tau_ypg = npy.zeros((ny, np, ng))
            self.taut_ypg = npy.zeros((ny, np, ng))
            self.create_kinetic(lmax,jlL,jl,ny,np,wt_jg,self.taut_ypg)
            self.create_kinetic(lmax,jlL,jl,len(points),np,w_jg,self.tau_ypg)            
            self.tauc_g = tauc_g

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a=None):
        if self.xc.get_functional().is_gllb():
            # The coefficients for GLLB-functional are evaluated elsewhere
            return 0.0

        if self.xc.get_functional().mgga:
            return self.MGGA(D_sp, H_sp)
        
        if self.xc.get_functional().gga:
            if self.xc.get_functional().uses_libxc:
                return self.GGA_libxc(D_sp, H_sp)
            else:
                return self.GGA(D_sp, H_sp)
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = dot3(self.B_Lqp, D_p)
            n_Lg = npy.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)
            nt_Lg = npy.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            for w, Y_L in zip(self.weights, self.Y_yL):
                n_g = npy.dot(Y_L, n_Lg)
                vxc_g = npy.zeros(self.ng)
                E += self.xc.get_energy_and_potential(n_g, vxc_g) * w
                dEdD_q = npy.dot(self.n_qg, vxc_g * self.dv_g)
                nt_g = npy.dot(Y_L, nt_Lg)
                vxct_g = npy.zeros(self.ng)
                E -= self.xc.get_energy_and_potential(nt_g, vxct_g) * w
                dEdD_q -= npy.dot(self.nt_qg, vxct_g * self.dv_g)
                dEdD_p += npy.dot(dot3(self.B_pqL, Y_L),
                                  dEdD_q) * w
        else: 
            Da_p = D_sp[0]
            Da_Lq = dot3(self.B_Lqp, Da_p)
            na_Lg = npy.dot(Da_Lq, self.n_qg)
            na_Lg[0] += self.nca_g * sqrt(4 * pi)
            nta_Lg = npy.dot(Da_Lq, self.nt_qg)
            nta_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0
            Db_p = D_sp[1]
            Db_Lq = dot3(self.B_Lqp, Db_p)
            nb_Lg = npy.dot(Db_Lq, self.n_qg)
            nb_Lg[0] += self.ncb_g * sqrt(4 * pi)
            ntb_Lg = npy.dot(Db_Lq, self.nt_qg)
            ntb_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            for w, Y_L in zip(self.weights, self.Y_yL):
                na_g = npy.dot(Y_L, na_Lg)
                vxca_g = npy.zeros(self.ng)
                nb_g = npy.dot(Y_L, nb_Lg)
                vxcb_g = npy.zeros(self.ng)
                E += self.xc.get_energy_and_potential(na_g, vxca_g,
                                                   nb_g, vxcb_g) * w
                dEdDa_q = npy.dot(self.n_qg, vxca_g * self.dv_g)
                dEdDb_q = npy.dot(self.n_qg, vxcb_g * self.dv_g)
                nta_g = npy.dot(Y_L, nta_Lg)
                vxcta_g = npy.zeros(self.ng)
                ntb_g = npy.dot(Y_L, ntb_Lg)
                vxctb_g = npy.zeros(self.ng)
                E -= self.xc.get_energy_and_potential(nta_g, vxcta_g,
                                                   ntb_g, vxctb_g) * w
                dEdDa_q -= npy.dot(self.nt_qg, vxcta_g * self.dv_g)
                dEdDb_q -= npy.dot(self.nt_qg, vxctb_g * self.dv_g)
                dEdDa_p += npy.dot(dot3(self.B_pqL, Y_L),
                                  dEdDa_q) * w
                dEdDb_p += npy.dot(dot3(self.B_pqL, Y_L),
                                  dEdDb_q) * w

        return E - self.Exc0

    def GGA(self, D_sp, H_sp):
        r_g = self.rgd.r_g
        xcfunc = self.xc.get_functional()
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = dot3(self.B_Lqp, D_p)
            n_Lg = npy.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)
            nt_Lg = npy.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            dndr_Lg = npy.zeros((self.Lmax, self.ng))
            dntdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(n_Lg[L], dndr_Lg[L])
                self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]
                n_g = npy.dot(Y_L, n_Lg)
                a1x_g = npy.dot(A_Li[:, 0], n_Lg)
                a1y_g = npy.dot(A_Li[:, 1], n_Lg)
                a1z_g = npy.dot(A_Li[:, 2], n_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dndr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E += w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * a1x_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * a1y_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * a1z_g))

                n_g = npy.dot(Y_L, nt_Lg)
                a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
                a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
                a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dntdr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E -= w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * a1x_g))
                dEdD_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * a1y_g))
                dEdD_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * a1z_g))

                y += 1
        else:
            Da_p = D_sp[0]
            Da_Lq = dot3(self.B_Lqp, Da_p)
            na_Lg = npy.dot(Da_Lq, self.n_qg)
            na_Lg[0] += self.nca_g * sqrt(4 * pi)
            nat_Lg = npy.dot(Da_Lq, self.nt_qg)
            nat_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnadr_Lg = npy.zeros((self.Lmax, self.ng))
            dnatdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(na_Lg[L], dnadr_Lg[L])
                self.rgd.derivative(nat_Lg[L], dnatdr_Lg[L])
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0

            Db_p = D_sp[1]
            Db_Lq = dot3(self.B_Lqp, Db_p)
            nb_Lg = npy.dot(Db_Lq, self.n_qg)
            nb_Lg[0] += self.ncb_g * sqrt(4 * pi)
            nbt_Lg = npy.dot(Db_Lq, self.nt_qg)
            nbt_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnbdr_Lg = npy.zeros((self.Lmax, self.ng))
            dnbtdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(nb_Lg[L], dnbdr_Lg[L])
                self.rgd.derivative(nbt_Lg[L], dnbtdr_Lg[L])
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]

                na_g = npy.dot(Y_L, na_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], na_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], na_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], na_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnadr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nb_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nb_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nb_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nb_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbdr_Lg)
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E += w * npy.dot(e_g, self.dv_g)

                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(dot3(self.B_pqL, Y_L),
                                     npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * (aa1x_g +
                                                                ab1x_g)))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * (aa1y_g +
                                                                ab1y_g)))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * (aa1z_g +
                                                                ab1z_g)))
                dEdDa_p += dEdD_p
                dEdDb_p += dEdD_p

                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * aa1x_g))
                dEdDa_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * aa1y_g))
                dEdDa_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * aa1z_g))

                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * ab1x_g))
                dEdDb_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * ab1y_g))
                dEdDb_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * ab1z_g))

                na_g = npy.dot(Y_L, nat_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], nat_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], nat_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], nat_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnatdr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nbt_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nbt_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nbt_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nbt_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbtdr_Lg)
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E -= w * npy.dot(e_g, self.dv_g)

                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                dEdD_p = w * npy.dot(dot3(self.B_pqL, Y_L),
                                     npy.dot(self.nt_qg, x_g))

                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * (aa1x_g +
                                                                ab1x_g)))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * (aa1y_g +
                                                                ab1y_g)))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * (aa1z_g +
                                                                ab1z_g)))
                dEdDa_p -= dEdD_p
                dEdDb_p -= dEdD_p

                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * aa1x_g))
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * aa1y_g))
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * aa1z_g))

                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * ab1x_g))
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * ab1y_g))
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * ab1z_g))

                y += 1

        return E - self.Exc0

    def GGA_libxc(self, D_sp, H_sp):
        r_g = self.rgd.r_g
        xcfunc = self.xc.get_functional()
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = dot3(self.B_Lqp, D_p)
            n_Lg = npy.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)
            nt_Lg = npy.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            dndr_Lg = npy.zeros((self.Lmax, self.ng))
            dntdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(n_Lg[L], dndr_Lg[L])
                self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]
                n_g = npy.dot(Y_L, n_Lg)
                a1x_g = npy.dot(A_Li[:, 0], n_Lg)
                a1y_g = npy.dot(A_Li[:, 1], n_Lg)
                a1z_g = npy.dot(A_Li[:, 2], n_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dndr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E += w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * a1x_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * a1y_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * a1z_g))

                n_g = npy.dot(Y_L, nt_Lg)
                a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
                a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
                a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dntdr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E -= w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * a1x_g))
                dEdD_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * a1y_g))
                dEdD_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * a1z_g))

                y += 1
        else:
            Da_p = D_sp[0]
            Da_Lq = dot3(self.B_Lqp, Da_p)
            na_Lg = npy.dot(Da_Lq, self.n_qg)
            na_Lg[0] += self.nca_g * sqrt(4 * pi)
            nat_Lg = npy.dot(Da_Lq, self.nt_qg)
            nat_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnadr_Lg = npy.zeros((self.Lmax, self.ng))
            dnatdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(na_Lg[L], dnadr_Lg[L])
                self.rgd.derivative(nat_Lg[L], dnatdr_Lg[L])
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0

            Db_p = D_sp[1]
            Db_Lq = dot3(self.B_Lqp, Db_p)
            nb_Lg = npy.dot(Db_Lq, self.n_qg)
            nb_Lg[0] += self.ncb_g * sqrt(4 * pi)
            nbt_Lg = npy.dot(Db_Lq, self.nt_qg)
            nbt_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnbdr_Lg = npy.zeros((self.Lmax, self.ng))
            dnbtdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(nb_Lg[L], dnbdr_Lg[L])
                self.rgd.derivative(nbt_Lg[L], dnbtdr_Lg[L])
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]

                na_g = npy.dot(Y_L, na_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], na_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], na_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], na_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnadr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nb_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nb_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nb_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nb_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbdr_Lg)
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E += w * npy.dot(e_g, self.dv_g)

                x_g = -deda2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(dot3(self.B_pqL, Y_L),
                                     npy.dot(self.n_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * aa1x_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * aa1y_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * aa1z_g))
                dEdDb_p += dEdD_p

                x_g = -deda2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(dot3(self.B_pqL, Y_L),
                                     npy.dot(self.n_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * ab1x_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * ab1y_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * ab1z_g))
                dEdDa_p += dEdD_p

                x_g = -2.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * aa1x_g))
                dEdDa_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * aa1y_g))
                dEdDa_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * aa1z_g))

                x_g = -2.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * ab1x_g))
                dEdDb_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * ab1y_g))
                dEdDb_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * ab1z_g))

                na_g = npy.dot(Y_L, nat_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], nat_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], nat_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], nat_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnatdr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nbt_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nbt_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nbt_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nbt_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbtdr_Lg)
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E -= w * npy.dot(e_g, self.dv_g)

                x_g = -deda2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(dot3(self.B_pqL, Y_L),
                                     npy.dot(self.nt_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * aa1x_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * aa1y_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * aa1z_g))
                dEdDb_p -= dEdD_p

                x_g = -deda2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(dot3(self.B_pqL, Y_L),
                                     npy.dot(self.nt_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * ab1x_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * ab1y_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * ab1z_g))
                dEdDa_p -= dEdD_p

                x_g = -2.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * aa1x_g))
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * aa1y_g))
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * aa1z_g))

                x_g = -2.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * ab1x_g))
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * ab1y_g))
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * ab1z_g))

                y += 1

        return E - self.Exc0

    def MGGA(self, D_sp, H_sp):
        r_g = self.rgd.r_g
        E = 0.0
        xcfunc = self.xc.get_functional()
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = dot3(self.B_Lqp, D_p)
            n_Lg = npy.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)
            nt_Lg = npy.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            dndr_Lg = npy.zeros((self.Lmax, self.ng))
            dntdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(n_Lg[L], dndr_Lg[L])
                self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            y = 0

            for w, Y_L in zip(self.weights, self.Y_yL):
                ## Calculate pseudo and all electron kinetic energy from orbitals
                taut_pg = self.taut_ypg[y]
                taut_g = npy.dot(D_p,taut_pg)
                tau_pg = self.tau_ypg[y]
                tau_g = npy.dot(D_p,tau_pg)
                tau_g += self.tauc_g / sqrt(4. * pi)
                A_Li = A_Liy[:self.Lmax, :, y]
                
                n_g = npy.dot(Y_L, n_Lg)
                a1x_g = npy.dot(A_Li[:, 0], n_Lg)
                a1y_g = npy.dot(A_Li[:, 1], n_Lg)
                a1z_g = npy.dot(A_Li[:, 2], n_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dndr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g,
                                            tau_g)
                E += w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * a1x_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * a1y_g))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * a1z_g))

                n_g = npy.dot(Y_L, nt_Lg)
                a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
                a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
                a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dntdr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g,
                                            taut_g)
                E -= w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p -= w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * a1x_g))
                dEdD_p -= w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * a1y_g))
                dEdD_p -= w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * a1z_g))

                y += 1
        else:
            Da_p = D_sp[0]
            Da_Lq = dot3(self.B_Lqp, Da_p)
            na_Lg = npy.dot(Da_Lq, self.n_qg)
            na_Lg[0] += self.nca_g * sqrt(4 * pi)
            nat_Lg = npy.dot(Da_Lq, self.nt_qg)
            nat_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnadr_Lg = npy.zeros((self.Lmax, self.ng))
            dnatdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(na_Lg[L], dnadr_Lg[L])
                self.rgd.derivative(nat_Lg[L], dnatdr_Lg[L])
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0
            
            Db_p = D_sp[1]
            Db_Lq = dot3(self.B_Lqp, Db_p)
            nb_Lg = npy.dot(Db_Lq, self.n_qg)
            nb_Lg[0] += self.ncb_g * sqrt(4 * pi)
            nbt_Lg = npy.dot(Db_Lq, self.nt_qg)
            nbt_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnbdr_Lg = npy.zeros((self.Lmax, self.ng))
            dnbtdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(nb_Lg[L], dnbdr_Lg[L])
                self.rgd.derivative(nbt_Lg[L], dnbtdr_Lg[L])
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                taut_pg = self.taut_ypg[y]
                tauat_g = npy.dot(Da_p,taut_pg)
                taubt_g = npy.dot(Db_p,taut_pg)
                tau_pg = self.tau_ypg[y]
                taua_g = npy.dot(Da_p,tau_pg)
                taub_g = npy.dot(Db_p,tau_pg)
                taua_g += self.tauc_g * 0.5 / sqrt(4. * pi)
                taub_g += self.tauc_g * 0.5 / sqrt(4. * pi)
                A_Li = A_Liy[:self.Lmax, :, y]                

                na_g = npy.dot(Y_L, na_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], na_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], na_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], na_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnadr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nb_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nb_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nb_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nb_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbdr_Lg)
                ab2_g += ab1_g**2
                 
                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                         (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g,
                                               taua_g,taub_g)
                E += w * npy.dot(e_g, self.dv_g)

                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp
                 
                dEdD_p = w * npy.dot(dot3(self.B_pqL, Y_L),
                                     npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * (aa1x_g +
                                                                ab1x_g)))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * (aa1y_g +
                                                                ab1y_g)))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * (aa1z_g +
                                                                ab1z_g)))
                dEdDa_p += dEdD_p
                dEdDb_p += dEdD_p
                 
                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                       npy.dot(self.n_qg, x_g))
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p += w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 0]),
                                       npy.dot(self.n_qg, x_g * aa1x_g))
                dEdDa_p += w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 1]),
                                       npy.dot(self.n_qg, x_g * aa1y_g))
                dEdDa_p += w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 2]),
                                       npy.dot(self.n_qg, x_g * aa1z_g))

                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                       npy.dot(self.n_qg, x_g))
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p += w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 0]),
                                       npy.dot(self.n_qg, x_g * ab1x_g))
                dEdDb_p += w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 1]),
                                       npy.dot(self.n_qg, x_g * ab1y_g))
                dEdDb_p += w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 2]),
                                       npy.dot(self.n_qg, x_g * ab1z_g))

                na_g = npy.dot(Y_L, nat_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], nat_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], nat_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], nat_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnatdr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nbt_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nbt_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nbt_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nbt_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbtdr_Lg)
                ab2_g += ab1_g**2
                 
                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g,
                                               tauat_g,taubt_g)
                E -= w * npy.dot(e_g, self.dv_g)

                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                dEdD_p = w * npy.dot(dot3(self.B_pqL, Y_L),
                                     npy.dot(self.nt_qg, x_g))
                 
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * (aa1x_g +
                                                                 ab1x_g)))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * (aa1y_g +
                                                                 ab1y_g)))
                dEdD_p += w * npy.dot(dot3(self.B_pqL,
                                           A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * (aa1z_g +
                                                                 ab1z_g)))
                dEdDa_p -= dEdD_p
                dEdDb_p -= dEdD_p
                
                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                       npy.dot(self.nt_qg, x_g))
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 0]),
                                       npy.dot(self.nt_qg, x_g * aa1x_g))
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 1]),
                                       npy.dot(self.nt_qg, x_g * aa1y_g))
                dEdDa_p -= w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 2]),
                                       npy.dot(self.nt_qg, x_g * aa1z_g))
                 
                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                       npy.dot(self.nt_qg, x_g))
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 0]),
                                       npy.dot(self.nt_qg, x_g * ab1x_g))
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 1]),
                                       npy.dot(self.nt_qg, x_g * ab1y_g))
                dEdDb_p -= w * npy.dot(dot3(self.B_pqL,
                                            A_Li[:, 2]),
                                       npy.dot(self.nt_qg, x_g * ab1z_g))
                
                y += 1
#        return 0.0
        return E - self.Exc0

    def GLLB(self, nucleus, gllb):
        D_sp = nucleus.D_sp
        Dresp_sp = nucleus.Dresp_sp
        H_sp = nucleus.H_sp
        extra_xc_data = nucleus.setup.extra_xc_data
        K_G = gllb.K_G
        reference_levels = [ gllb.fermi_level ]
        
        r_g = self.rgd.r_g

        # Normally, the response-part from core orbitals is calculated using the reference-level of setup-atom
        # If relaxed_core_response flag is on, the response-part is calculated using
        # core eigenvalues and self consistent reference level.
        if self.xc.xcfunc.xc.relaxed_core_response:
            core_response = npy.zeros(self.ng)
            njcore = extra_xc_data['njcore']
            for nc in range(0, njcore):
                psi2_g = extra_xc_data['core_orbital_density_'+str(nc)]
                deps = reference_levels[0]-extra_xc_data['core_eigenvalue_'+str(nc)]
                
                core_response[:] += psi2_g * K_G * (npy.where(deps<0, 0, deps))**(0.5)
        else:
            # Otherwise, the static core response from setup is used
            core_response = extra_xc_data['core_response']
        
        xcfunc = self.xc.xcfunc.xc.slater_xc
        vfunc = self.xc.xcfunc.xc.v_xc
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = dot3(self.B_Lqp, D_p)
            n_Lg = npy.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)

            Dresp_p = Dresp_sp[0]
            Dresp_Lq = dot3(self.B_Lqp, Dresp_p)
            nresp_Lg = npy.dot(Dresp_Lq, self.n_qg)
            
            nt_Lg = npy.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)

            ntresp_Lg = npy.dot(Dresp_Lq, self.nt_qg)
            
            dndr_Lg = npy.zeros((self.Lmax, self.ng))
            dntdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(n_Lg[L], dndr_Lg[L])
                self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]
                n_g = npy.dot(Y_L, n_Lg)

                if not (npy.all(n_g >= 0 )):
                    print "Warning.... negative density!"
                    return 0.0
                    #n_g = n_g - npy.min(n_g)
                
                nresp_g = npy.dot(Y_L, nresp_Lg)
                
                a1x_g = npy.dot(A_Li[:, 0], n_Lg)
                a1y_g = npy.dot(A_Li[:, 1], n_Lg)
                a1z_g = npy.dot(A_Li[:, 2], n_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dndr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)

                x_g = (2*e_g + nresp_g + core_response) / (n_g + SMALL_NUMBER)
                x_g[0] = x_g[1]

                if vfunc is not None:
                    # if vfunc.gga:
                    # Assume vfunc is a lib-xc LDA
                    assert(vfunc.gga == False)
                    v2_g = npy.zeros(self.ng)
                    e2_g = npy.zeros(self.ng)
                    vfunc.calculate_spinpaired(e2_g, n_g, v2_g)
                    e_g[:] += e2_g
                    x_g += v2_g
                

                #if gllb.relaxed_core_response:
                #    stuff todo here:

                E += w * npy.dot(e_g, self.dv_g)
                
                x_g *= self.dv_g
                
                B_Lqp = self.B_Lqp
                dEdD_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))

                n_g = npy.dot(Y_L, nt_Lg)
                ntresp_g = npy.dot(Y_L, ntresp_Lg)
                
                a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
                a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
                a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dntdr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)

                x_g = (2*e_g + ntresp_g) / (n_g + SMALL_NUMBER)
                x_g[0] = x_g[1]

                if vfunc is not None:
                    # if vfunc.gga:
                    # Assume vfunc is a lib-xc LDA
                    assert(vfunc.gga == False)
                    v2_g = npy.zeros(self.ng)
                    e2_g = npy.zeros(self.ng)
                    vfunc.calculate_spinpaired(e2_g, n_g, v2_g)
                    e_g[:] += e2_g
                    x_g += v2_g

                E -= w * npy.dot(e_g, self.dv_g)
                
                x_g *= self.dv_g
                
                B_Lqp = self.B_Lqp
                dEdD_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                y += 1
        else:
            raise NotImplementedError('GLLB spinpolarized xc-corrections')
        return E - self.Exc0

    def GLLBint(self, D_p, Dresp_p, Dlumo_p):
        r_g = self.rgd.r_g
        E = 0.0
        D_Lq = dot3(self.B_Lqp, D_p)
        n_Lg = npy.dot(D_Lq, self.n_qg)
        n_Lg[0] += self.nc_g * sqrt(4 * pi)
        nt_Lg = npy.dot(D_Lq, self.nt_qg)
        nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            
        Dresp_Lq = dot3(self.B_Lqp, Dresp_p)
        nresp_Lg = npy.dot(Dresp_Lq, self.n_qg)
        ntresp_Lg = npy.dot(Dresp_Lq, self.nt_qg)

        Dlumo_Lq = dot3(self.B_Lqp, Dlumo_p)
        nlumo_Lg = npy.dot(Dlumo_Lq, self.n_qg)
        ntlumo_Lg = npy.dot(Dlumo_Lq, self.nt_qg)
            
        y = 0
        for w, Y_L in zip(self.weights, self.Y_yL):
            A_Li = A_Liy[:self.Lmax, :, y]
            n_g = npy.dot(Y_L, n_Lg)
            nt_g = npy.dot(Y_L, nt_Lg)
            nresp_g = npy.dot(Y_L, nresp_Lg)
            ntresp_g = npy.dot(Y_L, ntresp_Lg)
            nlumo_g = npy.dot(Y_L, nlumo_Lg)
            ntlumo_g = npy.dot(Y_L, ntlumo_Lg)
            
            E += w * npy.dot(nlumo_g * nresp_g / (n_g + SMALL_NUMBER), self.dv_g)
            E -= w * npy.dot(ntlumo_g * ntresp_g / (nt_g + SMALL_NUMBER), self.dv_g)
                
            y += 1

        return E



    def two_phi_integrals(self,
                          D_sp # density matrix in packed(pack) form
                          ):
        """Evaluate the integral in the augmentation sphere.

        ::

                      /
          I_{i1 i2} = | d r [ phi_i1(r) phi_i2(r) v_xc[n](r) -
                      /       tphi_i1(r) tphi_i2(r) v_xc[tn](r) ]
                      a

        The result is given in packed(pack2) form.
        """
        I_sp = npy.zeros(D_sp.shape)
        self.calculate_energy_and_derivatives(D_sp, I_sp)
        return I_sp

    def four_phi_integrals(self, D_sp, fxc):
        """Calculate four-phi integrals.

        The density is given by the density matrix ``D_sp`` in packed(pack)
        form, and the resulting rank-four tensor is also returned in
        packed format. ``fxc`` is a radial object???
        """

        ns, np = D_sp.shape

        assert ns == 1 and not self.xc.get_functional().gga

        dot = npy.dot

        D_p = D_sp[0]
        D_Lq = dot3(self.B_Lqp, D_p)

        # Expand all-electron density in spherical harmonics:
        n_qg = self.n_qg
        n_Lg = dot(D_Lq, n_qg)
        n_Lg[0] += self.nc_g * sqrt(4 * pi)

        # Expand pseudo electron density in spherical harmonics:
        nt_qg = self.nt_qg
        nt_Lg = dot(D_Lq, nt_qg)
        nt_Lg[0] += self.nct_g * sqrt(4 * pi)

        # Allocate array for result:
        J_pp = npy.zeros((np, np))

        # Loop over 50 points on the sphere surface:
        for w, Y_L in zip(self.weights, self.Y_yL):
            B_pq = dot3(self.B_pqL, Y_L)

            fxcdv = fxc(dot(Y_L, n_Lg)) * self.dv_g
            dn2_qq = npy.inner(n_qg * fxcdv, n_qg)

            fxctdv = fxc(dot(Y_L, nt_Lg)) * self.dv_g
            dn2_qq -= npy.inner(nt_qg * fxctdv, nt_qg)

            J_pp += w * dot3(B_pq, npy.inner(dn2_qq, B_pq))

        return J_pp


    def create_kinetic(self,lmax,jlL,jl,ny,np,w_jg,tau_ypg):
        #no core kinetic energy added
        ng = self.ng
        jlmax = (jl[-1][0] + 1, lmax + 1)
        dphitdr_jlg = npy.zeros(jlmax + (ng,))
        for j1,l1 in jl:
            phit_jlg = self.rgd.r_g**l1 * w_jg[j1]                
            self.rgd.derivative(phit_jlg, dphitdr_jlg[j1][l1])

        Y_L = npy.zeros((self.Lmax))
        taut = npy.zeros((ng))
        for y in range(ny):
            A_Li = A_Liy[:self.Lmax, :, y]
            A_Lxg = A_Li[:, 0]
            A_Lyg = A_Li[:, 1]
            A_Lzg = A_Li[:, 2]
            Y_L = Y_nL[y,:self.Lmax]
            p = 0
            i1 = 0
            for j1, l1 , L1 in jlL:
                dphi1 = dphitdr_jlg[j1][l1]
                for j2, l2, L2 in jlL[i1:]:
                    dphi2 = dphitdr_jlg[j2][l2]
                    tau = A_Lxg[L1] * A_Lxg[L2] + A_Lyg[L1] * A_Lyg[L2] + A_Lzg[L1] * A_Lzg[L2] 
                    tau *= self.rgd.r_g**(l1 + l2) * w_jg[j1] * w_jg[j2] 
                    tau[1:] /= self.rgd.r_g[1:]**2                       
                    tau[0] = tau[1]
                    tau += dphi1 * dphi2 * Y_L[L1] * Y_L[L2]
                    tau *= 0.5
                    tau_ypg[y,p] = tau
                    p +=1
                i1 += 1
        return 
        
    def set_nspins(self, nspins):
        """change number of spins"""
        if nspins != self.nspins:
            self.nspins = nspins
            if nspins == 1:
                self.nc_g = self.nca_g + self.ncb_g
            else:
                self.nca_g = self.ncb_g = 0.5 * self.nc_g
