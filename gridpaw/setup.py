# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
from math import log, pi, sqrt
import sys

import Numeric as num
from ASE.ChemicalElements.name import names

from gridpaw.read_setup import PAWXMLParser
from gridpaw.gaunt import gaunt as G_L1L2L
from gridpaw.spline import Spline
from gridpaw.grid_descriptor import RadialGridDescriptor
from gridpaw.utilities import unpack, erf, fac, hartree
from gridpaw.xc_atom import XCAtom
from gridpaw.xc_functional import XCOperator

class Setup:
    def __init__(self, symbol, xcfunc, lmax=0, nspins=1, softgauss=True):
        xcname = xcfunc.get_xc_name()
        self.symbol = symbol
        self.xcname = xcname
        self.softgauss = softgauss
        
        (Z, Nc, Nv,
         e_total, e_kinetic, e_electrostatic, e_xc,
         e_kinetic_core,
         n_j, l_j, f_j, eps_j, rcut_j, id_j,
         ng, beta,
         nc_g, nct_g, vbar_g, gamma,
         phi_jg, phit_jg, pt_jg,
         e_kin_jj, X_p, ExxC,
         self.fingerprint,
         filename) = PAWXMLParser().parse(symbol, xcname)

        self.filename = filename
        self.Nv = Nv
        self.Z = Z
        self.X_p = X_p
        self.ExxC = ExxC

        nj = len(l_j)
        e_kin_jj.shape = (nj, nj)

        self.n_j = n_j
        self.l_j = l_j
        self.f_j = f_j
        self.eps_j = eps_j

        rcut = max(rcut_j)
        rcut2 = 2 * rcut
        gcut2 = int(rcut2 * ng / (rcut2 + beta))

        g = num.arange(ng, typecode=num.Float)
        r_g = beta * g / (ng - g)
        dr_g = beta * ng / (ng - g)**2
        d2gdr2 = -2 * ng * beta / (beta + r_g)**3

        if 0:
            for j in range(nj):
                norm = num.dot((phi_jg[j] * r_g)**2, dr_g)
                print id_j[j], norm, num.dot(phit_jg[j] * pt_jg[j] * r_g**2,
                                             dr_g)
            for j1 in range(nj):
                for j2 in range(nj):
                    if l_j[j1] == l_j[j2]:
                        print j1, j2, num.dot(phit_jg[j1] * pt_jg[j2] * r_g**2,
                                              dr_g)

        # Find cutoff for core density:
        if Nc == 0:
            rcore = 0.5
        else:
            N = 0.0
            g = ng - 1
            while N < 1e-7:
                N += sqrt(4 * pi) * nc_g[g] * r_g[g]**2 * dr_g[g]
                g -= 1
            rcore = r_g[g]

        ni = 0
        niAO = 0
        i = 0
        j = 0
        jlL_i = []
        self.nk = 0
        lmax1 = 0
        for l, f in zip(l_j, f_j):
            self.nk += 3 + l * (1 + 2 * l)
            if l > lmax1:
                lmax1 = l
            if f > 0:
                niAO += 2 * l + 1
            for m in range(2 * l + 1):
                jlL_i.append((j, l, l**2 + m))
                i += 1
            j += 1
        ni = i
        self.ni = ni
        self.niAO = niAO
        
        if 2 * lmax1 < lmax:
            lmax = 2 * lmax1

        if lmax1 > 1:
            lmax1 = 1
            
        np = ni * (ni + 1) / 2
        nq = nj * (nj + 1) / 2
        Lmax = (lmax + 1)**2
        
        # Construct splines:
        self.nct = Spline(0, rcore, nct_g, r_g=r_g, beta=beta)
        self.vbar = Spline(0, rcut2, vbar_g, r_g=r_g, beta=beta)

        def grr(phi_g, l, r_g):
            w_g = phi_g.copy()
            if l > 0:
                w_g[1:] /= r_g[1:]**l
                w1, w2 = w_g[1:3]
                r0, r1, r2 = r_g[0:3]
                w_g[0] = w2 + (w1 - w2) * (r0 - r2) / (r1 - r2) 
            return w_g
        
        self.pt_j = []
        for j in range(nj):
            l = l_j[j]
            self.pt_j.append(Spline(l, rcut2, grr(pt_jg[j], l, r_g),
                                    r_g=r_g, beta=beta))
     
        cutoff = 8.0 # ????????
        self.wtLCAO_j = []
        for j, phit_g in enumerate(phit_jg):
            if f_j[j] > 0:
                l = l_j[j]
                self.wtLCAO_j.append(Spline(l, cutoff,
                                            grr(phit_g, l, r_g),
                                            r_g=r_g, beta=beta))

        r_g = r_g[:gcut2].copy()
        dr_g = dr_g[:gcut2].copy()
        phi_jg = num.array([phi_g[:gcut2] for phi_g in phi_jg])
        phit_jg = num.array([phit_g[:gcut2] for phit_g in phit_jg])
        nc_g = nc_g[:gcut2].copy()
        nct_g = nct_g[:gcut2].copy()
        vbar_g = vbar_g[:gcut2].copy()

        T_Lqp = num.zeros((Lmax, nq, np), num.Float)
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL_i:
            for j2, l2, L2 in jlL_i[i1:]:
                if j1 < j2:
                    q = j2 + j1 * nj - j1 * (j1 + 1) / 2
                else:
                    q = j1 + j2 * nj - j2 * (j2 + 1) / 2
                T_Lqp[:, q, p] = G_L1L2L[L1, L2, :Lmax]
                p += 1
            i1 += 1

        g_lg = num.zeros((lmax + 1, gcut2), num.Float)
        rcgauss = rcut / sqrt(gamma)
        g_lg[0] = 4 / rcgauss**3 / sqrt(pi) * num.exp(-(r_g / rcgauss)**2)
        for l in range(1, lmax + 1):
            g_lg[l] = 2.0 / (2 * l + 1) / rcgauss**2 * r_g * g_lg[l - 1]
                
        for l in range(lmax + 1):
            g_lg[l] /= num.dot(r_g**(l + 2) * dr_g, g_lg[l])

        n_qg = num.zeros((nq, gcut2), num.Float)
        nt_qg = num.zeros((nq, gcut2), num.Float)
        q = 0
        for j1 in range(nj):
            for j2 in range(j1, nj):
                n_qg[q] = phi_jg[j1] * phi_jg[j2]
                nt_qg[q] = phit_jg[j1] * phit_jg[j2]
                q += 1

        Delta_lq = num.zeros((lmax + 1, nq), num.Float)
        for l in range(lmax + 1):
            Delta_lq[l] = num.dot(n_qg - nt_qg, r_g**(2 + l) * dr_g)
            
        self.Delta_pL = num.zeros((np, Lmax), num.Float)
        for l in range(lmax + 1):
            L = l**2
            for m in range(2 * l + 1):
                delta_p = num.dot(Delta_lq[l], T_Lqp[L + m])
                self.Delta_pL[:, L + m] = delta_p

        Delta = num.dot(nc_g - nct_g, r_g**2 * dr_g) - Z / sqrt(4 * pi)
        self.Delta0 = Delta

        def H(n_g, l):
            yrrdr_g = num.zeros(gcut2, num.Float)
            nrdr_g = n_g * r_g * dr_g
            hartree(l, nrdr_g, beta, ng, yrrdr_g)
            yrrdr_g *= r_g * dr_g
            return yrrdr_g
        
        wnc_g = H(nc_g, l=0)
        wnct_g = H(nct_g, l=0)
        
        wg_lg = [H(g_lg[l], l) for l in range(lmax + 1)]

        wn_lqg = [num.array([H(n_qg[q], l) for q in range(nq)])
                  for l in range(lmax + 1)]
        wnt_lqg = [num.array([H(nt_qg[q], l) for q in range(nq)])
                   for l in range(lmax + 1)]

        rdr_g = r_g * dr_g
        dv_g = r_g * rdr_g
        A = 0.5 * num.dot(nc_g, wnc_g)
        A -= sqrt(4 * pi) * Z * num.dot(rdr_g, nc_g)
        mct_g = nct_g + Delta * g_lg[0]
        wmct_g = wnct_g + Delta * wg_lg[0]
        A -= 0.5 * num.dot(mct_g, wmct_g)
        self.M = A
        AB = -num.dot(dv_g * nct_g, vbar_g)
        self.MB = AB
        
        A_q = 0.5 * (num.dot(wn_lqg[0], nc_g)
                     + num.dot(n_qg, wnc_g))
        A_q -= sqrt(4 * pi) * Z * num.dot(n_qg, rdr_g)

        A_q -= 0.5 * (num.dot(wnt_lqg[0], mct_g)
                     + num.dot(nt_qg, wmct_g))
        A_q -= 0.5 * (num.dot(mct_g, wg_lg[0])
                      + num.dot(g_lg[0], wmct_g)) * Delta_lq[0]
        self.M_p = num.dot(A_q, T_Lqp[0])
        
        AB_q = -num.dot(nt_qg, dv_g * vbar_g)
        self.MB_p = num.dot(AB_q, T_Lqp[0])

        A_lqq = []
        for l in range(lmax + 1):
            A_qq = 0.5 * num.dot(n_qg, num.transpose(wn_lqg[l]))
            A_qq -= 0.5 * num.dot(nt_qg, num.transpose(wnt_lqg[l]))
            A_qq -= 0.5 * num.outerproduct(Delta_lq[l],
                                            num.dot(wnt_lqg[l], g_lg[l]))
            A_qq -= 0.5 * num.outerproduct(num.dot(nt_qg, wg_lg[l]),
                                            Delta_lq[l])
            A_qq -= 0.5 * num.dot(g_lg[l], wg_lg[l]) * \
                      num.outerproduct(Delta_lq[l], Delta_lq[l])
            A_lqq.append(A_qq)
        
        self.M_pp = num.zeros((np, np), num.Float)
        L = 0
        for l in range(lmax + 1):
            for m in range(2 * l + 1):
                self.M_pp += num.dot(num.transpose(T_Lqp[L]),
                                    num.dot(A_lqq[l], T_Lqp[L]))
                L += 1
        
        # Make a radial grid descriptor:
        rgd = RadialGridDescriptor(r_g, dr_g)

        xc = XCOperator(xcfunc, rgd, nspins)

        self.xc = XCAtom(xc,
                         [grr(phi_g, l_j[j], r_g)
                          for j, phi_g in enumerate(phi_jg)],
                         [grr(phit_g, l_j[j], r_g)
                          for j, phit_g in enumerate(phit_jg)],
                         nc_g / sqrt(4 * pi), nct_g / sqrt(4 * pi),
                         rgd, [(j, l_j[j]) for j in range(nj)],
                         2 * lmax1, e_xc)

        self.rcut = rcut

        # Dont forget to change the onsite interaction energy for soft = 0 XXX
        if softgauss:
            rcutsoft = rcut2####### + 1.4
        else:
            rcutsoft = rcut2

##        rcutsoft += 1.0
        
        self.rcutsoft = rcutsoft
        
        if xcname != self.xcname:
            raise RuntimeError('Not the correct XC-functional!')
        
        # Use atomic all-electron energies as reference:
        self.Kc = e_kinetic_core - e_kinetic
        self.M -= e_electrostatic
        self.E = e_total

        self.O_ii = sqrt(4.0 * pi) * unpack(self.Delta_pL[:, 0].copy())

        K_q = []
        for j1 in range(nj):
            for j2 in range(j1, nj):
                K_q.append(e_kin_jj[j1, j2])
        self.K_p = sqrt(4 * pi) * num.dot(K_q, T_Lqp[0])
        
        self.lmax = lmax

        r = 0.02 * rcutsoft * num.arange(51, typecode=num.Float)
##        r = 0.04 * rcutsoft * num.arange(26, typecode=num.Float)
        alpha = rcgauss**-2
        self.alpha = alpha
        if softgauss:
            assert lmax <= 2
            alpha2 = 22.0 / rcutsoft**2
            alpha2 = 15.0 / rcutsoft**2

            vt0 = 4 * pi * (num.array([erf(x) for x in sqrt(alpha) * r]) -
                            num.array([erf(x) for x in sqrt(alpha2) * r]))
            vt0[0] = 8 * sqrt(pi) * (sqrt(alpha) - sqrt(alpha2))
            vt0[1:] /= r[1:]
            vt_l = [vt0]
            if lmax >= 1:
                arg = num.clip(alpha2 * r**2, 0.0, 700.0)
                e2 = num.exp(-arg)
                arg = num.clip(alpha * r**2, 0.0, 700.0)
                e = num.exp(-arg)
                vt1 = vt0 / 3 - 8 * sqrt(pi) / 3 * (sqrt(alpha) * e -
                                                    sqrt(alpha2) * e2)
                vt1[0] = 16 * sqrt(pi) / 9 * (alpha**1.5 - alpha2**1.5)
                vt1[1:] /= r[1:]**2
                vt_l.append(vt1)
                if lmax >= 2:
                    vt2 = vt0 / 5 - 8 * sqrt(pi) / 5 * \
                          (sqrt(alpha) * (1 + 2 * alpha * r**2 / 3) * e -
                           sqrt(alpha2) * (1 + 2 * alpha2 * r**2 / 3) * e2)
                    vt2[0] = 32 * sqrt(pi) / 75 * (alpha**2.5 -
                                                   alpha2**2.5)
                    vt2[1:] /= r[1:]**4
                    vt_l.append(vt2)

            self.Deltav_l = []
            for l in range(lmax + 1):
                vtl = vt_l[l]
                vtl[-1] = 0.0
                self.Deltav_l.append(Spline(l, rcutsoft, vtl))

        else:
            alpha2 = alpha
            self.Deltav_l = [Spline(l, rcutsoft, 0 * r)
                             for l in range(lmax + 1)]

        self.alpha2 = alpha2

        d_l = [fac[l] * 2**(2 * l + 2) / sqrt(pi) / fac[2 * l + 1]
               for l in range(3)]
        g = alpha2**1.5 * num.exp(-alpha2 * r**2)
        g[-1] = 0.0
        self.gt_l = [Spline(l, rcutsoft, d_l[l] * alpha2**l * g)
                     for l in range(lmax + 1)]

        # Construct atomic density matrix for the ground state (to be
        # used for testing):
        self.D_sp = num.zeros((1, np), num.Float)
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL_i:
            occ = f_j[j1] / (2.0 * l1 + 1)
            for j2, l2, L2 in jlL_i[i1:]:
                if j1 == j2 and L1 == L2:
                    self.D_sp[0, p] = occ
                p += 1
            i1 += 1

    def print_info(self, out):
        print >> out, self.symbol + '-setup:'
        print >> out, '  name   :', names[self.Z]
        print >> out, '  Z      :', self.Z
        print >> out, '  file   :', self.filename
        print >> out, '  cutoffs: %4.2f Bohr, lmax=%d' % (self.rcut, self.lmax)
        print >> out, '  valence states:'
        for n, l, f, eps in zip(self.n_j, self.l_j, self.f_j, self.eps_j):
            if f > 0:
                print >> out, '    %d%s(%d) %7.3f Ha' % (n, 'spd'[l], f, eps)
            else:
                print >> out, '     %s    %7.3f Ha' % ('spd'[l], eps)
        print >> out

    def calculate_rotations(self, R_slmm):
        nsym = len(R_slmm)
        self.R_sii = num.zeros((nsym, self.ni, self.ni), num.Float)
        i1 = 0
        for l in self.l_j:
            i2 = i1 + 2 * l + 1
            for s, R_lmm in enumerate(R_slmm):
                self.R_sii[s, i1:i2, i1:i2] = R_lmm[l]
            i1 = i2

    def symmetrize(self, a, D_aii, map_sa):
        D_ii = num.zeros((self.ni, self.ni), num.Float)
        for s, R_ii in enumerate(self.R_sii):
            D_ii += num.dot(R_ii, num.dot(D_aii[map_sa[s][a]],
                                              num.transpose(R_ii)))
        return D_ii / len(map_sa)

    # Get rid of all these methods: XXXXXXXXXXXXXXXXXXXXXX
    def get_smooth_core_density(self):
        return self.nct

    def get_number_of_atomic_orbitals(self):
        return self.niAO

    def get_number_of_partial_waves(self):
        return self.ni
    
    def get_recommended_grid_spacing(self):
        return 0.4 # self.h ???  XXXXXXXXXXXX

    def get_projectors(self):
        return self.pt_j

    def get_atomic_orbitals(self):
        return self.wtLCAO_j

    def delete_atomic_orbitals(self):
        del self.wtLCAO_j

    def get_shape_functions(self):
        return self.gt_l
    
    def get_potential(self):
        return self.Deltav_l

    def get_localized_potential(self):
        return self.vbar

    def get_number_of_valence_electrons(self):
        return self.Nv
