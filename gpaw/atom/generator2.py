#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from math import pi, log

import numpy as np
from scipy.special import gamma
from scipy.interpolate import interp1d

from gpaw.atom.configurations import configurations
from gpaw.atom.aeatom import AllElectronAtom, Channel, parse_ld_str
from gpaw.setup import BaseSetup
from gpaw.spline import Spline
from gpaw.basis_data import Basis
from gpaw.hgh import null_xc_correction


class PAWWaves:
    def __init__(self, ch, rcut):
        self.ch = ch
        self.rcut = rcut
        self.l = ch.l
        self.basis = ch.basis
        self.e_n = []
        self.f_n = []
        self.phi_ng = []
        self.phit_ng = None
        self.pt_ng = None
        
    def add(self, phi_g, n, e, f):
        self.phi_ng.append(phi_g)
        self.n_n.append(n)
        self.e_n.append(e)
        self.f_n.append(f)

    def pseudize(self):
        gd = self.basis.gd

        phi_ng = self.phi_ng
        N = len(phi_ng)
        phit_ng = self.phit_ng = gd.empty(N)
        gcut = gd.get_index(self.rcut)

        self.Q = 0
        self.nt_g = 0
        self.c_nx = []
        for n in range(N):
            phit_ng[n], c_x = gd.pseudize(phi_ng[n], gcut)
            self.c_nx.append(c_x)
            self.nt_g += self.f_n[n] / 4 / pi * phit_ng[n]**2
            self.Q += self.f_n[n]
            
        self.Q -= self.gd.integrate(self.nt_g)

    def hmm(self):
        pt_ng[n] = (-3.0 * c_x[2] * r_g -
                    10.0 * c_x[1] * r_g**3 -
                    21.0 * c_x[0] * r_g**5 +
                    (vtr_g - self.e_n[n] * r_g) * phit_ng[n])
        pt_ng[n, gcut:] = 0
        pt_ng = self.pt_ng = gd.empty(N)
        
        gcut = gd.get_index(self.rcut)
        u = gd.zeros()
        vr = interp1d(r_g, vr_g)
        if 0:
            u[:gcut+29] = self.ch.integrate_outwards(vr, r_g[:gcut+29],
                                                     self.e_n[n])[:,0]
            u[1:]/=r_g[1:]
            u[0]=u[1]
            phit_ng[n], c_x = gd.pseudize(u,#phi_ng[n], 
                                          gcut, self.l)
            pt_ng[n] = (-3.0 * c_x[2] * r_g -
                         10.0 * c_x[1] * r_g**3 -
                         21.0 * c_x[0] * r_g**5 +
                         (vtr_g - self.e_n[n] * r_g) * phit_ng[n])
            pt_ng[n, gcut:] = 0
            #gd.plot(phit_ng[n])
            #gd.plot(u, show=(n==0))
            gd.plot(pt_ng[n], show=(n==1))
            
        if 0:
            e = self.e_n[n]
            a_kb = [np.linalg.solve(A_bb - e * np.eye(len(A_bb)), s_b)
                    for s_b in s_kb]
            phit_kg = np.array([self.basis.expand(a_b) for a_b in a_kb])
            b_k = np.linalg.solve(phit_kg[:, gcut:gcut + 2].T,
                                  phi_ng[n][gcut:gcut + 2])
            a_b = np.dot(b_k, a_kb)
            phit_ng[n] = np.dot(b_k, phit_kg)
            d_k = np.dot(s_kb, a_b)
            x = np.dot(b_k, d_k)
            c_k = b_k / x
            pt_ng[n] = 4 * pi * np.dot(c_k, s_kg)

        self.dS_nn = np.empty((N, N))
        for n1 in range(N):
            for n2 in range(N):
                self.dS_nn[n1, n2] = gd.integrate(
                    phi_ng[n1] * phi_ng[n2] -
                    phit_ng[n1] * phit_ng[n2]) / (4 * pi)

        self.nt_g = self.phit_ng[0]**2 / (4 * pi)

        self.dH_nn = e * self.dS_nn - x
        self.Q = self.dS_nn[0, 0]
        print x, pt_ng[0,0],self.dS_nn,self.dH_nn
        dfgh
    def solve(self, vtr_g, s_g):
        gd = self.basis.gd
        r_g = gd.r_g

        phi_ng = self.phi_ng
        N = len(phi_ng)
        phit_ng = self.phit_ng = gd.empty(N)
        pt_ng = self.pt_ng = gd.empty(N)
        
        s_kg = [s_g, s_g * r_g**2]

        
        gcut = gd.get_index(self.rcut)
        
        vtr = interp1d(r_g, vtr_g)
        
        #self.e_n[1] = 1.5
        for n in range(N):
            u[:gcut+29] = self.ch.integrate_outwards(vr, r_g[:gcut + 10],
                                                     self.e_n[n])[:,0]
            u[1:]/=r_g[1:]
        a_nb = []
        s_nb = []
        for n in range(N):
            e = self.e_n[n]
            a_kb = [np.linalg.solve(A_bb - e * np.eye(len(A_bb)), s_b)
                    for s_b in s_kb]
            phit_kg = np.array([self.basis.expand(a_b) for a_b in a_kb])
            b_k = np.linalg.solve(phit_kg[:, gcut:gcut + 2].T,
                                  phi_ng[n][gcut:gcut + 2])
            a_nb.append(np.dot(b_k, a_kb))
            s_nb.append(np.dot(b_k, s_kb))
            phit_ng[n] = np.dot(b_k, phit_kg)
            phit_ng[n, gcut + 2:] = phi_ng[n][gcut + 2:]
        
        A_nn = np.inner(s_nb, a_nb)
        C_nn = np.linalg.inv(A_nn)
        
        pt_ng[:] = np.array([self.basis.expand(s_b)
                             for s_b in np.dot(C_nn, s_nb)])
        
        self.dS_nn = np.empty((N, N))
        for n1 in range(N):
            for n2 in range(N):
                self.dS_nn[n1, n2] = gd.integrate(
                    phi_ng[n1] * phi_ng[n2] -
                    phit_ng[n1] * phit_ng[n2]) / (4 * pi)
                #print gd.integrate(phit_ng[n1] * pt_ng[n2]) / (4 * pi)

        self.nt_g = self.phit_ng[0]**2 / (4 * pi)

        self.dH_nn = np.dot(self.dS_nn, np.diag(self.e_n[:N])) - A_nn.T
        self.Q = self.dS_nn[0, 0]
        print A_nn,C_nn, pt_ng[0,0],self.dS_nn,self.dH_nn


class PAWSetupGenerator:
    def __init__(self, aea, states, radii):
        self.aea = aea
        self.gd = aea.gd
        
        self.rcmax = max(radii)
        print states, radii, self.rcmax
        self.gcmax = self.gd.get_index(self.rcmax)

        self.waves_l = []
        vr = interp1d(self.gd.r_g, aea.vr_sg[0])
        for l, nn in enumerate(states):
            ch = aea.channels[l]
            waves = PAWWaves(ch, radii[l])
            for n in nn:
                if isinstance(n, int):
                    # Bound state:
                    e = ch.e_n[n - l - 1]
                    f = ch.f_n[n - l - 1]
                    phi_g = ch.phi_ng[n]
                else:
                    e = n
                    n = None
                    f = 0.0
                    phi_g = self.gd.zeros()
                    gc = self.gcmax + 10
                    u_g = ch.integrate_outwards(vr, self.gd.r_g[:gc], e)[:, 0]
                    phi_g[1:gc] = u_g[1:] / self.gd.r_g[1:gc]
                    if l == 0:
                        phi_g[0] = phi_g[1]
                print n,l,e,f
                waves.add(phi_g, n, e, f)
            self.waves_l.append(waves)

        self.alpha = log(1.0e4) / self.rcmax**2  # exp(-alpha*rcmax^2)=1.0e-4
        self.alpha = round(self.alpha, 2)
        self.ghat_g = (np.exp(-self.alpha * self.gd.r_g**2) *
                       (self.alpha / pi)**1.5)
        
        self.vtr_g = None

    def calculate_core_density(self):
        self.nc_g = self.gd.zeros()
        self.ncore = 0
        for l, ch in enumerate(self.aea.channels):
            for n, f in enumerate(ch.f_n):
                if (l < len(self.waves_l) and
                    n + l + 1 not in self.waves_l[l].n_n):
                    self.nc_g += f * ch.calculate_density(n)
                    self.ncore += f
        
        self.nct_g = self.gd.pseudize(self.nc_g, self.gcmax)[0]
        self.npseudocore = self.gd.integrate(self.nct_g)
        print self.gd.integrate(self.nc_g)
        
    def generate(self):
        self.calculate_core_density()

        self.nt_g = self.nct_g.copy()
        self.Q = -self.aea.Z + self.ncore - self.npseudocore

        self.vtr_g = self.find_zero_potential()
        
        if self.f0 > 0:
            self.nt_g += self.f0 * self.phit0_g**2 / (4 * pi)
        
        for waves in self.waves_l:
            waves.pseudize()
            self.nt_g += waves.nt_g
            self.Q += waves.Q
            
        self.rhot_g = self.nt_g + self.Q * self.ghat_g
        self.vHtr_g = self.gd.poisson(self.rhot_g)

        self.vxct_g = self.gd.zeros()
        exct_g = self.gd.zeros()
        self.exct = self.aea.xc.calculate_spherical(
            self.gd, self.nt_g.reshape((1, -1)), self.vxct_g.reshape((1, -1)))

        self.v0r_g = self.vtr_g - self.vHtr_g - self.vxct_g * self.gd.r_g

    def generate2(self):
        ntold_g = 0.0
        while True:
            self.update()
            dn = self.gd.integrate(abs(self.nt_g - ntold_g))
            sys.stdout.write('.')
            sys.stdout.flush()
            if dn < 1.0e-7:
                break
            ntold_g = self.nt_g
        print

    def find_zero_potential(self):
        self.l0 = len(self.waves_l)
        ch = self.aea.channels[self.l0]
        self.f0 = ch.f_n[0]
        e = ch.e_n[0]
        self.phit0_g, c_x = self.gd.pseudize_normalized(ch.phi_ng[0],
                                                        self.gcmax, l=self.l0,
                                                        points=6)
        print c_x
        r_g=self.gd.r_g
        n = len(c_x)
        v_g = (e +
               np.polyval(0.5 * c_x[:-1] *
                          range(2 * n - 1, 1, -2) *
                          range(2 * n - 2, 0, -2),
                          r_g**2) /
               np.polyval(c_x, r_g**2))
        v_g[self.gcmax:] = self.aea.vr_sg[0, self.gcmax:] / r_g[self.gcmax:]
        return v_g * r_g

    def plot(self):
        import matplotlib.pyplot as plt
        r_g = self.gd.r_g
        plt.plot(r_g, self.vxct_g, label='xc')
        plt.plot(r_g[1:], self.v0r_g[1:] / r_g[1:], label='0')
        plt.plot(r_g[1:], self.vHtr_g[1:] / r_g[1:], label='H')
        plt.plot(r_g[1:], self.vtr_g[1:] / r_g[1:], label='ps')
        plt.plot(r_g[1:], self.aea.vr_sg[0, 1:] / r_g[1:], label='ae')
        plt.axis(xmax=2 * self.rcmax,
                 ymin=self.vtr_g[1] / r_g[1],
                 ymax=max(0, (self.v0r_g[1:] / r_g[1:]).max()))
        plt.legend()
        
        plt.figure()

        for waves in self.waves_l:
            for phi_g, phit_g in zip(waves.phi_ng, waves.phit_ng):
                plt.plot(r_g, phi_g * r_g)
                plt.plot(r_g, phit_g * r_g, '--')

        plt.plot(r_g, self.aea.channels[self.l0].phi_ng[0] * r_g)
        plt.plot(r_g, self.phit0_g * r_g, '--')

        plt.axis(xmax=3 * self.rcmax)

        plt.figure()
        for waves in self.waves_l:
            for pt_g in waves.pt_ng:
                plt.plot(r_g, pt_g * r_g)
        plt.axis(xmax=3 * self.rcmax)

        plt.show()

    def logarithmic_derivative(self, l, energies, rcut):
        vtr = interp1d(self.gd.r_g, self.vtr_g)
        ch = Channel(l)
        gcut = self.gd.get_index(rcut)
        r_g = self.gd.r_g[:gcut + 1]

        if l < len(self.waves_l):
            # Nonlocal PAW stuff:
            waves = self.waves_l[l]
            pt_n = [interp1d(self.gd.r_g, -pt_g) for pt_g in waves.pt_ng]
            dH_nn = waves.dH_nn
            dS_nn = waves.dS_nn
            N = len(pt_n)
        else:
            N = 0

        u_xg = self.gd.zeros(2)
        u_nxg = self.gd.zeros((N, 2))
        logderivs = []
        for e in energies:
            u_xg[:, :gcut + 1] = ch.integrate_outwards(vtr, r_g, e).T
            if N:
                u_nxg[:, :, :gcut + 1] = [ch.integrate_outwards(vtr, r_g,
                                                                e, pt).T
                                          for pt in pt_n]
                A_nn = dH_nn - e * dS_nn
                pt_ng = -self.waves_l[l].pt_ng
                B_nn = self.gd.integrate(pt_ng * u_nxg[:, :1], -1) / (4 * pi)
                c_n  = self.gd.integrate(pt_ng * u_xg[0], -1) / (4 * pi)
                d_n = np.linalg.solve(np.dot(A_nn, B_nn) + np.eye(N),
                                      np.dot(A_nn, c_n))
                u_xg -= np.dot(u_nxg.T, d_n).T
            logderivs.append(u_xg[1, gcut] / u_xg[0, gcut])
        return logderivs
        
    def make_paw_setup(self):
        phit_g = self.zeropot.basis.expand(self.zeropot.C_nb[0])
        return PAWSetup(self.alpha, self.gd.r_g, phit_g, self.v0_g)


     
def build_parser(): 
    from optparse import OptionParser

    parser = OptionParser(usage='%prog [options] element',
                          version='%prog 0.1')
    parser.add_option('-f', '--xc-functional', type='string', default='LDA',
                      help='Exchange-Correlation functional ' +
                      '(default value LDA)',
                      metavar='<XC>')
    parser.add_option('-P', '--projectors',
                      help='Projector functions - use comma-separated - ' +
                      'nl values, where n can be pricipal quantum number ' +
                      '(integer) or energy (floating point number). ' +
                      'Example: 2s,0.5s,2p,0.5p,0.0d.')
    parser.add_option('-r', '--radius',
                      help='1.2 or 1.2,1.1,1.1')
    parser.add_option('-p', '--plot', action='store_true')
    parser.add_option('-l', '--logarithmic-derivatives',
                      metavar='spdfg,e1:e2:de,radius',
                      help='Plot logarithmic derivatives. ' +
                      'Example: -l spdf,-1:1:0.05,1.3. ' +
                      'Energy range and/or radius can be left out.')
    return parser


def main(AEA=AllElectronAtom):
    parser = build_parser()
    opt, args = parser.parse_args()

    if len(args) != 1:
        parser.error('Incorrect number of arguments')
    symbol = args[0]

    kwargs = {'xc': opt.xc_functional}
        
    aea = AEA(symbol, **kwargs)

    lmax = -1
    states = {}
    if opt.projectors:
        for s in opt.projectors.split(','):
            l = 'spdf'.find(s[-1])
            n = eval(s[:-1])
            if l in states:
                states[l].append(n)
            else:
                states[l] = [n]
            if l > lmax:
                lmax = l

    # Add empty bound states:
    for l, nn in states.items():
        for n in nn:
            if (isinstance(n, int) and
                (l not in aea.f_lsn or n - l > len(aea.f_lsn[l][0]))):
                aea.add(n, l, 0)

    # Add state for local potential (zero-potential, v-bar):
    aea.add(lmax + 2, lmax + 1, 0)

    aea.initialize()
    aea.run()
    #print 'no refine!!!'#
    aea.refine()
                
    if opt.radius:
        radii = [float(r) for r in opt.radius.split(',')]
    else:
        rave = aea.gd.integrate(aea.n_sg[0], 1) / aea.Z
        radii = [0.5 * rave]
    if lmax >= 0:
        radii += [radii[-1]] * (lmax + 1 - len(radii))

    projectors = [states[l] for l in range(lmax + 1)]
    gen = PAWSetupGenerator(aea, projectors, radii)
    gen.generate()
    
    if opt.logarithmic_derivatives:
        lvalues, energies, r = parse_ld_str(opt.logarithmic_derivatives)
        import matplotlib.pyplot as plt
        for l in lvalues:
            ld = gen.logarithmic_derivative(l, energies, r)
            plt.plot(energies, ld, '--')
            ld = aea.logarithmic_derivative(l, energies, r)
            plt.plot(energies, ld)
        plt.show()

    if opt.plot:
        gen.plot()

if __name__ == '__main__':
    main()

class PAWSetup:
    def __init__(self, alpha, r_g, phit_g, v0_g):
        self.natoms = 0
        self.E = 0.0
        self.Z = 1
        self.Nc = 0
        self.Nv = 1
        self.niAO = 1
        self.pt_j = []
        self.ni = 0
        self.l_j = []
        self.nct = None
        self.Nct = 0.0

        rc = 1.0
        r2_g = np.linspace(0, rc, 100)**2
        x_g = np.exp(-alpha * r2_g)
        x_g[-1] = 0 

        self.ghat_l = [Spline(0, rc,
                              (4 * pi)**0.5 * (alpha / pi)**1.5 * x_g)]

        self.vbar = Spline(0, rc, (4 * pi)**0.5 * v0_g[0] * x_g)

        r = np.linspace(0, 4.0, 100)
        phit = splev(r, splrep(r_g, phit_g))
        poly = np.polyfit(r[[-30,-29,-2,-1]], [0, 0, phit[-2], phit[-1]], 3)
        phit[-30:] -= np.polyval(poly, r[-30:])
        self.phit_j = [Spline(0, 4.0, phit)]
                              
        self.Delta_pL = np.zeros((0, 1))
        self.Delta0 = -1 / (4 * pi)**0.5
        self.lmax = 0
        self.K_p = self.M_p = self.MB_p = np.zeros(0)
        self.M_pp = np.zeros((0, 0))
        self.Kc = 0.0
        self.MB = 0.0
        self.M = 0.0
        self.xc_correction = null_xc_correction
        self.HubU = None
        self.dO_ii = np.zeros((0, 0))
        self.type = 'local'
        self.fingerprint = None
        
    def get_basis_description(self):
        return '1s basis cut off at 4 Bohr'

    def print_info(self, text):
        text('Local pseudo potential')
        
    def calculate_initial_occupation_numbers(self, magmom, hund, charge,
                                             nspins):
        return np.array([(1.0,)])

    def initialize_density_matrix(self, f_si):
        return np.zeros((1, 0))

    def calculate_rotations(self, R_slmm):
        self.R_sii = np.zeros((1, 0, 0))

if 0:
    aea = AllElectronAtom('H')
    aea.add(2, 1, 0)
    aea.initialize()
    aea.run()
    g = PAWSetupGenerator(aea, [(0.8, [1])], 1)
    #g = PAWSetupGenerator(aea, [], 0, 0.8)
    g.generate()
    g.plot()
    setup = g.make_paw_setup()
    from ase.data.molecules import molecule
    from gpaw import GPAW

    a = molecule('H', pbc=1, magmoms=[0])
    a.center(vacuum=2)
    a.set_calculator(
        GPAW(setups={0: setup}))
    a.get_potential_energy()
