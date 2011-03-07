#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from math import pi, log

import numpy as np
from scipy.special import gamma
from scipy.interpolate import interp1d

from gpaw.atom.configurations import configurations
from gpaw.atom.aeatom import AllElectronAtom, Channel
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
        
    def add(self, phi_g, e, f):
        self.phi_ng.append(phi_g)
        self.e_n.append(e)
        self.f_n.append(f)

    def solve(self, vtr_g, s_g):
        gd = self.basis.gd
        r_g = gd.r_g

        phi_ng = self.phi_ng
        N = len(phi_ng)
        phit_ng = self.phit_ng = gd.empty(N)
        pt_ng = self.pt_ng = gd.empty(N)
        
        s_kg = [s_g, s_g * r_g**2]
        s_kb = [gd.integrate(self.basis.basis_bg * s_g) for s_g in s_kg]

        
        gcut = gd.get_index(self.rcut)
        A_bb = self.basis.T_bb + self.basis.calculate_potential_matrix(vtr_g)

        for n in range(N):
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


class PAWSetupGenerator:
    def __init__(self, aea, states, radii):
        self.aea = aea
        self.gd = aea.gd
        
        self.rcmax = max(radii)
        self.gcmax = self.gd.get_index(self.rcmax)

        self.waves_l = []
        for l, nn in enumerate(states):
            ch = aea.channels[l]
            waves = PAWWaves(ch, radii[l])
            for n in nn:
                e = ch.e_n[n - l - 1]
                f = ch.f_n[n - l - 1]
                print n,l,e,f
                phi_g = ch.basis.expand(ch.C_nb[0])
                waves.add(phi_g, e, f)
            self.waves_l.append(waves)

        self.alpha = log(1.0e4) / self.rcmax**2  # exp(-alpha*rcmax^2)=1.0e-4
        self.alpha = round(self.alpha, 2)
        self.ghat_g = (np.exp(-self.alpha * self.gd.r_g**2) *
                       (self.alpha / pi)**1.5)

        l0 = len(states)
        ch0 = aea.get_channel(l0)
        self.zeropot = Channel(l0, 0, ch0.f_n, ch0.basis)
        self.zeropot.e_n = [ch0.e_n[0]]

        self.vtr_g = None

    def pseudize(self, a_g, gc, n=1):
        assert isinstance(gc, int) and gc > 10 and n == 1
        r_g = self.gd.r_g
        poly = np.polyfit([0, r_g[gc - 1], r_g[gc], r_g[gc + 1]],
                          [0, a_g[gc - 1], a_g[gc], a_g[gc + 1]], 3)
        at_g = a_g.copy()
        at_g[:gc] = np.polyval(poly, r_g[:gc])
        return at_g

    def generate(self):
        gd = self.gd
        
        self.vtr_g = self.pseudize(self.aea.vr_sg[0], self.gcmax)
        self.v0_g = 0.0

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

    def update(self):
        self.nt_g = 0.0
        self.Q = -1.0

        self.find_zero_potential()

        for waves in self.waves_l:
            waves.solve(self.vtr_g, self.ghat_g)
            self.nt_g += waves.nt_g
            self.Q += waves.Q
            
        self.rhot_g = self.nt_g + self.Q * self.ghat_g
        self.vHtr_g = self.gd.poisson(self.rhot_g)
        self.vxct_g = self.gd.zeros()
        exct_g = self.gd.zeros()
        self.exct = self.aea.xc.calculate_spherical(
            self.gd, self.nt_g.reshape((1, -1)), self.vxct_g.reshape((1, -1)))
        self.vtr_g = self.vHtr_g + (self.vxct_g + self.v0_g) * self.gd.r_g
        
    def find_zero_potential(self):
        dv0_g = self.ghat_g
        r_g = self.gd.r_g
        pot0 = self.zeropot
        e0 = pot0.e_n[0]
        V_bb = pot0.basis.calculate_potential_matrix(dv0_g * r_g)
        while True:
            pot0.solve(self.vtr_g)
            e = pot0.e_n[0]
            if abs(e - e0) < 1.0e-8:
                break

            c_b = pot0.C_nb[0]
            v = np.dot(np.dot(c_b, V_bb), c_b)
            a = (e0 - e) / v
            self.vtr_g += a * dv0_g * r_g
            self.v0_g += a * dv0_g

        self.nt_g += pot0.calculate_density()

    def plot(self):
        import matplotlib.pyplot as plt
        r_g = self.gd.r_g
        plt.plot(r_g, self.vxct_g, label='xc')
        plt.plot(r_g, self.v0_g, label='0')
        plt.plot(r_g[1:], self.vHtr_g[1:] / r_g[1:], label='H')
        plt.plot(r_g[1:], self.vtr_g[1:] / r_g[1:], label='ps')
        plt.plot(r_g[1:], self.aea.vr_sg[0, 1:] / r_g[1:], label='ae')
        plt.axis(xmax=2 * self.rcmax,
                 ymin=self.vtr_g[1] / r_g[1],
                 ymax=max(0, self.v0_g[0]))
        plt.legend()
        
        plt.figure()

        if len(self.waves_l) == 0:
            phit_g = self.zeropot.basis.expand(self.zeropot.C_nb[0])
            ch = self.aea.channels[0]
            phi_g = ch.basis.expand(ch.C_nb[0])
            plt.plot(r_g, phi_g * r_g)
            plt.plot(r_g, phit_g * r_g)

        for waves in self.waves_l:
            plt.plot(r_g, waves.phi_ng[0] * r_g)
            plt.plot(r_g, waves.phit_ng[0] * r_g)
        plt.axis(xmax=2 * self.rcmax)

        plt.figure()
        for waves in self.waves_l:
            plt.plot(r_g, waves.pt_ng[0] * r_g)
        plt.axis(xmax=2 * self.rcmax)

        plt.show()

    def logarithmic_derivative(self, l, energies, rcut):
        vtr = interp1d(self.gd.r_g, self.vtr_g)
        ch = self.aea.get_channel(l)
        gcut = self.gd.get_index(rcut)
        r_g = self.gd.r_g[:gcut + 1]

        # Nonlocal PAW stuff:
        nl_l = {}
        for l, waves in enumerate(self.waves_l):
            pt_n = [interp1d(self.gd.r_g, -pt_g) for pt_g in waves.pt_ng]
            nl_l[l] = (waves.dH_nn, waves.dS_nn, pt_n)

        u_xg = self.gd.zeros(2)
        logderivs = []
        for e in energies:
            u_xg[:, :gcut + 1] = ch.integrate_outwards(vtr, r_g, e).T
            if l in nl_l:
                dH_nn, dS_nn, pt_n = nl_l[l]
                N = len(pt_n)
                u_nxg = self.gd.zeros((N, 2))
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
                      help='2s3s2p3p3d')
    parser.add_option('-r', '--radius',
                      help='1.2 or 1.2,1.1,1.1')
    parser.add_option('-p', '--plot', action='store_true')
    parser.add_option('-l', '--logarithmic-derivatives',
                      help='-l 1.3,spdf,-2,2,100')
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
        s = opt.projectors
        while s:
            n = int(s[0])
            l = 'spdf'.find(s[1])
            if l in states:
                states[l].append(n)
            else:
                states[l] = [n]
            s = s[2:]
            if l > lmax:
                lmax = l
                
    for l, nn in states.items():
        for n in nn:
            if l not in aea.f_lsn or n - l > len(aea.f_lsn[l][0]):
                aea.add(n, l, 0)

    aea.add(lmax + 2, lmax + 1, 0)

    aea.initialize()
    aea.run()

    if opt.radius:
        radii = [float(r) for r in opt.radius.split(',')]
    else:
        rave = aea.gd.integrate(aea.n_sg[0], 1) / aea.Z
        radii = [2 * rave]
    if lmax >= 0:
        radii += [radii[-1]] * (lmax + 1 - len(radii))

    projectors = [states[l] for l in range(lmax + 1)]
    gen = PAWSetupGenerator(aea, projectors, radii)
    gen.generate()
    
    if opt.logarithmic_derivatives:
        rcut, lvalues, emin, emax, npoints = \
              opt.logarithmic_derivatives.split(',')
        rcut = float(rcut)
        lvalues = ['spdfg'.find(x) for x in lvalues]
        emin = float(emin)
        emax = float(emax)
        npoints = int(npoints)
        energies = np.linspace(emin, emax, npoints)
        import matplotlib.pyplot as plt
        for l in lvalues:
            ld = gen.logarithmic_derivative(l, energies, rcut)
            plt.plot(energies, ld)
            ld = aea.logarithmic_derivative(l, energies, rcut)
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
