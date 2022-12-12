from cmath import exp
from math import pi, sqrt

import numpy as npy
from gpaw.gauss import I
from gpaw.spherical_harmonics import YL
from gpaw.utilities import fac, warning

"""
assert lmax <= 2
            alpha2 = 22.0 / rcutsoft**2
            alpha2 = 15.0 / rcutsoft**2 # What the heck

            vt0 = 4 * pi * (erf(sqrt(alpha) * r) - erf(sqrt(alpha2) * r))
            vt0[0] = 8 * sqrt(pi) * (sqrt(alpha) - sqrt(alpha2))
            vt0[1:] /= r[1:]
            vt_l = [vt0]
            if lmax >= 1:
                arg = npy.clip(alpha2 * r**2, 0.0, 700.0)
                e2 = npy.exp(-arg)
                arg = npy.clip(alpha * r**2, 0.0, 700.0)
                e = npy.exp(-arg)
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

            self.vhat_l = []
            for l in range(lmax + 1):
                vtl = vt_l[l]
                vtl[-1] = 0.0
                self.vhat_l.append(Spline(l, rcutsoft, vtl))

        self.vhat_L = None
            vhat_L = create(vhat_l, finegd, spos_c, lfbc=lfbc)

    def add_hat_potential(self, vt2):
        if self.vhat_L is not None:
            self.vhat_L.add(vt2, self.Q_L)

            W_L = npy.zeros((s.lmax + 1)**2)
            for neighbor in self.neighbors:
                W_L += npy.dot(neighbor.v_LL, neighbor.nucleus().Q_L)
            U = 0.5 * npy.dot(self.Q_L, W_L)

            if self.vhat_L is not None:
                for x in self.vhat_L.iintegrate(nt_g, W_L):
                    yield None
            for x in self.ghat_L.iintegrate(vHt_g, W_L):
                yield None

            Epot = U + s.M + npy.dot(D_p, (s.M_p + npy.dot(s.M_pp, D_p)))

            F_Lc = npy.zeros(((lmax + 1)**2, 3))
            self.ghat_L.derivative(vHt_g, F_Lc)
            if self.vhat_L is not None:
                self.vhat_L.derivative(nt_g, F_Lc)

            dF = npy.zeros(((lmax + 1)**2, 3))
            for neighbor in self.neighbors:
                for c in range(3):
                    dF[:, c] += npy.dot(neighbor.dvdr_LLc[:, :, c],
                                        neighbor.nucleus().Q_L)
            F += npy.dot(self.Q_L, dF)


class Neighbor:
    def __init__(self, v_LL, dvdr_LLc, nucleus):
        self.nucleus = weakref.ref(nucleus)
        self.v_LL = v_LL
        self.dvdr_LLc = dvdr_LLc


class PairPotential:
    def __init__(self, setups):
        # Collect the pair potential cutoffs in a list:
        self.cutoff_a = []
        for setup in setups:
            self.cutoff_a.append((setup.symbol, setup.rcutsoft))

        # Make pair interactions:
        self.interactions = {}
        for setup1 in setups:
            for setup2 in setups:
                assert setup1 is setup2 or setup1.symbol != setup2.symbol
                interaction = GInteraction(setup1, setup2)
                self.interactions[(setup1.symbol, setup2.symbol)] = interaction

        self.neighborlist = None

        self.need_neighbor_list = False
        for setup in setups:
            if setup.type == 'ae':
                self.need_neighbor_list = True
                break

    def update(self, pos_ac, nuclei, domain, text):
        if not self.need_neighbor_list:
            return

        if self.neighborlist is None:
            # Make a neighbor list object:
            symbol_a = [nucleus.setup.symbol for nucleus in nuclei]
            self.neighborlist = NeighborList(symbol_a, pos_ac, domain,
                                             self.cutoff_a)
        else:
            updated = self.neighborlist.update_list(pos_ac)
            if updated:
                text('Neighbor list has been updated!')

        # Reset all pairs:
        for nucleus in nuclei:
            nucleus.neighbors = []

        # Make new pairs:
        cell_c = domain.cell_c
        for a1 in range(len(nuclei)):
            nucleus1 = nuclei[a1]
            symbol1 = nucleus1.setup.symbol
            for a2, offsets in self.neighborlist.neighbors(a1):
                nucleus2 = nuclei[a2]
                symbol2 = nucleus2.setup.symbol
                interaction = self.interactions[(symbol1, symbol2)]
                diff_c = pos_ac[a2] - pos_ac[a1]
                V_LL = npy.zeros(interaction.v_LL.shape)  #  XXXX!
                dVdr_LLc = npy.zeros(interaction.dvdr_LLc.shape)
                r_c = pos_ac[a2] - cell_c / 2
                for offset in offsets:
                    d_c = diff_c + offset
                    v_LL, dvdr_LLc = interaction(d_c)
                    V_LL += v_LL
                    dVdr_LLc += dvdr_LLc
                nucleus1.neighbors.append(Neighbor(V_LL, dVdr_LLc, nucleus2))
                if nucleus2 is not nucleus1:
                    nucleus2.neighbors.append(
                        Neighbor(npy.transpose(V_LL),
                                 -npy.transpose(dVdr_LLc, (1, 0, 2)),
                                 nucleus1))

    def print_info(self, out):
        print >> out, 'pair potential:'
        print >> out, '  cutoffs:'
        print >> out, '   ', setup.symbol, setup.rcutsoft * a0
        npairs = self.neighborlist.number_of_pairs() - len(pos_ac)
        if npairs == 0:
            print >> out, '  There are no pair interactions.'
        elif npairs == 1:
            print >> out, '  There is one pair interaction.'
        else:
            print >> out, '  There are %d pair interactions.' % npairs

GAUSS = False


d_l = [fac[l] * 2**(2 * l + 2) / sqrt(pi) / fac[2 * l + 1]
       for l in range(3)]

class GInteraction2:
    def __init__(self, setupa, setupb):
        self.softgauss = setupa.softgauss
        self.alpha = setupa.alpha
        self.beta = setupb.alpha
        self.alpha2 = setupa.alpha2
        self.beta2 = setupb.alpha2
        self.lmaxa = setupa.lmax
        self.lmaxb = setupb.lmax
        self.v_LL = npy.zeros(((self.lmaxa + 1)**2, (self.lmaxb + 1)**2))
        self.dvdr_LLc = npy.zeros(((self.lmaxa + 1)**2,
                                  (self.lmaxb + 1)**2,
                                  3))

##         rcutcomp = setupa.rcutcomp + setupb.rcutcomp
##         rcutfilter = setupa.rcutfilter + setupb.rcutfilter
##         rcutproj = max(setupa.rcut_j) + max(setupb.rcut_j)
##         rcore = setupa.rcore + setupb.rcore
##         self.cutoffs = ('Summed cutoffs: %4.2f(comp), %4.2f(filt), '
##                         '%4.2f(core), %4.2f(proj) Bohr' % (
##             rcutcomp, rcutfilter, rcore, rcutproj))
##         self.mindist = rcutproj * .6

    def __call__(self, R):
##         dist = sqrt(npy.sum(R**2))
##         if dist > 0 and dist < self.mindist:
##             from sys import stderr
##             print >> stderr, warning('Atomic distance: %4.2f Bohr.\n%s' % (
##                 dist, self.cutoffs))

        if not self.softgauss:
            return (self.v_LL, -self.dvdr_LLc)
        for la in range(self.lmaxa + 1):
            for ma in range(2 * la + 1):
                La = la**2 + ma
                for lb in range(self.lmaxb + 1):
                    for mb in range(2 * lb + 1):
                        Lb = lb**2 + mb
                        f = npy.zeros(4)
                        f2 = npy.zeros(4)
                        for ca, xa in YL[La]:
                            for cb, xb in YL[Lb]:
                                f += ca * cb * I(R, xa, xb,
                                                 self.alpha, self.beta)
                                f2 += ca * cb * I(R, xa, xb,
                                                 self.alpha2, self.beta2)
                        x = d_l[la] * d_l[lb]
                        f *= x * self.alpha**(1.5 + la) * \
                                 self.beta**(1.5 + lb)
                        f2 *= x * self.alpha2**(1.5 + la) * \
                                  self.beta2**(1.5 + lb)
##                         if npy.sometrue(R):
##                             assert npy.dot(R, R) > 0.1
##                         else:
##                             f[:] = 0.0
##                             if La == Lb:
##                                 f[0] = I_l[la] / self.rcut**(2 * la + 1)

                        self.v_LL[La, Lb] = f[0] - f2[0]
                        self.dvdr_LLc[La, Lb] = f[1:] - f2[1:]
        return (self.v_LL, -self.dvdr_LLc)
"""
