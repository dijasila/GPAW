# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.
from math import log, pi

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!

import gpaw.mpi as mpi
from gpaw.recursionmethod import RecursionMethod


def xas(paw):
    assert not mpi.parallel
    assert not paw.spinpol # restricted - for now

    nocc = paw.nvalence / 2
    for nucleus in paw.nuclei:
        if nucleus.setup.fcorehole != 0.0:
            break

    A_ci = nucleus.setup.A_ci

    n = paw.nbands - nocc
    eps_n = num.empty(paw.nkpts * n, num.Float)
    w_cn = num.empty((3, paw.nkpts * n), num.Float)
    n1 = 0
    for k in range(paw.nkpts):
        n2 = n1 + n
        eps_n[n1:n2] = paw.kpt_u[k].eps_n[nocc:] * paw.Ha
        P_ni = nucleus.P_uni[k, nocc:]
        a_cn = inner(A_ci, P_ni)
        a_cn *= num.conjugate(a_cn)
        w_cn[:, n1:n2] = paw.weight_k[k] * a_cn.real
        n1 = n2
        
    if paw.symmetry is not None:
        w0_cn = w_cn
        w_cn = num.zeros((3, paw.nkpts * n), num.Float)
        swaps = {}  # Python 2.4: use a set
        for swap, mirror in paw.symmetry.symmetries:
            swaps[swap] = None
        for swap in swaps:
            w_cn += num.take(w0_cn, swap)
        w_cn /= len(swaps)
        print swaps, paw.symmetry.symmetries
        print w_cn
    
    return eps_n, w_cn


class XAS:

    def __init__(self,calc):
        self.calc = calc

        # create initial wave function  psitch_cG
        self.psitch_cG = calc.gd.zeros(3)
        for nucleus in calc.nuclei:
            if nucleus.setup.fcorehole != 0.0:
                A_ci = nucleus.setup.A_ci
                ach = nucleus.a
                print "ach", ach
                if nucleus.pt_i is not None: # not all CPU's will have a contribution
                    print "added to psitch_cG"
                    nucleus.pt_i.add(self.psitch_cG, A_ci)
                    break
                
        #for x in self.psitch_cG[0]:
        #    print x
        
        self.rec = RecursionMethod(self.psitch_cG, self.calc)
        #self.rec_x = RecursionMethod(self.psitch_cG[0:1], self.calc)
        #self.rec_y = RecursionMethod(self.psitch_cG[1:2], self.calc)
        #self.rec_z = RecursionMethod(self.psitch_cG[2:3], self.calc)
    
    def run(nmax_iter=1000, tol=10e-11):
        asd

    def write(self,file = "xas.pickl"):
        import pickle
        a = self.rec.a
        b = self.rec.b
        pickle.dump((a,b),open(file,'w'))

    def load(self,file ="xas.pickl" ):
        import pickle
        self.rec.a, self.rec.b  = pickle.load(open(file))



        



def xas_recursion(calc, e_start, e_step, n_e, broadening):
    assert not mpi.parallel

    # create initial wave function  psitch_cG
    psitch_cG = calc.gd.zeros(3)
    for nucleus in calc.nuclei:
        if nucleus.setup.fcorehole != 0.0:
            #P_ni = nucleus.P_uni[0, nocc:] 
            A_ci = nucleus.setup.A_ci
            ach = nucleus.a
            print "ach", ach
            #print nucleus.pt_i
            
            if nucleus.pt_i is not None: # not all CPU's will have a contribution
                print "added to psitch_cG"
                nucleus.pt_i.add(psitch_cG, A_ci)
            break
    
    # call recursion method
    rec = RecursionMethod(psitch_cG, calc)
    e_vector =  e_start + range(n_e)*e_step
    intensity = []
    for e, ind in e_vector:
        c_frac = rec.cont_frac( complex(e, broadening), 1)
        intensity.append( - c_frac.imag / pi)

    return e_vector, intensity
    


#psitch_cG = gd.zeros(3)
#nucleus = nuclei[a]
#if nucleus.pt_i is not None: # not all CPU's will have a contribution
#    nucleus.pt_i.add(psitch_cG, A_ci)

def plot_xas(eps_n, w_n, fwhm=0.5, linbroad=None, N=1000):
    # returns stick spectrum, e_stick and a_stick
    # and broadened spectrum, e, a
    # linbroad = [0.5, 540, 550]
    eps_n_tmp = eps_n.copy()
    emin = min(eps_n_tmp) - 2 * fwhm
    emax = max(eps_n_tmp) + 2 * fwhm

    e = emin + num.arange(N + 1) * ((emax - emin) / N)
    a = num.zeros(N + 1, num.Float)

    e_stick = eps_n_tmp.copy()
    a_stick = num.zeros(len(eps_n_tmp), num.Float)


    if linbroad is None:
        #constant broadening fwhm
        alpha = 4*log(2) / fwhm**2
        for n, eps in enumerate(eps_n_tmp):
            x = -alpha * (e - eps)**2
            x = num.clip(x, -100.0, 100.0)
            w = w_n[n]
            a += w * (alpha / pi)**0.5 * num.exp(x)
            a_stick[n] = w
    else:
        # constant broadening fwhm until linbroad[1] and a constant broadening
        # over linbroad[2] with fwhm2= linbroad[0]
        fwhm2 = linbroad[0]
        lin_e1 = linbroad[1]
        lin_e2 = linbroad[2]
        for n, eps in enumerate(eps_n_tmp):
            if eps < lin_e1:
                alpha = 4*log(2) / fwhm**2
            elif eps <=  lin_e2:
                fwhm_lin = fwhm + (eps - lin_e1) * (fwhm2 - fwhm) / (lin_e2 - lin_e1)
                alpha = 4*log(2) / fwhm_lin**2
            elif eps >= lin_e2:
                alpha =  4*log(2) / fwhm2**2

            x = -alpha * (e - eps)**2
            x = num.clip(x, -100.0, 100.0)
            w = w_n[n]
            a += w * (alpha / pi)**0.5 * num.exp(x)
            a_stick[n] = w
        
    return e_stick, a_stick, e, a
