import numpy as np
from ase.data import atomic_numbers

from gpaw.utilities import pack2, erf
from gpaw.utilities.tools import md5_new
from gpaw.setup import BaseSetup
from gpaw.spline import Spline


def screen_potential(r, v, charge):
    """Split long-range potential into short-ranged contributions.

    The potential v is a long-ranted potential with the asymptotic form Z/r
    corresponding to the given charge.
    
    Return a potential vscreened and charge distribution rhocomp such that

      v(r) = vscreened(r) + vHartree[rhocomp](r).

    The returned quantities are truncated to a reasonable cutoff radius.
    """
    vr = v * r + charge # XXX 4
    
    err = 0.0
    i = len(vr)
    while err < 1e-6:
        i -= 1
        err = abs(vr[i])
    i += 1
    
    icut = np.searchsorted(r, r[i] * 1.1)
    rcut = r[icut]
    rshort = r[:icut]
    
    a = rcut / 4.0
    vcomp = charge * erf(rshort / (np.sqrt(2.0) * a)) / rshort
    # XXX divide by r
    rhocomp = charge * (np.sqrt(2.0 * np.pi) * a)**(-3) * \
        np.exp(-0.5 * (rshort / a)**2)
    vscreened = v[:icut] + vcomp
    return vscreened, rhocomp


def pseudoplot(pp):
    import pylab as pl
    
    fig = pl.figure()
    wfsax = fig.add_subplot(221)
    ptax = fig.add_subplot(222)
    vax = fig.add_subplot(223)
    rhoax = fig.add_subplot(224)

    def spline2grid(spline):
        rcut = spline.get_cutoff()
        r = np.linspace(0.0, rcut, 2000)
        return r, spline.map(r)

    for phit in pp.phit_j:
        r, y = spline2grid(phit)
        wfsax.plot(r, y, label='wf l=%d' % phit.get_angular_momentum_number())

    for pt in pp.pt_j:
        r, y = spline2grid(pt)
        ptax.plot(r, y, label='pr l=%d' % pt.get_angular_momentum_number())

    for ghat in pp.ghat_l:
        r, y = spline2grid(ghat)
        rhoax.plot(r, y, label='cc l=%d' % ghat.get_angular_momentum_number())

    r, y = spline2grid(pp.vbar)
    vax.plot(r, y, label='vbar')
    
    vax.set_ylabel('potential')
    rhoax.set_ylabel('density')
    wfsax.set_ylabel('wfs')
    ptax.set_ylabel('projectors')

    for ax in [vax, rhoax, wfsax, ptax]:
        ax.legend()

    pl.show()

class PseudoPotential(BaseSetup):
    def __init__(self, data, basis):
        self.data = data

        self.R_sii = None
        self.HubU = None
        self.lq = None

        self.filename = None
        self.fingerprint = None
        self.symbol = data.symbol
        self.type = data.name

        self.Z = data.Z
        self.Nv = data.Nv
        self.Nc = data.Nc

        self.ni = sum([2 * l + 1 for l in data.l_j])
        self.pt_j = data.get_projectors()
        self.phit_j = basis.tosplines()
        self.basis = basis
        self.nao = sum([2 * phit.get_angular_momentum_number() + 1
                        for phit in self.phit_j])

        self.Nct = 0.0
        self.nct = Spline(0, 1.0, [0., 0., 0.])

        self.lmax = 0

        self.xc_correction = None

        r, l_comp, g_comp = data.get_compensation_charge_functions()
        self.ghat_l = [Spline(l, r[-1], g) for l, g in zip(l_comp, g_comp)]
        #self.ghat_l = [Spline(0, r[-1], g)]
        self.rcgauss = data.rcgauss

        # accuracy is rather sensitive to this
        self.vbar = data.get_local_potential()

        _np = self.ni * (self.ni + 1) // 2
        self.Delta0 = data.Delta0
        self.Delta_pL = np.zeros((_np, 1))

        self.E = 0.0
        self.Kc = 0.0
        self.M = 0.0
        self.M_p = np.zeros(_np)
        self.M_pp = np.zeros((_np, _np))

        self.K_p = data.expand_hamiltonian_matrix()
        self.MB = 0.0
        self.MB_p = np.zeros(_np)
        self.dO_ii = np.zeros((self.ni, self.ni))

        self.f_j = data.f_j
        self.n_j = data.n_j
        self.l_j = data.l_j
        self.nj = len(data.l_j)

        # We don't really care about these variables
        self.rcutfilter = None
        self.rcore = None

        self.N0_p = np.zeros(_np) # not really implemented
        self.nabla_iiv = None
        self.rnabla_iiv = None
        self.rxp_iiv = None
        self.phicorehole_g = None
        self.rgd = data.rgd
        self.rcut_j = data.rcut_j
        self.tauct = None
        self.Delta_iiL = None
        self.B_ii = None
        self.dC_ii = None
        self.X_p = None
        self.ExxC = None
        self.dEH0 = 0.0
        self.dEH_p = np.zeros(_np)
        self.extra_xc_data = {}

        self.wg_lg = None
        self.g_lg = None
