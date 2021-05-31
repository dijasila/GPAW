from gpaw.xc.functional import XCFunctional
import numpy as np
from gpaw.utilities.tools import construct_reciprocal
import gpaw.mpi as mpi
from functools import wraps
from time import time
from ase.parallel import parprint
from enum import Enum

def timer(f):

    @wraps(f)
    def wrapped(*args):
        t1 = time()
        res = f(*args)
        t2 = time()
        parprint(f"{f.__name__} took {t2-t1} seconds")
        return res

    return wrapped

def make_spline_coefficients(alphas):
    n = len(alphas)
    y = np.eye(n)
    a = y
    h = alphas[1:] - alphas[:-1]
    alph = 3 * ((a[2:] - a[1:-1]) / h[1:, np.newaxis] -
                (a[1:-1] - a[:-2]) / h[:-1, np.newaxis])

    l = np.zeros((n, n))
    mu = np.zeros((n, n))
    z = np.zeros((n, n))
    for i in range(1, n - 1):
        l[i] = 2 * (alphas[i + 1] - alphas[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alph[i - 1] - h[i - 1] * z[i - 1]) / l[i]

    b = np.zeros((n, n))
    c = np.zeros((n, n))
    d = np.zeros((n, n))

    for i in range(n - 2, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) /3 / h[i]

    C_aip = np.zeros((n, n, 4))
    C_aip[:, :-1, 0] = a[:-1].T
    C_aip[:, :-1, 1] = b[:-1].T
    C_aip[:, :-1, 2] = c[:-1].T
    C_aip[:, :-1, 3] = d[:-1].T
    C_aip[-1, -1, 0] = 1.0

    return C_aip                     



def define_indicator(ia, alphas, C_aip):
    nalphas = len(alphas)
    na = np.logical_and
    def _i(n_g):
        C_ip = C_aip[ia]
        res_g = np.zeros_like(n_g)
        if ia < nalphas - 1 and ia != 0:
            inds = na(n_g >= alphas[ia], n_g < alphas[ia+1])
            inds1 = []
        elif ia == 0:
            inds = n_g <= alphas[1]
            inds1 = n_g <= alphas[0]
        else:
            assert ia == nalphas - 1
            inds = n_g >= alphas[nalphas - 2]
            inds1 = n_g >= alphas[nalphas - 1]

        a, b, c, d = C_ip[ia]
        d_g = (n_g[inds] - alphas[ia])
        res_g[inds] += a + d_g * (b + d_g * (c + d_g * d))

        res_g[inds1] = 1.0
        return res_g

    def _di(n_g):
        C_ip = C_aip[ia]
        res_g = np.zeros_like(n_g)
        if ia < nalphas - 1 and ia != 0:
            inds = na(n_g >= alphas[ia], n_g < alphas[ia+1])
            inds1 = []
        elif ia == 0:
            inds = n_g <= alphas[1]
            inds1 = n_g <= alphas[0]
        else:
            assert ia == nalphas - 1
            inds = n_g >= alphas[nalphas - 2]
            inds1 = n_g >= alphas[nalphas - 1]

        a, b, c, d = C_ip[ia]
        d_g = (n_g[inds] - alphas[ia])
        res_g[inds] += d_g * (b + d_g * (2.0 * c + 3.0 * d_g * d))
        res_g[inds1] = 0.0
        return res_g

    return _i, _di



def _define_indicator(ia, alphas, dummy):
    nalphas = len(alphas)
    na = np.logical_and
    if ia == 0:
        def _i(n_g):
            res_g = np.zeros_like(n_g)
            inds = n_g < alphas[1]
            res_g[inds] = (alphas[1] - n_g[inds]) / (alphas[1] - alphas[0])
            inds = n_g <= alphas[0]
            res_g[inds] = 1.0
            return res_g

        def _di(n_g):
            res_g = np.zeros_like(n_g)
            inds = n_g < alphas[1]
            res_g[inds] = -1.0 / (alphas[1] - alphas[0])
            inds = n_g <= alphas[0]
            res_g[inds] = 0.0
            return res_g

    elif ia == nalphas - 1:
        def _i(n_g):
            res_g = np.zeros_like(n_g)
            inds = n_g >= alphas[ia - 1]
            res_g[inds] = (n_g[inds] - alphas[ia - 1]) / \
                (alphas[-1] - alphas[-2])
            inds = n_g >= alphas[ia]
            res_g[inds] = 1.0
            return res_g

        def _di(n_g):
            res_g = np.zeros_like(n_g)
            inds = n_g >= alphas[ia - 1]
            res_g[inds] = 1.0 / (alphas[-1] - alphas[-2])
            inds = n_g >= alphas[ia]
            res_g[inds] = 0.0
            return res_g
    else:
        def _i(n_g):
            res_g = np.zeros_like(n_g)
            inds = na(n_g >= alphas[ia - 1], n_g < alphas[ia])
            res_g[inds] = (n_g[inds] - alphas[ia - 1]) / \
                (alphas[ia] - alphas[ia - 1])
            inds = na(n_g >= alphas[ia], n_g < alphas[ia + 1])
            res_g[inds] = (alphas[ia + 1] - n_g[inds]) / \
                (alphas[ia + 1] - alphas[ia])
            return res_g

        def _di(n_g):
            res_g = np.zeros_like(n_g)
            inds = na(n_g > alphas[ia - 1], n_g <= alphas[ia])
            res_g[inds] = 1.0 / (alphas[ia] - alphas[ia - 1])
            inds = na(n_g > alphas[ia], n_g < alphas[ia + 1])
            res_g[inds] = -1.0 / (alphas[ia + 1] - alphas[ia])
            return res_g

    return _i, _di


def define_alphas(n_sg, nalphas):
    minn = np.min(n_sg)
    maxn = np.max(n_sg)
    if np.allclose(minn, maxn):
        maxn *= 1.05
    # The indicator anchors
    minanchor = 1e-6  # 1e-4 * 200 / nalphas + 5*1e-4
    minn = max(minn, minanchor)
    # alphas = np.linspace(minn, maxn, nalphas)
    alphas = np.exp(np.linspace(np.log(minn), np.log(maxn), nalphas))

    return alphas

class Modes(Enum):
    WLDA = 'WLDA'
    rWLDA = 'rWLDA'
    fWLDA = 'fWLDA'


class WfTypes(Enum):
    lorentz = "lorentz"
    gauss = "gauss"
    exponential = "exponential"
    diracdelta = "diracdelta"

class DensityTypes(Enum):
    pseudo = "pseudo"
    smoothAE = "smoothAE"
    AE = "AE"


class WLDA(XCFunctional):
    def __init__(self, settings):
        """Init WLDA.

        Allowed keys and values for settings:
        weightfunction: "lorentz", "gauss", "exponential"
        c1: positive float
        hartreexc: bool
        """
        XCFunctional.__init__(self, 'WLDA', 'LDA')
        # MODE SETTINGS
        mode = settings.get('mode', 'fWLDA')
        if mode.endswith("x"):
            self.exchange_only = True
            mode = mode[:-1]
        else:
            self.exchange_only = False

        assert mode in ['WLDA', 'rWLDA', 'fWLDA']
        self.mode = getattr(Modes, mode)        
        
        wftype = settings.get('wftype', 'exponential')
        assert wftype in ['lorentz', 'gauss', 'exponential', 'diracdelta']
        self.wftype = getattr(WfTypes, wftype)

        self.c1 = settings.get('c1', None)
        if self.c1 is None:
            assert self.wftype == WfTypes.exponential or self.wftype == WfTypes.diracdelta
            if self.mode == Modes.WLDA:
                self.c1 = 7.57576
            elif self.mode == Modes.rWLDA:
                self.c1 = 5.0
            else:
                self.c1 = 4.80769
        assert self.c1 > 0

        from gpaw.xc.lda import LDA, PurePythonLDAKernel
        self.lda_xc = LDA(PurePythonLDAKernel())

        self.lambd = settings.get('lambda', 2.0/3.0)

        self.hxc = settings.get('hxc', True)
        assert type(self.hxc) == bool

        self.use_hartree_correction = settings.get('use_hartree_correction', True)

        # NUMERICAL PARAMETERS
        self.nindicators = settings.get('nindicators', 50)

        density_type = settings.get('density_type', 'pseudo')
        assert density_type in ['pseudo', 'smoothAE', 'AE'], density_type
        self.density_type = getattr(DensityTypes, density_type)
        assert self.density_type in [DensityTypes.pseudo, DensityTypes.smoothAE, DensityTypes.AE]

        self.atoms = settings.get("atoms", None)
        if self.density_type == DensityTypes.AE:
            assert self.atoms is not None

        self.rcut_factor = float(settings.get("rc", 0.8))

        self.save = settings.get("save", False)
        self.saveindex = 0
        
        # self.tolerance = settings.get("tolerance", 1e-8)
        self.T = settings.get("T", 0.01)
        self.mu = settings.get("mu", 20)

        # Preparation and logging
        self.settings = None # settings
        self.log_parameters()

        self._K_G = None

    def log_parameters(self):
        """Log all parameters.
        
        Because there are so many parameters we log them in
        a file with a unique name for debugging and verification
        purposes."""

        # This is definitely the best way to do it
        unique_id = str(np.random.rand()).split(".")[1]
        fname = f'WLDA-{unique_id}.log'

        # Start with dumping settings
        # then write specific parameters for easy reading
        # f'settings: {self.settings.items()}',
        msg = [f'Mode: {self.mode}',
               f'Exchange only: {self.exchange_only}',
               f'wftype: {self.wftype}',
               f'c1: {self.c1}',
               f'lambda: {self.lambd}',
               f'Density type: {self.density_type.value}',
               f'nindicators: {self.nindicators}',
               f'rcut_factor: {self.rcut_factor}',
               f'Use Hartree corr.: {self.use_hartree_correction}',
               f'Treat Hartree corr. as X energy: {self.hxc}']
        if mpi.rank == 0:
            with open(fname, 'w+') as f:
                f.write('\n'.join(msg))

    def initialize(self, density, hamiltonian, wfs, occupations=None):
        self.density = density
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.occupations = occupations

    def do_lda(self, n_sg, wn_sg, gd):
        """Calculate two version of LDA.

        First calculate LDA on the pseudo density.
        This forms the first part of the WLDA energy:
        E^WLDA = E^LDA + ...

        Then calculate LDA on the "working density".
        This enters in the correction term:
        E^WLDA = ... + Delta E^WLDA.
        """
        eldax_g = np.zeros_like(n_sg[0])
        eldac_g = np.zeros_like(n_sg[0])
        vldax_sg = np.zeros_like(n_sg)
        vldac_sg = np.zeros_like(n_sg)

        we_corr_x_g = np.zeros_like(wn_sg[0])
        we_corr_c_g = np.zeros_like(wn_sg[0])
        wv_corr_x_sg = np.zeros_like(wn_sg)
        wv_corr_c_sg = np.zeros_like(wn_sg)

        self.calculate_lda(n_sg,
                           eldax_g, eldac_g,
                           vldax_sg, vldac_sg)

        self.calculate_lda(wn_sg,
                           we_corr_x_g, we_corr_c_g,
                           wv_corr_x_sg, wv_corr_c_sg)

        # Now define arrays to hold corrections and
        # re-distribute corrections across nodes
        e_corr_x_g = np.zeros_like(n_sg[0])
        e_corr_c_g = np.zeros_like(n_sg[0])
        v_corr_x_sg = np.zeros_like(n_sg)
        v_corr_c_sg = np.zeros_like(n_sg)

        gd.distribute(we_corr_x_g, e_corr_x_g)
        gd.distribute(we_corr_c_g, e_corr_c_g)
        gd.distribute(wv_corr_x_sg, v_corr_x_sg)
        gd.distribute(wv_corr_c_sg, v_corr_c_sg)

        # Define convenience variables
        # The first arrays are simply LDA
        self.elda_g = eldax_g + eldac_g
        self.vlda_sg = vldax_sg + vldac_sg

        # The next arrays are used in Delta E^WLDA
        if self.exchange_only:
            self.e_corr_g = e_corr_x_g
            self.v_corr_sg = v_corr_x_sg
        else:
            self.e_corr_g = e_corr_x_g + e_corr_c_g
            self.v_corr_sg = v_corr_x_sg + v_corr_x_sg

    def calculate_lda(self, n_sg, eldax_g, eldac_g, vldax_sg, vldac_sg):
        from gpaw.xc.lda import lda_x, lda_c
        if len(n_sg) == 1:
            n = n_sg[0]
            n[n < 1e-20] = 1e-40

            lda_x(0, eldax_g, n, vldax_sg)

            lda_c(0, eldac_g, n, vldac_sg, 0)
        else:
            na = 2. * n_sg[0]
            na[na < 1e-20] = 1e-40
            nb = 2. * n_sg[1]
            nb[nb < 1e-20] = 1e-40
            n = 0.5 * (na + nb)
            zeta = 0.5 * (na - nb) / n

            lda_x(1, eldax_g, na, vldax_sg[0])
            lda_x(1, eldax_g, nb, vldax_sg[1])

            lda_c(1, eldac_g, n, vldac_sg, zeta)

    def sign_regularization(self, dv_sg):
        # Indices where Delta V_xc is less than 0.0
        indices = dv_sg < 0.0
        dv_sg[indices] = 0.0
        return dv_sg

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        """Interface for GPAW."""
        parprint("WLDA is the next PBE")
        wn_sg = self.get_working_density(n_sg, gd)

        # This function constructs arrays holding e.g.:
        # self.e_lda_g : LDA energy
        # self.e_corr_g : LDA energy on working density
        self.do_lda(n_sg, wn_sg, gd)

        # Construct arrays for un-distributed energy and potential
        exc_g = np.zeros_like(wn_sg[0])
        vxc_sg = np.zeros_like(wn_sg)

        nstar_sg, alpha_indices = self.wlda_x(wn_sg, exc_g, vxc_sg)

        if not self.exchange_only:
            nstar_sg = self.wlda_c(wn_sg, exc_g, vxc_sg, nstar_sg, alpha_indices)
            # nstar_sg is now weighted density from \int phi(r-r', n_total(r'))n_sigma(r')

        # vxc_sg = self.sign_regularization(vxc_sg, self.v_corr_sg)

        # Arrays for undistributed Hartree
        eHa_g = np.zeros_like(exc_g)
        vHa_sg = np.zeros_like(vxc_sg)
        # Arrays for distributed Hartree
        deHa_g = np.zeros_like(e_g)
        dvHa_sg = np.zeros_like(v_sg)
        self.hartree_correction(self.gd, eHa_g, vHa_sg,
                                wn_sg, nstar_sg, alpha_indices)

        # e_g holds the WLDA *energy*
        gd.distribute(exc_g, e_g)
        gd.distribute(vxc_sg, v_sg)
        # deHa holds the Hartree *correction*
        gd.distribute(eHa_g, deHa_g)
        gd.distribute(vHa_sg, dvHa_sg)


        if self.mode == Modes.fWLDA:
            e_g[:] = self.elda_g + self.lambd * (e_g - self.e_corr_g) + self.lambd * deHa_g
            v_sg[:] = (self.vlda_sg
                       + self.lambd * self.sign_regularization(v_sg - self.v_corr_sg)
                       + self.lambd * dvHa_sg)
        elif self.mode == Modes.rWLDA:
            e_g[:] = self.elda_g + self.lambd * (e_g - self.e_corr_g) + deHa_g
            v_sg[:] = (self.vlda_sg 
                       + self.lambd * self.sign_regularization(v_sg - self.v_corr_sg)
                       + dvHa_sg)
        else:
            assert self.mode == Modes.WLDA
            e_g[:] = self.elda_g + (e_g - self.e_corr_g) + deHa_g
            v_sg[:] = (self.vlda_sg 
                       + self.sign_regularization(v_sg - self.v_corr_sg) + dvHa_sg)

        # If saving is enabled we save some arrays for debugging
        if self.save and mpi.rank == 0:
            tid = np.random.rand()
            np.save(f"n_sg{tid}.npy", n_sg)
            np.save(f"wn_sg{tid}.npy", wn_sg)
            np.save(f"nstar_sg{tid}.npy", nstar_sg)
            np.save(f"v_sg{tid}.npy", v_sg)
            np.save(f"vxc_sg{tid}.npy", vxc_sg)

    def get_working_density(self, n_sg, gd):
        """Construct the "working" density.

        The working density is an approximation to the AE density
        (or sometimes not depending on the selected mode).

        The construction consists of the following steps:

        1. Correct negative density values.
        2. Collect density, so all ranks have full density.
        3. Construct new grid_descriptor for collected density.
        4. Apply approximation to AE density (or not).

        The approximation scheme involves truncating the AE density
        at points closer than some radius of the atoms and replacing full
        density with a smooth polynomial. This procedure is implemented
        in the setup class (gpaw/setup.py).
        """
        n_sg[n_sg < 1e-20] = 1e-40
        # n2_sg = gd.collect(n_sg, broadcast=True)
        

        wn_sg, gd1 = self.density_correction(gd, n_sg)
        wn_sg[wn_sg < 1e-20] = 1e-20
        # gd1 = gd.new_descriptor(comm=mpi.serial_comm)
        self.gd = gd1

        return wn_sg

    def wlda_x(self, wn_sg, exc_g, vxc_sg, return_potentials=False):
        """
        Calculate WLDA exchange energy.

        If the system is spin-polarized, we use the formula
            E_x[n_up, n_down] = (E[2n_up] + E[2n_down])/2

        If the system is spin-paired, we can reuse the
        weighted density because we don't calculate the
        exchange functional with 2 times the density.

        Thus

        If spin-paired return density and alpha_indices

        If spin-polarized return nothing (None)
        """
        if len(wn_sg) == 2:
            # If spin-polarized calculate exchange energy according to
            # E[n_up, n_down] = (E[2n_up] + E[2n_down]) / 2
            wn_sg *= 2.0

        nstar_sg, alpha_indices = self.get_weighted_density(wn_sg)

        e1_g = np.zeros_like(wn_sg[0])
        e2_g = np.zeros_like(wn_sg[0])
        e3_g = np.zeros_like(wn_sg[0])
        v1_sg = np.zeros_like(wn_sg)
        v2_sg = np.zeros_like(wn_sg)
        v3_sg = np.zeros_like(wn_sg)

        if len(wn_sg) == 1:
            spin = 0
            self.lda_x1(spin, e1_g, wn_sg[0], nstar_sg[0], v1_sg[0],
                        alpha_indices)

            self.lda_x2(spin, e2_g, wn_sg[0], nstar_sg[0], v2_sg[0],
                        alpha_indices)

            self.lda_x3(spin, e3_g, wn_sg[0], nstar_sg[0], v3_sg[0],
                        alpha_indices)
            exc_g[:] = e1_g + e2_g - e3_g
            vxc_sg[:] = v1_sg + v2_sg - v3_sg
            return nstar_sg, alpha_indices
        else:
            spin = 1
            self.lda_x1(spin, e1_g, wn_sg[0], nstar_sg[0], v1_sg[0],
                        alpha_indices)
            self.lda_x1(spin, e1_g, wn_sg[1], nstar_sg[1], v1_sg[1],
                        alpha_indices)

            self.lda_x2(spin, e2_g, wn_sg[0], nstar_sg[0], v2_sg[0],
                        alpha_indices)
            self.lda_x2(spin, e2_g, wn_sg[1], nstar_sg[1], v2_sg[1],
                        alpha_indices)

            self.lda_x3(spin, e3_g, wn_sg[0], nstar_sg[0], v3_sg[0],
                        alpha_indices)
            self.lda_x3(spin, e3_g, wn_sg[1], nstar_sg[1], v3_sg[1],
                        alpha_indices)

            exc_g[:] = e1_g + e2_g - e3_g
            vxc_sg[:] = v1_sg + v2_sg - v3_sg
            if return_potentials:
                return nstar_sg, e1_g, e2_g, e3_g, v1_sg, v2_sg, v3_sg
            else:
                return nstar_sg, None

    def wlda_c(self, wn_sg, exc_g, vxc_sg, nstar_sg, alpha_indices, return_potentials=False):
        """Calculate the WLDA correlation energy."""
        if nstar_sg is None or alpha_indices is None:
            assert len(wn_sg) == 2
            n_g = (wn_sg[0] + wn_sg[1]) * 0.5
            nstar_sg, alpha_indices = self.get_weighted_density_for_spinpol_correlation(n_g, wn_sg * 0.5)
        else:
            assert len(wn_sg) == 1
            self.alphadensity_g = wn_sg[0]

        e1_g = np.zeros_like(wn_sg[0])
        e2_g = np.zeros_like(wn_sg[0])
        e3_g = np.zeros_like(wn_sg[0])
        v1_sg = np.zeros_like(wn_sg)
        v2_sg = np.zeros_like(wn_sg)
        v3_sg = np.zeros_like(wn_sg)

        if len(wn_sg) == 1:
            spin = zeta = 0
            self.lda_c1(0, e1_g, wn_sg[0], nstar_sg,
                        v1_sg[0], zeta, alpha_indices)
            self.lda_c2(0, e2_g, wn_sg[0], wn_sg, nstar_sg,
                        v2_sg[0], zeta, alpha_indices)
            self.lda_c3(0, e3_g, wn_sg[0], wn_sg, nstar_sg,
                        v3_sg[0], zeta, alpha_indices)
        else:
            spin = 1
            zeta_g = (wn_sg[0] - wn_sg[1]) / (wn_sg[0] + wn_sg[1])
            zeta_g[np.isclose(wn_sg[0] + wn_sg[1], 0.0)] = 0.0

            zetastar_g = (nstar_sg[0] - nstar_sg[1]) / (nstar_sg[0] + nstar_sg[1])
            zetastar_g[np.isclose(nstar_sg[0] + nstar_sg[1], 0.0)] = 0.0

            self.lda_c1(spin, e1_g, n_g, nstar_sg, v1_sg, zeta_g,
                        alpha_indices)
            self.lda_c2(spin, e2_g, n_g, wn_sg * 0.5, nstar_sg, v2_sg, zetastar_g,
                        alpha_indices)
            self.lda_c3(spin, e3_g, n_g, wn_sg * 0.5, nstar_sg, v3_sg, zetastar_g,
                        alpha_indices)

        exc_g[:] += e1_g + e2_g - e3_g
        vxc_sg[:] += v1_sg + v2_sg - v3_sg
        if return_potentials:
            return nstar_sg, e1_g, e2_g, e3_g, v1_sg, v2_sg, v3_sg
        else:
            return nstar_sg

    def get_weighted_density(self, wn_sg):
        """Set up the weighted density.

        This is done by applying the mapping

            n |-> n*(r) = int phi(r-r', n(r')) n(r')dr'
        This is done via the convolution theorem by replacing

            phi(r-r', n(r')) -> sum_a phi(r-r', a)f_a(r')n(r')

        f_a(r') is close to one if n(r') is close to a.

        Each rank has some subset of a's.

        First construct and distribute the as across the ranks.

        Then apply the weighting functional.

        Sum across ranks so all ranks have
        the full weighted density.

        Finally correct potentially negative values from
        numerically inaccuracy to avoid complex energies.
        """
        alpha_indices = self.construct_alphas(wn_sg)
        assert len(alpha_indices) == 0 or type(alpha_indices[0]) == int
        if self.nindicators < mpi.size and mpi.rank > self.nindicators:
            assert len(alpha_indices) == 0

        nstar_sg = self.alt_weight(wn_sg, alpha_indices, self.gd)

        mpi.world.sum(nstar_sg)

        nstar_sg[nstar_sg < 1e-20] = 1e-40

        return nstar_sg, alpha_indices

    def get_weighted_density_for_spinpol_correlation(self, ntotal_g, wn_sg):
        """Get the weighted density for correlation.
        
        Only applicable in the spin-polarized case.

        Calculates

        n*_up = \int phi(r - r', n_total(r')) n_up(r') dr'
        n*_up = \sum_alpha \int phi(r - r', alpha) f_alpha(n_total(r')) * n_up(r') dr'

        and similarly for n_down.
        """
        
        # ntotal_g = wn_sg.sum(axis=0)

        alpha_indices = self.construct_alphas_for_spinpol(ntotal_g)
        assert len(alpha_indices) == 0 or type(alpha_indices[0]) == int

        if self.nindicators < mpi.size and mpi.rank > self.nindicators:
            assert len(alpha_indices) == 0

        nstar_sg = self.alt_weight_for_spinpol(ntotal_g, alpha_indices, wn_sg, self.gd)
        mpi.world.sum(nstar_sg)

        nstar_sg[nstar_sg < 1e-20] = 1e-40

        return nstar_sg, alpha_indices
                
    def construct_alphas_for_spinpol(self, ntotal_g):
        alpha_indices = self.distribute_alphas(self.nindicators, mpi.rank, mpi.size)
        self.alphas = define_alphas(ntotal_g, self.nindicators)

        self.alphadensity_g = ntotal_g

        return alpha_indices

    def construct_alphas(self, wn_sg):
        """Construct indicator data.

        Construct alphas that are used to approximately
        calculate the weighted density.

        The alphas should cover all the values of the
        density. A denser grid should be more accurate,
        but will be more computationally expensive.

        After setting up the full grid of alphas, each rank
        determines which alphas it should calculate.
        """
        alpha_indices = self.distribute_alphas(self.nindicators,
                                               mpi.rank, mpi.size)
        self.alphas = define_alphas(wn_sg, self.nindicators)
        self.C_aip = make_spline_coefficients(self.alphas)
        self.wn_sg = wn_sg

        return alpha_indices

    def ind_asg(self, ia, spin, density):
        if type(ia) != int and type(ia) != list:
            raise ValueError(f"Incorrect type: {ia}, {type(ia)}")

        if spin is None:
            assert type(ia) == int
            _i, _ = define_indicator(ia, self.alphas, self.C_aip)
            return _i(density)

        if type(ia) == list:
            _is = [define_indicator(x, self.alphas, self.C_aip)[0] for x in ia]
            if type(spin) == list:
                return np.array([[_i(density[s]) for s in spin]
                                 for _i in _is])
            else:
                return np.array([_i(density[spin]) for _i in _is])

        if type(ia) == int:
            _i, _ = define_indicator(ia, self.alphas, self.C_aip)

            if type(spin) == list:
                return np.array([_i(density[s]) for s in spin])
            else:
                return _i(density[spin])

    def dind_asg(self, ia, spin, density):
        _, _di = define_indicator(ia, self.alphas, self.C_aip)

        if spin is None:
            return _di(density)

        if type(spin) == list:
            return np.array([_di(density[s]) for s in spin])
        else:
            return _di(density[spin])

    def distribute_alphas(self, nindicators, rank, size):
        """Distribute alphas across mpi ranks."""
        if nindicators <= size:
            return range(rank, min(rank + 1, nindicators))

        nalphas = nindicators // size
        nalphas0 = nalphas + (nindicators - nalphas * size)
        assert (nalphas * (size - 1) + nalphas0 == nindicators)

        if rank == 0:
            start = 0
            end = nalphas0
        else:
            start = nalphas0 + (rank - 1) * nalphas
            end = start + nalphas

        return range(start, end)

    def alt_weight(self, wn_sg, my_alpha_indices, gd):
        """Calculate the weighted density.

        Calculate

            n*(r) = int phi(r-r', a)f_a(r')n('r)

        via the convolution theorem

            n*(r) = phi * (f_a n)
        """
        nstar_sg = np.zeros_like(wn_sg)

        for ia in my_alpha_indices:
            nstar_sg += self.apply_kernel(wn_sg, ia, gd)

        return nstar_sg

    def alt_weight_for_spinpol(self, ntotal_g, my_alpha_indices, wn_sg, gd):
        nstar_sg = np.zeros_like(wn_sg)

        for ia in my_alpha_indices:
            nstar_sg += self.apply_kernel_for_spinpol(ntotal_g, ia, wn_sg, gd)

        return nstar_sg

    def apply_kernel(self, wn_sg, ia, gd):
        """Apply the WLDA kernel at given alpha.

        Applies the kernel via the convolution theorem.
        """
        spins = list(range(wn_sg.shape[0]))
        # f_a(n_sigma(r))
        f_sg = self.ind_asg(ia, spins, self.wn_sg) * wn_sg
        assert wn_sg.shape == f_sg.shape, spins
        f_sG = self.fftn(f_sg, axes=(1, 2, 3))

        w_sG = np.array([self.get_weight_function(
            ia, gd, self.alphas) for _ in spins])
        assert w_sG.shape == f_sG.shape
        r_sg = self.ifftn(w_sG * f_sG, axes=(1, 2, 3))

        return r_sg.real

    def apply_kernel_for_spinpol(self, ntotal_g, ia, wn_sg, gd):
        f_sg = np.zeros_like(wn_sg)
        indicator_g = self.ind_asg(ia, None, ntotal_g)
        f_sg[0] = indicator_g * wn_sg[0]
        f_sg[1] = indicator_g * wn_sg[1]
        assert wn_sg.shape == f_sg.shape
        assert len(f_sg.shape) == 4
        f_sG = self.fftn(f_sg, axes=(1, 2, 3))

        w_G = self.get_weight_function(ia, gd, self.alphas)
        # Everytime there is a self which contains data
        # I have to make sure that the chain of calls leading
        # to this point is correct
        r_sg = np.zeros_like(wn_sg)
        r_sg[0] = self.ifftn(w_G * f_sG[0], axes=(0, 1, 2))
        r_sg[1] = self.ifftn(w_G * f_sG[1], axes=(0, 1, 2))

        return r_sg.real

    def fftn(self, arr, axes=None):
        """Interface for fftn."""
        # if axes is None:
        #     sqrtN = np.sqrt(np.array(arr.shape).prod())
        # else:
        #     sqrtN = 1
        #     for ax in axes:
        #         sqrtN *= arr.shape[ax]
        #     sqrtN = np.sqrt(sqrtN)

        return np.fft.fftn(arr, axes=axes)  # , norm="ortho") / sqrtN

    def ifftn(self, arr, axes=None):
        """Interface for ifftn."""
        # if axes is None:
        #     sqrtN = np.sqrt(np.array(arr.shape).prod())
        # else:
        #     sqrtN = 1
        #     for ax in axes:
        #         sqrtN *= arr.shape[ax]
        #     sqrtN = np.sqrt(sqrtN)

        return np.fft.ifftn(arr, axes=axes)  # , norm="ortho") * sqrtN

    def get_weight_function(self, ia, gd, alphas):
        """Evaluates and returns phi(q, alpha)."""
        alpha = alphas[ia]
        # c1 = self.settings.get("c1", None) or 5.8805
        # if c1 <= 0:
        #     raise ValueError(f'c1 must be positive. Got c1 = {c1}')

        K_G = self._get_K_G(gd)
        qt = K_G / (self.c1 * abs(alpha)**(1 / 3))

        if self.wftype == WfTypes.lorentz:
            return np.exp(-qt)
        elif self.wftype == WfTypes.gauss:
            return np.exp(-qt**2 / 4)
        elif self.wftype == WfTypes.exponential:
            return 1 / (1 + qt**2)**2
        elif self.wftype == WfTypes.diracdelta:
            return 1.0 + 0.0 * qt
        else:
            raise ValueError(
                f'Weightfunction type "{self.wftype}" not allowed.')

    def _get_K_G(self, gd):
        """Get the norms of the reciprocal lattice vectors."""
        if self._K_G is not None:
            return self._K_G
        assert gd.comm.size == 1
        k2_Q, _ = construct_reciprocal(gd)
        k2_Q[0, 0, 0] = 0
        self._K_G = k2_Q**(1 / 2)
        return self._K_G

    def lda_x2(self, spin, e, wn_g, nstar_g, v, my_alpha_indices):
        """Apply the WLDA-1 exchange functional.

        Calculate e[n*]n
        de/dn (n^*) * n^* * n/n^*
        
        """
        from gpaw.xc.lda import lda_constants
        assert spin in [0, 1]
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / nstar_g) ** (1 / 3.)
        ex = C1 / rs

        dexdrs = -ex / rs

        if spin == 0:
            e[:] += wn_g * ex
        else:
            e[:] += 0.5 * wn_g * ex
        v += ex
        t1 = rs * dexdrs / 3. # -dex/dn (n^*) * n^*
        ratio = self.regularize(wn_g, nstar_g, divergent=False)
        # ratio[np.isclose(nstar_g, 0.0, atol=1e-8)] = 0.0
        v += self.fold_with_derivative(-t1 * ratio,
                                       wn_g, my_alpha_indices, 0, wn_g[np.newaxis])


    def regularize(self, n1_sg, n2_sg, divergent):
        # fd = lambda x : 1 / (np.exp((x - self.mu) / (self.T) ) + 1)
        # def fermi(n_sg):
        #     inds = (n_sg - self.mu) / self.T > 10**2
        #     res_sg = n_sg * 0.0
        #     res_sg[inds] = 0.0
        #     res_sg[np.logical_not(inds)] = fd(n_sg[np.logical_not(inds)])
        #     assert (res_sg <= 1.0).all()
        #     assert (res_sg >= 0.0).all()
        #     return res_sg

        # return f_sg * fermi(f_sg)
        # mask = np.logical_and(np.isclose(n1_sg, 0.0, atol=1e-8),
        #                       np.isclose(n2_sg, 0.0, atol=1e-8))
        mask = np.isclose(n2_sg, 0.0, atol=1e-8)
        ratio = n1_sg / n2_sg
        if divergent:
            ratio[mask] = 1e8
        else:
            ratio[mask] = 1e-8
        return ratio

    def lda_x1(self, spin, e, wn_g, nstar_g, v, my_alpha_indices):
        """Apply the WLDA-2 exchange functional.

        Calculate e[n]n*

        \int ex(r) * dn*(r)/dn(r') dr
        \int e[n] dn^*/dn + de/dn n^*
        de/dn * n * n^*/n
        """
        from gpaw.xc.lda import lda_constants
        assert spin in [0, 1]
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / wn_g) ** (1 / 3.)
        ex = C1 / rs
        dexdrs = -ex / rs
        if spin == 0:
            e[:] += nstar_g * ex
        else:
            e[:] += 0.5 * nstar_g * ex
        v += self.fold_with_derivative(ex, wn_g, my_alpha_indices, 0, wn_g[np.newaxis])
        ratio = self.regularize(nstar_g, wn_g, divergent=True)
        # ratio *= fermi(ratio)
        # ratio[np.isclose(wn_g, 0.0, atol=1e-4)] = 0.0
        v -= rs * dexdrs / 3 * ratio

    def lda_x3(self, spin, e, wn_g, nstar_g, v, my_alpha_indices):
        """Apply the WLDA-3 exchange functional.

        Calculate e[n*]n*
        """
        from gpaw.xc.lda import lda_constants
        assert spin in [0, 1]
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / nstar_g) ** (1 / 3.)
        ex = C1 / rs
        dexdrs = -ex / rs
        if spin == 0:
            e[:] += nstar_g * ex
        else:
            e[:] += 0.5 * nstar_g * ex
        v[:] = ex - rs * dexdrs / 3.
        v[:] = self.fold_with_derivative(v, wn_g, my_alpha_indices, 0, wn_g[np.newaxis])

    def lda_c2(self, spin, e, wntotal_g, wn_sg, nstar_sg, v, zeta, my_alpha_indices):
        """Apply the WLDA-2 correlation functional.

        Calculate
            e^lda_c[n*]n
        and
            (de^lda_c[n*] / dn*) * (dn* / dn) + e^lda_c[n*]
        """
        assert spin in [0, 1]
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / nstar_sg.sum(axis=0)) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

        if spin == 0:
            e[:] += wntotal_g * ec
            v += ec
            ratio = self.regularize(wn_sg, nstar_sg, divergent=False)
            # ratio[np.isclose(nstar_sg, 0.0)] = 0.0
            v -= self.fold_with_derivative(rs * decdrs_0 / 3. * ratio[0],
                                           wntotal_g, my_alpha_indices, None, self.alphadensity_g)
        else:
            e1, decdrs_1 = G(rs ** 0.5,
                             0.015545, 0.20548, 14.1189,
                             6.1977, 3.3662, 0.62517)
            alpha, dalphadrs = G(rs ** 0.5,
                                 0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                                 0.49671)
            alpha *= -1.
            dalphadrs *= -1.
            zp = 1.0 + zeta
            zm = 1.0 - zeta
            xp = zp ** (1 / 3.)
            xm = zm ** (1 / 3.)
            f = CC1 * (zp * xp + zm * xm - 2.0)
            f1 = CC2 * (xp - xm)
            zeta3 = zeta * zeta * zeta
            zeta4 = zeta * zeta * zeta * zeta
            x = 1.0 - zeta4
            decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                      decdrs_1 * f * zeta4 +
                      dalphadrs * f * x * IF2)
            decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                        f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
            ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4
            e[:] += wntotal_g * ec

            ratio = self.regularize(wntotal_g, nstar_sg.sum(axis=0), divergent=False)
            # ratio[np.isclose(nstar_sg.sum(axis=0), 0.0)] = 0.0
            v[0] += ec
            v[0] -= self.fold_with_derivative_for_spinpol((rs * decdrs / 3.0
                                                           - (zeta - 1.0)
                                                           * decdzeta) * ratio,
                                                          wn_sg[0], self.alphadensity_g,
                                                          0, 0, my_alpha_indices)
            v[0] -= self.fold_with_derivative_for_spinpol((rs * decdrs / 3.0
                                                           - (zeta + 1.0)
                                                           * decdzeta) * ratio,
                                                          wn_sg[1], self.alphadensity_g,
                                                          1, 0, my_alpha_indices)

            v[1] += ec
            v[1] -= self.fold_with_derivative_for_spinpol((rs * decdrs / 3.0
                                                           - (zeta - 1.0)
                                                           * decdzeta) * ratio,
                                                          wn_sg[0], self.alphadensity_g,
                                                          0, 1, my_alpha_indices)
            v[1] -= self.fold_with_derivative_for_spinpol((rs * decdrs / 3.0
                                                           - (zeta + 1.0)
                                                           * decdzeta) * ratio,
                                                          wn_sg[1], self.alphadensity_g,
                                                          1, 1, my_alpha_indices)

    def lda_c1(self, spin, e, wntotal_g, nstar_sg, v, zeta, my_alpha_indices):
        """Apply the WLDA-1 correlation functional.

        Calculate
            e^lda_c[n] n*
        and
            (e^lda_c[n]) * (dn* / dn) + de^lda_c/dn n*
        """
        assert spin in [0, 1]
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / wntotal_g) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

        if spin == 0:
            e[:] += nstar_sg.sum(axis=0) * ec # nstar_sg.sum(axis=0) == nstar_sg[0]
            v += self.fold_with_derivative(ec, wntotal_g, my_alpha_indices, None, self.alphadensity_g)
            ratio = self.regularize(nstar_sg.sum(axis=0), wntotal_g, divergent=True)
            # ratio[np.isclose(wntotal_g, 0.0)] = 0.0
            v -= rs * decdrs_0 / 3. * ratio
        else:
            e1, decdrs_1 = G(rs ** 0.5,
                             0.015545, 0.20548, 14.1189,
                             6.1977, 3.3662, 0.62517)
            alpha, dalphadrs = G(rs ** 0.5,
                                 0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                                 0.49671)
            alpha *= -1.
            dalphadrs *= -1.
            zp = 1.0 + zeta
            zm = 1.0 - zeta
            xp = zp ** (1 / 3.)
            xm = zm ** (1 / 3.)
            f = CC1 * (zp * xp + zm * xm - 2.0)
            f1 = CC2 * (xp - xm)
            zeta3 = zeta * zeta * zeta
            zeta4 = zeta * zeta * zeta * zeta
            x = 1.0 - zeta4
            decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                      decdrs_1 * f * zeta4 +
                      dalphadrs * f * x * IF2)
            decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                        f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
            ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4
            e[:] += nstar_sg.sum(axis=0) * ec
            assert len(ec.shape) == 3

            ratio = self.regularize(nstar_sg.sum(axis=0), wntotal_g, divergent=True)
            # ratio[np.isclose(wntotal_g, 0.0)] = 0.0

            v[0] += self.fold_with_derivative_for_spinpol(ec, wntotal_g,
                                                          self.alphadensity_g,
                                                          0, 0, my_alpha_indices)
            v[0] -= (rs * decdrs / 3.0
                     - (zeta - 1.0)
                     * decdzeta) * ratio

            v[1] += self.fold_with_derivative_for_spinpol(ec, wntotal_g, self.alphadensity_g,
                                                          0, 0, my_alpha_indices)
            v[1] -= (rs * decdrs / 3.0
                     - (zeta + 1.0) * decdzeta) * ratio

    def lda_c3(self, spin, e, wntotal_g, wn_sg, nstar_sg, v, zeta, my_alpha_indices):
        """Apply the WLDA-3 correlation functional to n*.

        Calculate
            e^lda_c[n*]n* (used in the energy)
        and
            (v^lda_c[n*]) * (dn* / dn) (the potential from the correlation)
        """
        assert spin in [0, 1]
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / nstar_sg.sum(axis=0)) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

        if spin == 0:
            e[:] += nstar_sg[0] * ec
            v_place = np.zeros_like(v)
            v_place += ec - rs * decdrs_0 / 3.
            v += self.fold_with_derivative(v_place,
                                           wntotal_g, my_alpha_indices, None, self.alphadensity_g)
        else:
            e1, decdrs_1 = G(rs ** 0.5,
                             0.015545, 0.20548, 14.1189,
                             6.1977, 3.3662, 0.62517)
            alpha, dalphadrs = G(rs ** 0.5,
                                 0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                                 0.49671)
            alpha *= -1.
            dalphadrs *= -1.
            zp = 1.0 + zeta
            zm = 1.0 - zeta
            xp = zp ** (1 / 3.)
            xm = zm ** (1 / 3.)
            f = CC1 * (zp * xp + zm * xm - 2.0)
            f1 = CC2 * (xp - xm)
            zeta3 = zeta * zeta * zeta
            zeta4 = zeta * zeta * zeta * zeta
            x = 1.0 - zeta4
            decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                      decdrs_1 * f * zeta4 +
                      dalphadrs * f * x * IF2)
            decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                        f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
            ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4

            e[:] += nstar_sg.sum(axis=0) * ec

            bare_v0 = ec - (rs * decdrs / 3.0 - (zeta - 1.0) * decdzeta)
            bare_v1 = ec - (rs * decdrs / 3.0 - (zeta + 1.0) * decdzeta)
            v[0] += self.fold_with_derivative_for_spinpol(bare_v0,
                                                          wn_sg[0],
                                                          self.alphadensity_g,
                                                          0, 0,
                                                          my_alpha_indices)
            v[0] += self.fold_with_derivative_for_spinpol(bare_v1,
                                                          wn_sg[1],
                                                          self.alphadensity_g,
                                                          1, 0,
                                                          my_alpha_indices)


            # v[1] = ec - (rs * decdrs / 3.0 - (zeta + 1.0) * decdzeta)
            v[1] += self.fold_with_derivative_for_spinpol(bare_v0,
                                                          wn_sg[0],
                                                          self.alphadensity_g,
                                                          0, 1,
                                                          my_alpha_indices)
            v[1] += self.fold_with_derivative_for_spinpol(bare_v1,
                                                          wn_sg[1],
                                                          self.alphadensity_g,
                                                          1, 1,
                                                          my_alpha_indices)
            # v[1] = self.fold_with_derivative(v[1],
            #                                  wn_g, my_alpha_indices, None, self.alphadensity_g)

    def fold_with_derivative(self, f_g, n_g, my_alpha_indices, spin, density, mpisum=True):
        """Fold function f_g with the derivative of the weighted density.

        Calculate
            (f) * (dn*/dn) = int dr' f(r') dn*(r') / dn(r)
        """
        assert np.allclose(f_g, f_g.real)
        assert np.allclose(n_g, n_g.real)
        assert len(my_alpha_indices) == 0 or type(my_alpha_indices[0]) == int

        res_g = np.zeros_like(f_g)

        for ia in my_alpha_indices:
            # ind_g = self.get_indicator_g(n_g, ia)
            # dind_g = self.get_dindicator_g(n_g, ia)
            ind_g = self.ind_asg(ia, spin, density)
            dind_g = self.dind_asg(ia, spin, density)

            fac_g = ind_g + dind_g * n_g
            int_G = self.fftn(f_g)
            w_G = self.get_weight_function(ia, self.gd, self.alphas)
            r_g = self.ifftn(w_G * int_G)
            res_g += r_g.real * fac_g
            # assert np.allclose(res_g, res_g.real)

        res_g = np.ascontiguousarray(res_g)

        if mpisum:
            mpi.world.sum(res_g)

        assert res_g.shape == f_g.shape
        assert res_g.shape == n_g.shape

        return res_g

    def fold_with_derivative_for_spinpol(self, f_g, n1_g, ntotal_g, spin1, spin2, my_alpha_indices):
        """Calculate f_g convoluted with (delta n^star_spin1) / (delta n_spin2)."""

        assert np.allclose(f_g, f_g.real)
        assert np.allclose(n1_g, n1_g.real)
        assert len(my_alpha_indices) == 0 or type(my_alpha_indices[0]) == int

        res_g = np.zeros_like(f_g)

        for ia in my_alpha_indices:
            ind_g = self.ind_asg(ia, None, ntotal_g)
            dind_g = self.dind_asg(ia, None, ntotal_g)

            if spin1 == spin2:
                fac_g = ind_g + dind_g * n1_g
            else:
                fac_g = dind_g * n1_g

            int_G = self.fftn(f_g)
            w_G = self.get_weight_function(ia, self.gd, self.alphas)
            r_g = self.ifftn(w_G * int_G)
            res_g += r_g.real * fac_g
            # assert np.allclose(res_g, res_g.real)

        res_g = np.ascontiguousarray(res_g)
        mpi.world.sum(res_g)
        assert res_g.shape == f_g.shape
        assert res_g.shape == n1_g.shape

        return res_g

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        return self.lda_xc.calculate_paw_correction(setup, D_sp, dEdD_sp=dEdD_sp, a=a)

    def density_correction(self, gd, n_sg):
        """Apply density correction.

        This approximates the AE density around the atoms.
        The true density is replaced by a smooth function
        close to the atoms. The distance within which the
        AE density is replaced is determined by self.rcut_factor.

        If self.rcut_factor is None, the pseudo density is returned.
        """
        if self.density_type == DensityTypes.AE:
            res_sg, resgd = self.density.get_all_electron_density(atoms=self.atoms, gridrefinement=2)
            # wn_sg = self.density.redistributor.aux_gd.collect(res, broadcast=True)
            wn_sg = resgd.collect(res_sg, broadcast=True)
            gd1 = resgd.new_descriptor(comm=mpi.serial_comm)
            return wn_sg, gd1
        elif self.density_type == DensityTypes.smoothAE:
            # TODO Think about how to handle this.
            # For other two modes we can collect after correcting
            # but for this mode we have to collect before.
            raise NotImplementedError
            from gpaw.xc.WDAUtils import correct_density
            return correct_density(n_sg, gd, self.wfs.setups,
                                   self.wfs.spos_ac, self.rcut_factor)
        else:
            wn_sg = gd.collect(n_sg, broadcast=True)
            gd1 = gd.new_descriptor(comm=mpi.serial_comm)
            return wn_sg, gd1

    def hartree_correction(self, gd, e_g, v_sg,
                           wn_sg, nstar_sg, my_alpha_indices):
        """Apply the correction to the Hartree energy.

        Calculate
            Delta E = -0.5 int dr' (n-n*)(r)(n-n*)(r')/|r-r'|

        This corrects the Hartree energy so it is effectively
        calculated with n* instead of n.

        Also calculate the resulting correction to the
        potential:
            Delta v = dDeltaE/dn
        With
            V := int dr (n-n*)(r')/|r-r'|
        Delta v is equal to
            Delta v(r) = -2 * (V(r)
                    -    int dr' dn*(r')/dn(r) V(r')
        """
        if not self.use_hartree_correction:
            return

        if len(wn_sg) == 2 and not self.hxc:
            # Undo modification of density performed by wlda_x
            wn_sg *= 0.5
            nsum_sg = np.array([wn_sg.sum(axis=0)])
            v1_sg = np.zeros_like(nsum_sg)

            nstar_sg, my_alpha_indices = self.get_weighted_density(nsum_sg)

            self.do_hartree_corr(gd, nsum_sg.sum(axis=0), nstar_sg.sum(axis=0),
                                 e_g, v1_sg, [0], my_alpha_indices)

            mpi.world.sum(v1_sg)

            v_sg[0] += v1_sg[0]
            v_sg[1] += v1_sg[0]

        elif len(wn_sg) == 2 and self.hxc:
            nstar_sg, my_alpha_indices = self.get_weighted_density(wn_sg)

            # Density already multipled by 2 so don't need to do it
            # here
            e1_g = np.zeros_like(e_g)
            v1_sg = np.zeros_like(v_sg)
            self.do_hartree_corr(gd, wn_sg[0], nstar_sg[0], e1_g, v1_sg, [
                                 0], my_alpha_indices)
            self.do_hartree_corr(gd, wn_sg[1], nstar_sg[1], e1_g, v1_sg, [
                                 1], my_alpha_indices)


            mpi.world.sum(v1_sg)

            e_g += 0.5 * e1_g
            v_sg += v1_sg
        else:
            v1_sg = np.zeros_like(v_sg)
            self.do_hartree_corr(gd, wn_sg.sum(axis=0), nstar_sg.sum(axis=0),
                                 e_g, v1_sg, [0], my_alpha_indices)
            mpi.world.sum(v1_sg)
            v_sg += v1_sg

    def do_hartree_corr(self, gd, n_g, nstar_g,
                        e_g, v_sg, spins, my_alpha_indices):
        # \int dr(n^* - n) \int dr' (n^* - n)' / |r-r'|
        # V(r) = \int dr' (n^* - n)' / |r-r'| <- solve poisson for n^* - n
        # nabla^2 V(r) = -4pi (n^* - n)
        # q^2 V(q) = -4pi (n^*  n)(q)
        K_G = self._get_K_G(gd).copy()
        K_G[0, 0, 0] = 1.0

        dn_g = n_g - nstar_g

        V_G = self.fftn(dn_g) / K_G**2 * 4 * np.pi * 0.5
        V_g = self.ifftn(V_G).real

        e_g[:] -= (dn_g * V_g)

        for s in spins:
            im = 2 * (- self.fold_with_derivative(V_g,
                                                  n_g,
                                                  my_alpha_indices,
                                                  s, self.wn_sg,
                                                  mpisum=False))
            if mpi.rank == 0:
                v_sg[s, :] -= (2 * V_g + im)
            else:
                v_sg[s, :] -= im

    def do_radial_lda(self, n_sg):
        eldax_g = np.zeros_like(n_sg[0])
        eldac_g = np.zeros_like(n_sg[0])
        vldax_sg = np.zeros_like(n_sg)
        vldac_sg = np.zeros_like(n_sg)
            
        self.calculate_lda(n_sg, eldax_g, eldac_g, vldax_sg, vldac_sg)

        self.elda_g = eldax_g + eldac_g
        self.vlda_sg = vldax_sg + vldac_sg

        if self.exchange_only:
            self.e_corr_g = eldax_g
            self.v_corr_sg = vldax_sg
        else:
            self.e_corr_g = eldax_g + eldac_g
            self.v_corr_sg = vldax_sg + vldac_sg
        
    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        """Calculate the XC energy and potential for an atomic system.

        Is called when using GPAW's atom module with WLDA.

        We assume this function is never called in parallel.

        Modifies:
            v_sg to contain the XC potential

        Returns:
            The /total/ XC energy
        """
        self.do_radial_lda(n_sg)

        if e_g is None:
            e_g = rgd.empty()
        self.rgd = rgd
        n_sg[n_sg < 1e-20] = 1e-40

        self.setup_radial_indicators(n_sg, self.nindicators)

        nstar_sg, v1, v2, v3 = self.radial_x(n_sg, e_g, v_sg)

        if not self.exchange_only:
            nstar_sg = self.radial_c(n_sg, e_g, v_sg, nstar_sg)

        eHa_g = np.zeros_like(e_g)
        vHa_sg = np.zeros_like(v_sg)
        self.radial_hartree_correction(rgd, n_sg, nstar_sg, vHa_sg, eHa_g)

        if self.mode == Modes.fWLDA:
            e_g[:] = self.elda_g + self.lambd * (e_g - self.e_corr_g) + self.lambd * eHa_g
            v_sg[:] = self.vlda_sg + self.lambd * self.sign_regularization(v_sg - self.v_corr_sg) + self.lambd * vHa_sg
        elif self.mode == Modes.rWLDA:
            e_g[:] = self.elda_g + self.lambd * (e_g - self.e_corr_g) + eHa_g
            v_sg[:] = self.vlda_sg + self.lambd * self.sign_regularization(v_sg - self.v_corr_sg) + vHa_sg
        else:
            assert self.mode == Modes.WLDA
            e_g[:] = self.elda_g + (e_g - self.e_corr_g) + eHa_g
            v_sg[:] = self.vlda_sg + self.sign_regularization(v_sg - self.v_corr_sg) + vHa_sg

        E = rgd.integrate(e_g)

        if self.save:
            np.save(f"deltaexc_{self.saveindex}.npy", e_g - self.e_corr_g)
            np.save(f"deltavxc_{self.saveindex}.npy", v_sg - self.v_corr_sg)
            np.save(f"deltaeh_{self.saveindex}.npy", eHa_g)
            np.save(f"deltavh_{self.saveindex}.npy", vHa_sg)
            np.save(f"elda_{self.saveindex}.npy", self.elda_g)
            np.save(f"vlda_{self.saveindex}.npy", self.vlda_sg)
            np.save("r_g.npy", rgd.r_g)
            self.saveindex += 1

        return E

    def radial_x(self, n_sg, e_g, v_sg):
        """Calculate WLDA exchange energy.

        Returns nstar_sg if the calculation is spin-paired
        so it can be reused.
        """
        if len(n_sg) == 2:
            n_sg *= 2
        spin = len(n_sg) - 1

        self.setup_radial_indicators(n_sg, self.nindicators)

        nstar_sg = self.radial_weighted_density(n_sg)

        e1_g = np.zeros_like(e_g)
        e2_g = np.zeros_like(e_g)
        e3_g = np.zeros_like(e_g)

        v1_sg = np.zeros_like(v_sg)
        v2_sg = np.zeros_like(v_sg)
        v3_sg = np.zeros_like(v_sg)

        if spin == 0:
            self.radial_x1(spin, 0, e1_g, n_sg[0], nstar_sg[0], v1_sg[0])
            self.radial_x2(spin, 0, e2_g, n_sg[0], nstar_sg[0], v2_sg[0])
            self.radial_x3(spin, 0, e3_g, n_sg[0], nstar_sg[0], v3_sg[0])

            e_g[:] = e1_g + e2_g - e3_g
            v_sg[:] = v1_sg + v2_sg - v3_sg

            return nstar_sg, v1_sg, v2_sg, v3_sg
        else:
            self.radial_x1(spin, 0, e1_g, n_sg[0], nstar_sg[0], v1_sg[0])
            self.radial_x2(spin, 0, e2_g, n_sg[0], nstar_sg[0], v2_sg[0])
            self.radial_x3(spin, 0, e3_g, n_sg[0], nstar_sg[0], v3_sg[0])

            self.radial_x1(spin, 1, e1_g, n_sg[1], nstar_sg[1], v1_sg[1])
            self.radial_x2(spin, 1, e2_g, n_sg[1], nstar_sg[1], v2_sg[1])
            self.radial_x3(spin, 1, e3_g, n_sg[1], nstar_sg[1], v3_sg[1])

            e_g[:] = e1_g + e2_g - e3_g
            v_sg[:] = v1_sg + v2_sg - v3_sg

            return None, v1_sg, v2_sg, v3_sg

    def radial_c(self, n_sg, e_g, v_sg, nstar_sg):
        """Calculate WLDA correlation energy."""
        spin = len(n_sg) - 1

        e1_g = np.zeros_like(e_g)
        e2_g = np.zeros_like(e_g)
        e3_g = np.zeros_like(e_g)

        v1_sg = np.zeros_like(v_sg)
        v2_sg = np.zeros_like(v_sg)
        v3_sg = np.zeros_like(v_sg)

        if spin == 0:
            zeta = 0
            self.radial_c1(spin, e1_g, n_sg[0], n_sg, nstar_sg, v1_sg, zeta)
            self.radial_c2(spin, e2_g, n_sg[0], n_sg, nstar_sg, v2_sg, zeta)
            self.radial_c3(spin, e3_g, n_sg[0], n_sg, nstar_sg, v3_sg, zeta)

            e_g[:] += e1_g + e2_g - e3_g
            v_sg[:] += v1_sg + v2_sg - v3_sg

            return nstar_sg
        else:
            n_g = n_sg.sum(axis=0) * 0.5
            self.setup_radial_indicators(n_sg * 0.5, self.nindicators)
            # nstar_sg = self.radial_weighted_density(n_g)
            nstar_sg = self.radial_weighted_density_spinpol(n_sg * 0.5)
            zeta_g = (n_sg[0] - n_sg[1]) / (n_sg[0] + n_sg[1])
            zeta_g[np.isclose((n_sg[0] + n_sg[1]), 0.0)] = 0.0

            zetastar_g = (nstar_sg[0] - nstar_sg[1]) / (nstar_sg[0] + nstar_sg[1])
            zetastar_g[np.isclose((nstar_sg[0] + nstar_sg[1]), 0.0)] = 0.0

            self.radial_c1(spin, e1_g, n_g, n_sg * 0.5, nstar_sg, v1_sg, zeta_g)
            self.radial_c2(spin, e2_g, n_g, n_sg * 0.5, nstar_sg, v2_sg, zetastar_g)
            self.radial_c3(spin, e3_g, n_g, n_sg * 0.5, nstar_sg, v3_sg, zetastar_g)

            e_g[:] += e1_g + e2_g - e3_g
            v_sg[:] += v1_sg + v2_sg - v3_sg

            return None
    
    def setup_radial_indicators(self, n_sg, nalphas):
        """Set up the indicator functions for an atomic system.

        We first define a grid of indicator-values/anchors.

        Then we evaluate the indicator functions on the
        density.
        """
        if n_sg.ndim == 1:
            spin = 1
            n_sg = np.array([n_sg])
        else:
            spin = n_sg.shape[0]

        alphas = define_alphas(n_sg, nalphas)
        self.C_aip = make_spline_coefficients(alphas)

        N = 2**14
        r_x = np.linspace(0, self.rgd.r_g[-1], N)

        ns = len(n_sg)
        n_sx = np.zeros((ns, len(r_x)))
        for s in range(ns):
            n_sx[s, :] = self.rgd.interpolate(n_sg[s], r_x)

        i_asg = np.zeros((nalphas,) + n_sg.shape)
        di_asg = np.zeros((nalphas,) + n_sg.shape)
        i_asx = np.zeros((nalphas,) + n_sx.shape)
        di_asx = np.zeros((nalphas,) + n_sx.shape)
        
        ti_ag = np.zeros((nalphas,) + n_sg.shape[1:])
        dti_ag = np.zeros((nalphas,) + n_sg.shape[1:])
        ti_ax = np.zeros((nalphas,) + n_sx.shape[1:])
        dti_ax = np.zeros((nalphas,) + n_sx.shape[1:])


        for ia, alpha in enumerate(alphas):
            _i, _di = define_indicator(ia, alphas, self.C_aip)

            for s in range(spin):
                i_asg[ia, s, :] = _i(n_sg[s])
                di_asg[ia, s, :] = _di(n_sg[s])
                i_asx[ia, s, :] = _i(n_sx[s])
                di_asx[ia, s, :] = _di(n_sx[s])
                
            ti_ag[ia, :] = _i(n_sg.sum(axis=0))
            dti_ag[ia, :] = _di(n_sg.sum(axis=0))

            ti_ax[ia, :] = _i(n_sx.sum(axis=0))
            dti_ax[ia, :] = _di(n_sx.sum(axis=0))

        # assert np.allclose(i_asg.sum(axis=0), 1.0)
        # assert np.allclose(di_asg.sum(axis=0), 0.0)
        # assert np.allclose(i_asx.sum(axis=0), 1.0)
        # assert np.allclose(di_asx.sum(axis=0), 0.0)
        # assert np.allclose(ti_ag.sum(axis=0), 1.0)
        # assert np.allclose(dti_ag.sum(axis=0), 0.0)

        self.alphas = alphas
        self.i_asg = i_asg
        self.di_asx = di_asx
        self.di_asg = di_asg
        self.i_asx = i_asx
        self.ti_ag = ti_ag
        self.dti_ag = dti_ag
        self.ti_ax = ti_ax
        self.dti_ax = dti_ax

    def radial_weighted_density(self, n_sg):
        """Calculate the weighted density for an atomic system.

        n*(k) = sum_a phi(k, a)(i_a[n]n)(k)

        where f(k) is found using spherical Bessel transforms.
        This is just the standard Fourier transform
        specialized to a spherically symmetric function.

        The spherical Bessel transform is implemented
        in gpaw.atom.radialgd.fsbt, where we need the
        l = 0 version only.

        First we need to define a grid with regular spacing.
        Then we need to transfer the density to the regular grid.
        Then we can do the Fourier transform via fsbt.
        We multiply the weight function and the FT'ed
        indicator times density and do inverse FT via fsbt.
        Finally, we transfer the resulting array back to the
        radial grid (with uneven spacing probably).
        """
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS
        rgd = self.rgd
        N = 2**14
        r_x = np.linspace(0, rgd.r_g[-1], N)
        G_k = np.linspace(0, np.pi / r_x[1], N // 2 + 1)

        na, ns = self.i_asx.shape[:2]

        nstar_sg = np.zeros_like(n_sg)
        for ia in range(na):
            for s in range(ns):
                n_x = rgd.interpolate(n_sg[s], r_x)
                n_x[n_x < 1e-20] = 1e-40

                i_x = self.i_asx[ia, s, :]
                in_k = self.radial_fft(r_x, n_x * i_x)

                phi_k = self.radial_weight_function(self.alphas[ia], G_k)

                res_x = self.radial_ifft(r_x, in_k * phi_k)
                nstar_sg[s, :] += IUS(r_x, res_x)(rgd.r_g)

        nstar_sg[nstar_sg < 1e-20] = 1e-40
        return nstar_sg

    def radial_weighted_density_spinpol(self, n_sg):
        """Calculate the weighted density for an atomic system.

        n*(k) = sum_a phi(k, a)(i_a[n]n)(k)

        where f(k) is found using spherical Bessel transforms.
        This is just the standard Fourier transform
        specialized to a spherically symmetric function.

        The spherical Bessel transform is implemented
        in gpaw.atom.radialgd.fsbt, where we need the
        l = 0 version only.

        First we need to define a grid with regular spacing.
        Then we need to transfer the density to the regular grid.
        Then we can do the Fourier transform via fsbt.
        We multiply the weight function and the FT'ed
        indicator times density and do inverse FT via fsbt.
        Finally, we transfer the resulting array back to the
        radial grid (with uneven spacing probably).
        """
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS
        rgd = self.rgd
        N = 2**14
        r_x = np.linspace(0, rgd.r_g[-1], N)
        G_k = np.linspace(0, np.pi / r_x[1], N // 2 + 1)

        na, ns = self.i_asx.shape[:2]
        assert ns == 2

        nstar_sg = np.zeros_like(n_sg)
        for ia in range(na):
            for s in range(ns):
                n_x = rgd.interpolate(n_sg[s], r_x)
                n_x[n_x < 1e-20] = 1e-40

                i_x = self.ti_ax[ia, :]
                in_k = self.radial_fft(r_x, n_x * i_x)

                phi_k = self.radial_weight_function(self.alphas[ia], G_k)

                res_x = self.radial_ifft(r_x, in_k * phi_k)
                nstar_sg[s, :] += IUS(r_x, res_x)(rgd.r_g)

        nstar_sg[nstar_sg < 1e-20] = 1e-40
        return nstar_sg

    def radial_fft(self, r_x, f_x):
        """Calculate the FT for a spherically symmetric function.

        Calculates
            int d^3r e^(-ikr)f(||r||)

        By a simple calculation this can be shown to be
        equal to the spherical Bessel transform for l = 0
        times 4pi.
        """
        # Verify that we have a uniform grid
        assert np.allclose(r_x[0], 0.0)
        assert np.allclose(r_x[1], r_x[2] - r_x[1])
        from gpaw.atom.radialgd import fsbt
        N = len(r_x)
        assert N % 2 == 0
        G_k = np.linspace(0, np.pi / r_x[1], N // 2 + 1)
        return fsbt(0, f_x, r_x, G_k) * 4 * np.pi

    def radial_ifft(self, r_x, f_k):
        from gpaw.atom.radialgd import fsbt
        N = len(r_x)
        assert N % 2 == 0
        G_k = np.linspace(0, np.pi / r_x[1], N // 2 + 1)
        res = fsbt(0, f_k, G_k, r_x[:N // 2 + 1]) / (2 * np.pi**2)
        res_x = np.zeros_like(r_x)
        res_x[:len(res)] = res
        return res_x

    def radial_weight_function(self, alpha, G_k):
        """Evaluates and returns phi(q, alpha)."""
        qt = (G_k / (self.c1 * abs(alpha)**(1 / 3)))

        wf = self.wftype
        if wf == WfTypes.lorentz:
            return np.exp(-qt)
        elif wf == WfTypes.gauss:
            return np.exp(-qt**2 / 4)
        elif wf == WfTypes.exponential:
            return 1 / (1 + qt**2)**2
        else:
            raise ValueError(f'Weightfunction type "{wf}" not allowed.')

    def radial_x2(self, spin, spinindex, e_g, n_g, nstar_g, v_g):
        """Calculate e_x[n*]n and potential."""
        from gpaw.xc.lda import lda_constants
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / nstar_g)**(1. / 3.)
        ex = C1 / rs

        dexdrs = -ex / rs

        if spin == 0:
            e_g[:] += n_g * ex
        else:
            e_g[:] += 0.5 * n_g * ex

        v_g += ex
        t1 = rs * dexdrs / 3.0
        ratio = self.regularize(n_g, nstar_g, divergent=False)
        # ratio[np.isclose(nstar_g, 0.0)] = 0.0
        v_g += self.radial_derivative_fold(-t1 * ratio, n_g, spinindex)

    def radial_c2(self, spin, e_g, ntotal_g, n_sg, nstar_sg, v_sg, zeta):
        """Calculate e_c[n*]n and potential."""
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / nstar_sg.sum(axis=0)) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

        if spin == 0:
            e_g[:] += ec * ntotal_g

            v_sg[0, :] += ec
            t1 = rs * decdrs_0 / 3.0
            ratio = self.regularize(ntotal_g, nstar_sg.sum(axis=0), divergent=False)
            # ratio[np.isclose(nstar_sg.sum(axis=0), 0.0)] = 0.0
            v_sg[0, :] += self.radial_derivative_fold3(-t1 * ratio, ntotal_g, 
                                                   0, 0, self.ti_ag, self.dti_ag)
        else:
            e1, decdrs_1 = G(rs**0.5,
                             0.015545, 0.20548, 14.1189,
                             6.1977, 3.3662, 0.62517)
            alpha, dalphadrs = G(rs ** 0.5,
                                 0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                                 0.49671)

            alpha *= -1
            dalphadrs *= -1
            zp = 1.0 + zeta
            zm = 1.0 - zeta
            xp = zp**(1. / 3.)
            xm = zm ** (1 / 3.)
            f = CC1 * (zp * xp + zm * xm - 2.0)
            f1 = CC2 * (xp - xm)
            zeta3 = zeta * zeta * zeta
            zeta4 = zeta * zeta * zeta * zeta
            x = 1.0 - zeta4
            decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                      decdrs_1 * f * zeta4 +
                      dalphadrs * f * x * IF2)
            decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                        f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
            ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4

            e_g[:] += ntotal_g * ec
            t1 = rs * decdrs / 3.0 - (zeta - 1.0) * decdzeta
            t2 = rs * decdrs / 3.0 - (zeta + 1.0) * decdzeta
            ratio = self.regularize(ntotal_g, nstar_sg.sum(axis=0), divergent=False)
            # ratio[np.isclose(nstar_sg.sum(axis=0), 0.0)] = 0.0

            
            v_sg[0] += ec 
            v_sg[0] -= self.radial_derivative_fold3(t1 * ratio, n_sg[0], 0, 0,
                                                    self.ti_ag, self.dti_ag)
            v_sg[0] -= self.radial_derivative_fold3(t2 * ratio, n_sg[1], 1, 0,
                                                    self.ti_ag, self.dti_ag)

            v_sg[1] += ec
            v_sg[1] -= self.radial_derivative_fold3(t1 * ratio, n_sg[0], 0, 1,
                                                    self.ti_ag, self.dti_ag)
            v_sg[1] -= self.radial_derivative_fold3(t2 * ratio, n_sg[1], 1, 1,
                                                    self.ti_ag, self.dti_ag)
            # v_sg[1] += ec - \
            #     self.radial_derivative_fold(
            #         t1 * ratio, n_g, 0) - (zeta + 1.0) * decdzeta

    def radial_x1(self, spin, spinindex, e_g, n_g, nstar_g, v_g):
        """Calculate e_x[n]n* and potential.

        It is necessary to regularize the potential as
        below to avoid the full potential increasing
        unphysically.
        """
        from gpaw.xc.lda import lda_constants
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / n_g)**(1. / 3.)
        ex = C1 / rs

        dexdrs = -ex / rs

        if spin == 0:
            e_g[:] += ex * nstar_g
        else:
            e_g[:] += 0.5 * nstar_g * ex

        v_g[:] += self.radial_derivative_fold(ex, n_g, spinindex)
        ratio = self.regularize(nstar_g, n_g, divergent=True)
        # ratio[np.isclose(n_g, 0.0)] = 0.0
        v_g[:] += -rs * dexdrs / 3.0 * ratio * \
            np.logical_not(np.isclose(rs * dexdrs / 3.0, 0.0))

    def radial_c1(self, spin, e_g, ntotal_g, n_sg, nstar_sg, v_sg, zeta):
        """Calculate e_c[n]n* and potential."""
        from gpaw.xc.lda import lda_constants, G

        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / ntotal_g) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

        if spin == 0:
            e_g[:] += ec * nstar_sg.sum(axis=0)
            v_sg[0, :] += self.radial_derivative_fold3(ec, ntotal_g, 0, 0, 
                                                       self.ti_ag, self.dti_ag)
            ratio = self.regularize(nstar_sg.sum(axis=0), ntotal_g, divergent=True)
            # ratio[np.isclose(ntotal_g, 0.0)] = 0.0
            v_sg[0, :] += -rs * decdrs_0 / 3.0 * ratio * \
                          np.logical_not(np.isclose(rs * decdrs_0 / 3.0, 0.0))
        else:
            e1, decdrs_1 = G(rs ** 0.5,
                             0.015545, 0.20548, 14.1189,
                             6.1977, 3.3662, 0.62517)
            alpha, dalphadrs = G(rs ** 0.5,
                                 0.016887, 0.11125, 10.357,
                                 3.6231, 0.88026, 0.49671)
            alpha *= -1.
            dalphadrs *= -1.
            zp = 1.0 + zeta
            zm = 1.0 - zeta
            xp = zp ** (1 / 3.)
            xm = zm ** (1 / 3.)
            f = CC1 * (zp * xp + zm * xm - 2.0)
            f1 = CC2 * (xp - xm)
            zeta3 = zeta * zeta * zeta
            zeta4 = zeta * zeta * zeta * zeta
            x = 1.0 - zeta4
            decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                      decdrs_1 * f * zeta4 +
                      dalphadrs * f * x * IF2)
            decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                        f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
            ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4
            e_g[:] += nstar_sg.sum(axis=0) * ec

            ratio = self.regularize(nstar_sg.sum(axis=0), ntotal_g, divergent=True)
            # ratio[np.isclose(ntotal_g, 0.0)] = 0.0

            v_sg[0] += self.radial_derivative_fold3(ec, ntotal_g, 0, 0, self.ti_ag, self.dti_ag) 
            v_sg[0] -= (rs * decdrs / 3.0 - (zeta - 1.0) * decdzeta) * ratio

            v_sg[1] += self.radial_derivative_fold3(ec, ntotal_g, 0, 0, self.ti_ag, self.dti_ag)
            v_sg[1] -= (rs * decdrs / 3.0 - (zeta + 1.0) * decdzeta) * ratio

    def radial_x3(self, spin, spinindex, e_g, n_g, nstar_g, v_g):
        """Calculate e_x[n*]n* and potential."""
        from gpaw.xc.lda import lda_constants
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / nstar_g)**(1. / 3.)
        ex = C1 / rs

        dexdrs = -ex / rs

        if spin == 0:
            e_g[:] += ex * nstar_g
        else:
            e_g[:] += 0.5 * nstar_g * ex

        v_g[:] += (ex - rs * dexdrs / 3.0)
        v_g[:] = self.radial_derivative_fold(v_g, n_g, spinindex)

    def radial_c3(self, spin, e_g, ntotal_g, n_sg, nstar_sg, v_sg, zeta):
        """Calculate e_c[n*]n* and potential."""
        from gpaw.xc.lda import lda_constants, G
        C0I, C1, CC1, CC2, IF2 = lda_constants()

        rs = (C0I / nstar_sg.sum(axis=0)) ** (1 / 3.)
        ec, decdrs_0 = G(rs ** 0.5,
                         0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

        if spin == 0:
            e_g[:] += ec * nstar_sg.sum(axis=0)

            v_sg[0, :] += self.radial_derivative_fold3(ec - rs * decdrs_0 / 3.0, ntotal_g,
                                                   0, 0, self.ti_ag, self.dti_ag)
        else:
            e1, decdrs_1 = G(rs ** 0.5,
                             0.015545, 0.20548, 14.1189,
                             6.1977, 3.3662, 0.62517)
            alpha, dalphadrs = G(rs ** 0.5,
                                 0.016887, 0.11125, 10.357,
                                 3.6231, 0.88026, 0.49671)
            alpha *= -1.
            dalphadrs *= -1.
            zp = 1.0 + zeta
            zm = 1.0 - zeta
            xp = zp ** (1 / 3.)
            xm = zm ** (1 / 3.)
            f = CC1 * (zp * xp + zm * xm - 2.0)
            f1 = CC2 * (xp - xm)
            zeta3 = zeta * zeta * zeta
            zeta4 = zeta * zeta * zeta * zeta
            x = 1.0 - zeta4
            decdrs = (decdrs_0 * (1.0 - f * zeta4) +
                      decdrs_1 * f * zeta4 +
                      dalphadrs * f * x * IF2)
            decdzeta = (4.0 * zeta3 * f * (e1 - ec - alpha * IF2) +
                        f1 * (zeta4 * e1 - zeta4 * ec + x * alpha * IF2))
            ec += alpha * IF2 * f * x + (e1 - ec) * f * zeta4
            e_g[:] += nstar_sg.sum(axis=0) * ec

            bare_v0 = ec - (rs * decdrs / 3.0 - (zeta - 1.0) * decdzeta)
            bare_v1 = ec - (rs * decdrs / 3.0 - (zeta + 1.0) * decdzeta)

            v_sg[0] += self.radial_derivative_fold3(bare_v0, n_sg[0],
                                                    0, 0, self.ti_ag, self.dti_ag)
            v_sg[0] += self.radial_derivative_fold3(bare_v1, n_sg[1],
                                                    1, 0, self.ti_ag, self.dti_ag)

            v_sg[1] += self.radial_derivative_fold3(bare_v0, n_sg[0],
                                                    0, 1, self.ti_ag, self.dti_ag)
            v_sg[1] += self.radial_derivative_fold3(bare_v1, n_sg[1],
                                                    1, 1, self.ti_ag, self.dti_ag)

            # t = self.radial_derivative_fold(ec - rs * decdrs / 3.0, n_g, 0)
            # v_g[0] += t - (zeta - 1.0) * decdzeta
            # v_g[1] += t - (zeta + 1.0) * decdzeta

    def radial_derivative_fold(self, f_g, n_g, spin):
        """Calculate folding expression that appears in
        the derivative of the energy.

        Implements
            F(r) = int f(r') dn*(r') / dn(r)

        Calculates
            sum_a [(f_g FOLD phi_a) * (i_a + di_a * n)]
        """
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS
        rgd = self.rgd
        N = 2**14
        r_x = np.linspace(0, rgd.r_g[-1], N)
        G_k = np.linspace(0, np.pi / r_x[1], N // 2 + 1)

        res_g = np.zeros_like(f_g)
        for ia, a in enumerate(self.alphas):
            f_x = rgd.interpolate(f_g, r_x)
            f_k = self.radial_fft(r_x, f_x)
            phi_k = self.radial_weight_function(self.alphas[ia], G_k)
            res_g += (IUS(r_x,
                          self.radial_ifft(r_x, f_k * phi_k))(rgd.r_g)
                      * (self.i_asg[ia, spin, :]
                         + self.di_asg[ia, spin, :] * n_g))

        return res_g

    def radial_derivative_fold2(self, f_g, n_g, i_ag, di_ag):
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS
        rgd = self.rgd
        N = 2**14
        r_x = np.linspace(0, rgd.r_g[-1], N)
        G_k = np.linspace(0, np.pi / r_x[1], N // 2 + 1)

        res_g = np.zeros_like(f_g)
        for ia, a in enumerate(self.alphas):
            f_x = rgd.interpolate(f_g, r_x)
            f_k = self.radial_fft(r_x, f_x)
            phi_k = self.radial_weight_function(self.alphas[ia], G_k)
            res_g += (IUS(r_x,
                          self.radial_ifft(r_x, f_k * phi_k))(rgd.r_g)
                      * (i_ag[ia, :]
                         + di_ag[ia, :] * n_g))

        return res_g

    def radial_derivative_fold3(self, f_g, n_g, spin1, spin2, i_ag, di_ag):
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS
        rgd = self.rgd
        N = 2**14
        r_x = np.linspace(0, rgd.r_g[-1], N)
        G_k = np.linspace(0, np.pi / r_x[1], N // 2 + 1)

        res_g = np.zeros_like(f_g)
        for ia, a in enumerate(self.alphas):
            f_x = rgd.interpolate(f_g, r_x)
            f_k = self.radial_fft(r_x, f_x)
            phi_k = self.radial_weight_function(self.alphas[ia], G_k)

            if spin1 == spin2:
                fac_g = i_ag[ia, :] + di_ag[ia, :] * n_g
            else:
                fac_g = di_ag[ia, :] * n_g

            res_g += IUS(r_x, self.radial_ifft(r_x, f_k * phi_k))(rgd.r_g) * fac_g

        return res_g

    def radial_hartree_correction(self, rgd, n_sg, nstar_sg, v_sg, e_g):
        """Calculate energy correction -(n-n*)int (n-n*)(r')/(r-r').

        Also potential.
        """
        if not self.use_hartree_correction:
            return

        if len(n_sg) == 2 and not self.hxc:
            # Undo modification by radial_x
            n_sg *= 0.5
            nsum_sg = np.array([n_sg.sum(axis=0)])
            v1_sg = np.zeros_like(nsum_sg)

            self.setup_radial_indicators(nsum_sg, self.nindicators)
            nstar_sg = self.radial_weighted_density(nsum_sg)

            self.do_radial_hartree_corr(rgd, nsum_sg.sum(axis=0),
                                        nstar_sg.sum(axis=0),
                                        e_g, v1_sg, [0])
            v_sg[0] += v1_sg[0]
            v_sg[1] += v1_sg[0]

        elif len(n_sg) == 2 and self.hxc:
            self.setup_radial_indicators(n_sg, self.nindicators)
            nstar_sg = self.radial_weighted_density(n_sg)

            e1_g = np.zeros_like(e_g)
            v1_sg = np.zeros_like(v_sg)

            self.do_radial_hartree_corr(
                rgd, n_sg[0], nstar_sg[0], e1_g, v1_sg, [0])
            self.do_radial_hartree_corr(
                rgd, n_sg[1], nstar_sg[1], e1_g, v1_sg, [1])

            e_g += 0.5 * e1_g
            v_sg += 0.5 * v1_sg

        else:
            self.do_radial_hartree_corr(rgd, n_sg.sum(axis=0),
                                        nstar_sg.sum(axis=0),
                                        e_g, v_sg, [0])

    def do_radial_hartree_corr(self, rgd, n_g, nstar_g, e_g, v_sg, spins):
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS
        N = 2**14
        r_x = np.linspace(0, rgd.r_g[-1], N)
        G_k = np.linspace(0, np.pi / r_x[1], N // 2 + 1)

        dn_g = n_g - nstar_g

        dn_x = rgd.interpolate(dn_g, r_x)
        dn_k = self.radial_fft(r_x, dn_x)
        pot_k = np.zeros_like(dn_k)
        pot_k[1:] = dn_k[1:] / G_k[1:]**2 * 4 * np.pi * 0.5
        pot_x = self.radial_ifft(r_x, pot_k)
        pot_g = IUS(r_x, pot_x)(rgd.r_g)

        e_g -= pot_g * dn_g

        for s in spins:
            v_sg[s, :] -= 2 * (pot_g
                               - self.radial_derivative_fold(pot_g, n_g, s))
