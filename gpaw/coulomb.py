from math import pi, sqrt

from ase.units import Hartree
import numpy as np
from numpy.fft import fftn

from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.pair_density import PairDensity2 as PairDensity
from gpaw.poisson import PoissonSolver, FFTPoissonSolver
from gpaw.utilities import unpack, packed_index, unpack2
from gpaw.utilities.ewald import madelung
from gpaw.utilities.tools import construct_reciprocal, tri2full, symmetrize
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.blas import r2k


def get_vxc(paw, spin=0, U=None):
    """Calculate matrix elements of the xc-potential."""
    assert not paw.hamiltonian.xc.xcfunc.orbital_dependent, "LDA/GGA's only"
    assert paw.wfs.dtype == float, 'Complex waves not implemented'

    if U is not None:  # Rotate xc matrix
        return np.dot(U.T.conj(), np.dot(get_vxc(paw, spin), U))

    gd = paw.hamiltonian.gd
    psit_nG = paw.wfs.kpt_u[spin].psit_nG[:]
    if paw.density.nt_sg is None:
        paw.density.interpolate_pseudo_density()
    nt_g = paw.density.nt_sg[spin]
    vxct_g = paw.density.finegd.zeros()
    paw.hamiltonian.xc.get_energy_and_potential(nt_g, vxct_g)
    vxct_G = gd.empty()
    paw.hamiltonian.restrict(vxct_g, vxct_G)
    Vxc_nn = np.zeros((paw.wfs.bd.nbands, paw.wfs.bd.nbands))

    # Apply pseudo part
    r2k(.5 * gd.dv, psit_nG, vxct_G * psit_nG, .0, Vxc_nn)  # lower triangle
    tri2full(Vxc_nn, 'L')  # Fill in upper triangle from lower
    gd.comm.sum(Vxc_nn)

    # Add atomic PAW corrections
    for a, P_ni in paw.wfs.kpt_u[spin].P_ani.items():
        D_sp = paw.density.D_asp[a][:]
        H_sp = np.zeros_like(D_sp)
        paw.wfs.setups[a].xc_correction.calculate_energy_and_derivatives(
            D_sp, H_sp)
        H_ii = unpack(H_sp[spin])
        Vxc_nn += np.dot(P_ni, np.dot(H_ii, P_ni.T))
    return Vxc_nn * Hartree


class Coulomb:
    """Class used to evaluate two index coulomb integrals."""
    def __init__(self, gd, poisson=None):
        """Class should be initialized with a grid_descriptor 'gd' from
           the gpaw module.
        """
        self.gd = gd
        self.poisson = poisson

    def load(self, method):
        """Make sure all necessary attributes have been initialized"""

        assert method in ('real', 'recip_gauss', 'recip_ewald'),\
            str(method) + ' is an invalid method name,\n' +\
            'use either real, recip_gauss, or recip_ewald'

        if method.startswith('recip'):
            if self.gd.comm.size > 1:
                raise RuntimeError("Cannot do parallel FFT, use method='real'")
            if not hasattr(self, 'k2'):
                self.k2, self.N3 = construct_reciprocal(self.gd)
            if method.endswith('ewald') and not hasattr(self, 'ewald'):
                # cutoff radius
                assert self.gd.orthogonal
                rc = 0.5 * np.average(self.gd.cell_cv.diagonal())
                # ewald potential: 1 - cos(k rc)
                self.ewald = (np.ones(self.gd.n_c) -
                              np.cos(np.sqrt(self.k2) * rc))
                # lim k -> 0 ewald / k2
                self.ewald[0, 0, 0] = 0.5 * rc**2
            elif method.endswith('gauss') and not hasattr(self, 'ng'):
                gauss = Gaussian(self.gd)
                self.ng = gauss.get_gauss(0) / sqrt(4 * pi)
                self.vg = gauss.get_gauss_pot(0) / sqrt(4 * pi)
        else:  # method == 'real'
            if not hasattr(self, 'solve'):
                if self.poisson is not None:
                    self.solve = self.poisson.solve
                else:
                    solver = PoissonSolver(nn=2)
                    solver.set_grid_descriptor(self.gd)
                    solver.initialize(load_gauss=True)
                    self.solve = solver.solve

    def coulomb(self, n1, n2=None, Z1=None, Z2=None, method='recip_gauss'):
        """Evaluates the coulomb integral of n1 and n2

        The coulomb integral is defined by::

                                      *
                      /    /      n1(r)  n2(r')
          (n1 | n2) = | dr | dr'  -------------,
                      /    /         |r - r'|

        where n1 and n2 could be complex.

        real:
           Evaluate directly in real space using gaussians to neutralize
           density n2, such that the potential can be generated by standard
           procedures

        recip_ewald:
           Evaluate by Fourier transform.
           Divergence at division by k^2 is avoided by utilizing the Ewald /
           Tuckermann trick, which formaly requires the densities to be
           localized within half of the unit cell.

        recip_gauss:
           Evaluate by Fourier transform.
           Divergence at division by k^2 is avoided by removing total charge
           of n1 and n2 with gaussian density ng::

                                                   *          *    *
            (n1|n2) = (n1 - Z1 ng|n2 - Z2 ng) + (Z2 n1 + Z1 n2 - Z1 Z2 ng | ng)

           The evaluation of the integral (n1 - Z1 ng|n2 - Z2 ng) is done in
           k-space using FFT techniques.
        """
        self.load(method)
        # determine integrand using specified method
        if method == 'real':
            I = self.gd.zeros()
            if n2 is None:
                n2 = n1
                Z2 = Z1
            self.solve(I, n2, charge=Z2, eps=1e-12, zero_initial_phi=True)
            I += madelung(self.gd.cell_cv) * self.gd.integrate(n2)
            I *= n1.conj()
        elif method == 'recip_ewald':
            n1k = fftn(n1)
            if n2 is None:
                n2k = n1k
            else:
                n2k = fftn(n2)
            I = n1k.conj() * n2k * self.ewald * 4 * pi / (self.k2 * self.N3)
        else:  # method == 'recip_gauss':
            # Determine total charges
            if Z1 is None:
                Z1 = self.gd.integrate(n1)
            if Z2 is None and n2 is not None:
                Z2 = self.gd.integrate(n2)

            # Determine the integrand of the neutral system
            # (n1 - Z1 ng)* int dr'  (n2 - Z2 ng) / |r - r'|
            nk1 = fftn(n1 - Z1 * self.ng)
            if n2 is None:
                I = abs(nk1)**2 * 4 * pi / (self.k2 * self.N3)
            else:
                nk2 = fftn(n2 - Z2 * self.ng)
                I = nk1.conj() * nk2 * 4 * pi / (self.k2 * self.N3)

            # add the corrections to the integrand due to neutralization
            if n2 is None:
                I += (2 * np.real(np.conj(Z1) * n1) -
                      abs(Z1)**2 * self.ng) * self.vg
            else:
                I += (np.conj(Z1) * n2 + Z2 * n1.conj() -
                      np.conj(Z1) * Z2 * self.ng) * self.vg
        if n1.dtype == float and (n2 is None or n2.dtype == float):
            return np.real(self.gd.integrate(I))
        else:
            return self.gd.integrate(I)


class CoulombNEW:
    def __init__(self, gd, setups, spos_ac, fft=False):
        assert gd.comm.size == 1
        self.rhot1_G = gd.empty()
        self.rhot2_G = gd.empty()
        self.pot_G = gd.empty()
        self.dv = gd.dv
        if fft:
            self.poisson = FFTPoissonSolver()
        else:
            self.poisson = PoissonSolver(nn=3)
        self.poisson.set_grid_descriptor(gd)
        self.poisson.initialize()
        self.setups = setups

        # Set coarse ghat
        self.Ghat = LFC(gd, [setup.ghat_l for setup in setups],
                        integral=sqrt(4 * pi))
        self.Ghat.set_positions(spos_ac)

    def calculate(self, nt1_G, nt2_G, P1_ap, P2_ap):
        I = 0.0
        self.rhot1_G[:] = nt1_G
        self.rhot2_G[:] = nt2_G

        Q1_aL = {}
        Q2_aL = {}
        for a, P1_p in P1_ap.items():
            P2_p = P2_ap[a]
            setup = self.setups[a]

            # Add atomic corrections to integral
            I += 2 * np.dot(P1_p, np.dot(setup.M_pp, P2_p))

            # Add compensation charges to pseudo densities
            Q1_aL[a] = np.dot(P1_p, setup.Delta_pL)
            Q2_aL[a] = np.dot(P2_p, setup.Delta_pL)
        self.Ghat.add(self.rhot1_G, Q1_aL)
        self.Ghat.add(self.rhot2_G, Q2_aL)

        # Add coulomb energy of compensated pseudo densities to integral
        self.poisson.solve(self.pot_G, self.rhot2_G, charge=None,
                           eps=1e-12, zero_initial_phi=True)
        I += np.vdot(self.rhot1_G, self.pot_G) * self.dv

        return I * Hartree


class HF:
    def __init__(self, paw):
        paw.initialize_positions()
        self.nspins = paw.wfs.nspins
        self.nbands = paw.wfs.bd.nbands
        self.restrict = paw.hamiltonian.restrict
        self.pair_density = PairDensity(paw.density, paw.atoms, finegrid=True)
        self.dv = paw.wfs.gd.dv
        self.dtype = paw.wfs.dtype
        self.setups = paw.wfs.setups

        # Allocate space for matrices
        self.nt_G = paw.wfs.gd.empty()
        self.rhot_g = paw.density.finegd.empty()
        self.vt_G = paw.wfs.gd.empty()
        self.vt_g = paw.density.finegd.empty()
        self.poisson_solve = paw.hamiltonian.poisson.solve

    def apply(self, paw, u=0):
        H_nn = np.zeros((self.nbands, self.nbands), self.dtype)
        self.soft_pseudo(paw, H_nn, u=u)
        self.atomic_val_val(paw, H_nn, u=u)
        self.atomic_val_core(paw, H_nn, u=u)
        return H_nn * Hartree

    def soft_pseudo(self, paw, H_nn, h_nn=None, u=0):
        if h_nn is None:
            h_nn = H_nn
        kpt = paw.wfs.kpt_u[u]
        pd = self.pair_density
        deg = 2 / self.nspins
        fmin = 1e-9
        Htpsit_nG = np.zeros(kpt.psit_nG.shape, self.dtype)

        for n1 in range(self.nbands):
            psit1_G = kpt.psit_nG[n1]
            f1 = kpt.f_n[n1] / deg
            for n2 in range(n1, self.nbands):
                psit2_G = kpt.psit_nG[n2]
                f2 = kpt.f_n[n2] / deg
                if f1 < fmin and f2 < fmin:
                    continue

                pd.initialize(kpt, n1, n2)
                pd.get_coarse(self.nt_G)
                pd.add_compensation_charges(self.nt_G, self.rhot_g)
                self.poisson_solve(self.vt_g, -self.rhot_g,
                                   charge=-float(n1 == n2), eps=1e-12,
                                   zero_initial_phi=True)
                self.restrict(self.vt_g, self.vt_G)
                Htpsit_nG[n1] += f2 * self.vt_G * psit2_G
                if n1 != n2:
                    Htpsit_nG[n2] += f1 * self.vt_G * psit1_G

                v_aL = paw.density.ghat.dict()
                paw.density.ghat.integrate(self.vt_g, v_aL)
                for a, v_L in v_aL.items():
                    v_ii = unpack(np.dot(paw.wfs.setups[a].Delta_pL, v_L))
                    P_ni = kpt.P_ani[a]
                    h_nn[:, n1] += f2 * np.dot(P_ni, np.dot(v_ii, P_ni[n2]))
                    if n1 != n2:
                        h_nn[:, n2] += f1 * np.dot(P_ni,
                                                   np.dot(v_ii, P_ni[n1]))

        symmetrize(h_nn)  # Grrrr why!!! XXX

        # Fill in lower triangle
        r2k(0.5 * self.dv, kpt.psit_nG[:], Htpsit_nG, 1.0, H_nn)

        # Fill in upper triangle from lower
        tri2full(H_nn, 'L')

    def atomic_val_val(self, paw, H_nn, u=0):
        deg = 2 / self.nspins
        kpt = paw.wfs.kpt_u[u]
        for a, P_ni in kpt.P_ani.items():
            # Add atomic corrections to the valence-valence exchange energy
            # --
            # >  D   C     D
            # --  ii  iiii  ii
            setup = paw.wfs.setups[a]
            D_p = paw.density.D_asp[a][kpt.s]
            H_p = np.zeros_like(D_p)
            D_ii = unpack2(D_p)
            ni = len(D_ii)
            for i1 in range(ni):
                for i2 in range(ni):
                    A = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            A += setup.M_pp[p13, p24] * D_ii[i3, i4]
                    p12 = packed_index(i1, i2, ni)
                    H_p[p12] -= 2 / deg * A / ((i1 != i2) + 1)
            H_nn += np.dot(P_ni, np.inner(unpack(H_p), P_ni.conj()))

    def atomic_val_core(self, paw, H_nn, u=0):
        kpt = paw.wfs.kpt_u[u]
        for a, P_ni in kpt.P_ani.items():
            dH_ii = unpack(-paw.wfs.setups[a].X_p)
            H_nn += np.dot(P_ni, np.inner(dH_ii, P_ni.conj()))
