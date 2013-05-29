from math import sqrt, pi
import numpy as np

from gpaw.utilities import pack, unpack2
from gpaw.utilities.tools import pick
from gpaw.lfc import LocalizedFunctionsCollection as LFC, BasisFunctions


class PairDensity2:
    def  __init__(self, density, atoms, finegrid):
        """Initialization needs a paw instance, and whether the compensated
        pair density should be on the fine grid (boolean)"""

        self.density = density
        self.finegrid = finegrid

        if not finegrid:
            density.Ghat = LFC(density.gd,
                               [setup.ghat_l
                                for setup in density.setups],
                               integral=sqrt(4 * pi))
            density.Ghat.set_positions(atoms.get_scaled_positions() % 1.0)

    def initialize(self, kpt, n1, n2):
        """Set wave function indices."""
        self.n1 = n1
        self.n2 = n2
        self.spin = kpt.s
        self.P_ani = kpt.P_ani
        self.psit1_G = pick(kpt.psit_nG, n1)
        self.psit2_G = pick(kpt.psit_nG, n2)

    def get_coarse(self, nt_G):
        """Get pair density"""
        np.multiply(self.psit1_G.conj(), self.psit2_G, nt_G)

    def add_compensation_charges(self, nt_G, rhot_g):
        """Add compensation charges to input pair density, which
        is interpolated to the fine grid if needed."""

        if self.finegrid:
            # interpolate the pair density to the fine grid
            self.density.interpolator.apply(nt_G, rhot_g)
        else:
            # copy values
            rhot_g[:] = nt_G
        
        # Determine the compensation charges for each nucleus
        Q_aL = {}
        for a, P_ni in self.P_ani.items():
            assert P_ni.dtype == float
            # Generate density matrix
            P1_i = P_ni[self.n1]
            P2_i = P_ni[self.n2]
            D_ii = np.outer(P1_i.conj(), P2_i)
            # allowed to pack as used in the scalar product with
            # the symmetric array Delta_pL
            D_p  = pack(D_ii, tolerance=1e30)
            
            # Determine compensation charge coefficients:
            Q_aL[a] = np.dot(D_p, self.density.setups[a].Delta_pL)

        # Add compensation charges
        if self.finegrid:
            self.density.ghat.add(rhot_g, Q_aL)
        else:
            self.density.Ghat.add(rhot_g, Q_aL)


class PairDensity:
    def  __init__(self, paw):
        self.set_paw(paw)
        
    def set_paw(self, paw):
        """basic initialisation knowing"""
        self.wfs = paw.wfs
        self.density = paw.density
        self.setups = paw.wfs.setups
        self.spos_ac = paw.atoms.get_scaled_positions()
        assert paw.wfs.dtype == float
        
    def initialize(self, kpt, i, j):
        """initialize yourself with the wavefunctions"""
        self.i = i
        self.j = j
        self.spin = kpt.s
        self.P_ani = kpt.P_ani
        
        self.wfi = kpt.psit_nG[i]
        self.wfj = kpt.psit_nG[j]

    def get(self, finegrid=False):
        """Get pair density"""
        nijt = self.wfi * self.wfj
        if not finegrid:
            return nijt 

        # interpolate the pair density to the fine grid
        nijt_g = self.density.finegd.empty()
        self.density.interpolator.apply(nijt, nijt_g)

        return nijt_g

    def with_compensation_charges(self, finegrid=False):
        """Get pair density including the compensation charges"""
        rhot = self.get(finegrid)

        # Determine the compensation charges for each nucleus
        Q_aL = {}
        for a, P_ni in self.P_ani.items():
            # Generate density matrix
            Pi_i = P_ni[self.i]
            Pj_i = P_ni[self.j]
            D_ii = np.outer(Pi_i, Pj_i)
            # allowed to pack as used in the scalar product with
            # the symmetric array Delta_pL
            D_p  = pack(D_ii)
            
            # Determine compensation charge coefficients:
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)

        # Add compensation charges
        if finegrid:
            self.density.ghat.add(rhot, Q_aL)
        else:
            if not hasattr(self.density, 'Ghat'):
                self.density.Ghat = LFC(self.density.gd,
                                        [setup.ghat_l
                                         for setup in self.setups],
                                        integral=sqrt(4 * pi))
                self.density.Ghat.set_positions(self.spos_ac)
            self.density.Ghat.add(rhot, Q_aL)
                
        return rhot

    def with_ae_corrections(self, finegrid=False):
        """Get pair density including the AE corrections"""
        nij_g = self.get(finegrid)
        
        # Generate the density matrix
        D_ap = {}
#        D_aii = {}
        for a, P_ni in self.P_ani.items():
            Pi_i = P_ni[self.i]
            Pj_i = P_ni[self.j]
            D_ii = np.outer(Pi_i.conj(), Pj_i)
            # Note: D_ii is not symmetric but the products of partial waves are
            # so that we can pack
            D_ap[a] = pack(D_ii)
#            D_aii[a] = D_ii
        
        # Load partial waves if needed
        if ((finegrid and (not hasattr(self, 'phi'))) or
            ((not finegrid) and (not hasattr(self, 'Phi')))):
            
            # Splines
            splines = {}
            phi_aj = []
            phit_aj = []
            for a, id in enumerate(self.setups.id_a):
                if id in splines:
                    phi_j, phit_j = splines[id]
                else:
                    # Load splines:
                    phi_j, phit_j = self.setups[a].get_partial_waves()[:2]
                    splines[id] = (phi_j, phit_j)
                phi_aj.append(phi_j)
                phit_aj.append(phit_j)
            
            # Store partial waves as class variables
            if finegrid:
                gd = self.density.finegd
                self.__class__.phi = BasisFunctions(gd, phi_aj)
                self.__class__.phit = BasisFunctions(gd, phit_aj)
                self.__class__.phi.set_positions(self.spos_ac)
                self.__class__.phit.set_positions(self.spos_ac)
            else:
                gd = self.density.gd
                self.__class__.Phi = BasisFunctions(gd, phi_aj)
                self.__class__.Phit = BasisFunctions(gd, phit_aj)
                self.__class__.Phi.set_positions(self.spos_ac)
                self.__class__.Phit.set_positions(self.spos_ac)
        
        # Add AE corrections
        if finegrid:
            phi = self.phi
            phit = self.phit
            gd = self.density.finegd
        else:
            phi = self.Phi
            phit = self.Phit
            gd = self.density.gd
        
        rho_MM = np.zeros((phi.Mmax, phi.Mmax))
        M1 = 0
        for a, setup in enumerate(self.setups):
            ni = setup.ni
            D_p = D_ap.get(a)
            if D_p is None:
                D_p = np.empty((ni * (ni + 1) // 2))
            if gd.comm.size > 1:
                gd.comm.broadcast(D_p, self.wfs.rank_a[a])
            D_ii = unpack2(D_p)
#            D_ii = D_aii.get(a)
#            if D_ii is None:
#                D_ii = np.empty((ni, ni))
#            if gd.comm.size > 1:
#                gd.comm.broadcast(D_ii, self.wfs.rank_a[a])
            M2 = M1 + ni
            rho_MM[M1:M2, M1:M2] = D_ii
            M1 = M2
        
        # construct_density assumes symmetric rho_MM and
        # takes only the upper half of it
        phi.construct_density(rho_MM, nij_g, q=-1)
        phit.construct_density(-rho_MM, nij_g, q=-1)
        # TODO: use ae_valence_density_correction
#        phi.lfc.ae_valence_density_correction(
#            rho_MM, nij_g, np.zeros(len(phi.M_W), np.intc), np.zeros(self.na))
#        phit.lfc.ae_valence_density_correction(
#            -rho_MM, nij_g, np.zeros(len(phit.M_W), np.intc), np.zeros(self.na))
            
        return nij_g
