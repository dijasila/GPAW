from math import sqrt, pi
import numpy as np

from gpaw.utilities import pack
from gpaw.utilities.tools import pick
from gpaw.lfc import LocalizedFunctionsCollection as LFC


class PairDensity:
    def  __init__(self, paw):
        """basic initialisation knowing"""

        self.interpolate = paw.density.interpolator.apply
        self.fineghat = paw.density.ghat
        self.gd = paw.density.gd
        self.setups = paw.wfs.setups
        self.spos_ac = paw.atoms.get_scaled_positions()
        assert paw.wfs.dtype == float

        self.ghat = None
        
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
        return self.interpolate(nijt)

    def with_compensation_charges(self, finegrid=False):
        """Get pair densisty including the compensation charges"""
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

        if self.ghat is None:
            if finegrid:
                self.ghat = self.fineghat
            else:
                self.ghat = LFC(self.gd,
                                [setup.ghat_l
                                 for setup in self.setups],
                                integral=sqrt(4 * pi))
                self.ghat.set_positions(self.spos_ac)

        # Add compensation charges
        self.ghat.add(rhot, Q_aL)
                
        return rhot
