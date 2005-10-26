# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gridpaw.interaction import GInteraction2 as GInteraction
from gridpaw.neighbor_list import NeighborList


class Neighbor:
    def __init__(self, v, dvdr, nucleus):
        self.nucleus = nucleus
        self.v = v
        self.dvdr = dvdr


class PairPotential:
    def __init__(self, domain, setups):
        self.cell_i = domain.cell_i
        self.bc_i = domain.periodic_i

        # Collect the pair potential cutoffs in a list:
        self.cutoff_a = []
        for Z, setup in setups.items():
            self.cutoff_a.append((Z, setup.rcut2))
        
        # Make pair interactions:
        items = setups.items()
        self.interactions = {}
        for Z1, setup1 in items:
            for Z2, setup2 in items:
                interaction = GInteraction(setup1, setup2)
                self.interactions[(Z1, Z2)] = interaction

        self.neighborlist = None
        
    def update(self, pos_ai, nuclei):
        if self.neighborlist is None:
            # Make a neighbor list object:
            drift = 0.3
            Z_a = [nucleus.setup.Z for nucleus in nuclei]
            self.neighborlist = NeighborList(Z_a, pos_ai,
                                             self.cell_i, self.bc_i,
                                             self.cutoff_a, drift)
            updated = False
        else:
            updated = self.neighborlist.update_list(pos_ai)

        # Reset all pairs:
        for nucleus in nuclei:
            nucleus.neighbors = []

        # Make new pairs:
        for n1 in range(len(nuclei)):
            nucleus1 = nuclei[n1]
            Z1 = nucleus1.setup.Z
            for n2, offsets in self.neighborlist.neighbors(n1):
                nucleus2 = nuclei[n2]
                Z2 = nucleus2.setup.Z
                interaction = self.interactions[(Z1, Z2)]
                diff = pos_ai[n2] - pos_ai[n1]
                V = num.zeros(interaction.v_LL.shape, num.Float)
                dVdr = num.zeros(interaction.dvdr_LLi.shape, num.Float)
                for offset in offsets:
                    difference = diff + offset
                    v, dvdr = interaction(difference)
                    V += v
                    dVdr += dvdr
#                print V, dVdr
                nucleus1.neighbors.append(Neighbor(V, dVdr, nucleus2))
                if nucleus2 is not nucleus1:
                    nucleus2.neighbors.append(
                        Neighbor(num.transpose(V),
                                 -num.transpose(dVdr, (1, 0, 2)), nucleus1))
        return updated

    def print_info(self, out):
        pass
    """

        print >> out, 'pair potential:'
        print >> out, '  cutoffs:'
            print >> out, '   ', setup.symbol, setup.rcut2 * a0
        npairs = self.neighborlist.number_of_pairs() - len(pos_ai)
        if npairs == 0:
            print >> out, '  There are no pair interactions.'
        elif npairs == 1:
            print >> out, '  There is one pair interaction.'
        else:
            print >> out, '  There are %d pair interactions.' % npairs"""
