from gpaw.utilities.gauss import Gaussian
import numpy as np

# Quantum subsystem contains density in qmgd, and this
# class solves the potential that this density creates
# in the larger classical subsystem, which is defined by the
# GridDescriptor clgd. Various approaches can be implemented.
#  
class PotentialTransformer:
    def __init__(self,
                  clgd,
                  qmgd,
                  index_offset_1,
                  index_offset_2):
        self.clgd = clgd
        self.qmgd = qmgd
        self.index_offset_1 = index_offset_1
        self.index_offset_2 = index_offset_2
    
    def getPotential(self):
        pass

# Calculate the multipole moments of the quantum density, then
# use them to determine the potential also outside quantum system. 
class GaussianPotentialTransformer(PotentialTransformer):
    def __init__(self,
                 clgd,
                 qmgd,
                 index_offset_1,
                 index_offset_2,
                 num_refinements,
                 coarseners,
                 clgd_global,
                 remove_moment_qm):
        PotentialTransformer.__init__(self,
                                      clgd,
                                      qmgd,
                                      index_offset_1,
                                      index_offset_2)
        
        self.num_refinements = num_refinements
        self.coarseners = coarseners
        self.clgd_global = clgd_global
        self.remove_moment_qm = remove_moment_qm
        

    def getPotential(self,
                      center,
                      inside_potential,         # already calculated potential inside quantum grid 
                      outside_potential = None, # potential outside is saved here 
                      rho = None,               # quantum charge density
                      moments = None            # multipole moments of the quantum charge density
                      ):
        if moments == None: # moments not provided, so they must be calculated from rho
            assert(rho != None)
            _moments = []
            for L in range(self.remove_moment_qm):
                _moments.append(Gaussian(self.qmgd).get_moment(rho, L))
        else:
            _moments = moments
        
        # From quantum to classical grid outside the overlapping region (multipole expansion)
        outside_potential = np.sum(np.array([m * Gaussian(self.clgd_global, center=center).get_gauss_pot(l) for m, l in zip(_moments, range(self.remove_moment_qm))]), axis=0)
        outside_potential[self.index_offset_1[0]:self.index_offset_2[0] - 1,
                          self.index_offset_1[1]:self.index_offset_2[1] - 1,
                          self.index_offset_1[2]:self.index_offset_2[2] - 1] = 0.0
        
        # From quantum to classical grid inside the overlapping region (coarsened potential)
        for n in range(self.num_refinements):
            inside_potential = self.coarseners[n].apply(inside_potential)
        
        # Combine inside and outside potentials
        full_potential = np.copy(outside_potential)
        full_potential[self.index_offset_1[0]:self.index_offset_2[0] - 1,
                       self.index_offset_1[1]:self.index_offset_2[1] - 1,
                       self.index_offset_1[2]:self.index_offset_2[2] - 1] += inside_potential[:]
        
        return full_potential
    