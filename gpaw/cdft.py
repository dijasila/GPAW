import numpy as np
from ase.data import covalent_radii


def gaussians(gd, positions, numbers):
    r_Rv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
    radii = covalent_radii[numbers]
    cutoffs = radii + 3.0
    sigmas = radii * min(covalent_radii) + 0.5
    result_R = gd.zeros()
    for pos, Z, rc, sigma in zip(positions, numbers, cutoffs, sigmas):
        d2_R = ((r_Rv - pos)**2).sum(3)
        a_R = Z / (sigma * (2 * np.pi)**0.5) * np.exp(-d2_R / (2 * sigma**2))
        a_R[d2_R > rc**2] = 0.0
        result_R += a_R
    return result_R
    
    
class CDFT(Calculator):
    ... = ['energy']
    
    def __init__(self, calc, charges, regions):
        self.calc = calc
        self.charge_i = np.array(charges)
        self.indices_i = regions
        self.n_iR = []
        
    def ():
        for indices in self.indices_i:
            n_R = gau
    
    
