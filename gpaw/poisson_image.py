import numpy as np
from gpaw.poisson_extended import ExtendedPoissonSolver as EPS
#from gpaw.utilities import h2gpts
from gpaw.dipole_correction import dipole_correction
#from ase.units import Bohr, Ha


# XXX add class description

class EPS2(EPS):
    def __init__(self, direction, side, *args, **kwargs):
        super(EPS2, self).__init__(*args, **kwargs)
        # Currently z-direction only. Fix this
        assert direction == 2
        self.direction = direction
        self.c = direction
        # left or right side for mirror charge?
        self.side = side

    def set_grid_descriptor(self, gd):
        super(EPS2, self).set_grid_descriptor(gd)
        cell_cv = self.gd.cell_cv
        # Allow only orthogonal cells for sanities sake
        for i in range(2):
            dir2 = (self.direction + 1 + i) % 3
            assert np.dot(cell_cv[self.direction], cell_cv[dir2]) < 1e-8
        # This is not supposed to be used with all periodic boundary conditions...
        assert not self.gd.pbc_c[self.direction]

    def get_description(self):
        description = super(EPS2, self).get_description()

        lines = [description]
        lines.extend(['    Adding mirror charges to neutralize the system'])
        return '\n'.join(lines)

    # Stolen from dipole_correction.fdsolve
    def solve(self, vHt_g, rhot_g, **kwargs):
        drhot_g, dvHt_g, potential_shift = dipole_correction(self.direction,
                                                             self.gd_original,
                                                             rhot_g)

        #print("drhot_g[z]: {}".format(drhot_g.mean(0).mean(0)))
        #print("dvHt_g[z]: {}".format(dvHt_g.mean(0).mean(0)*Ha))
        #print("pot shift: {}".format(potential_shift*Ha))

        # shift dipole potential to be zero at "electrode"
        if self.side == 'left':
            offset = (dvHt_g).mean(0).mean(0)[0]
        else:
            offset = (dvHt_g).mean(0).mean(0)[-1]  
        dvHt_g -= offset
        
        vHt_g -= dvHt_g

        iters = super(EPS2, self).solve(vHt_g, rhot_g + drhot_g, **kwargs)
        
        #from matplotlib import pyplot as pl
        #pl.plot(vHt_g.mean(0).mean(0)*Ha)
        #pl.plot((vHt_g+dvHt_g).mean(0).mean(0)*Ha, '-.')
        #pl.plot((dvHt_g).mean(0).mean(0)*Ha, '.')
        #pl.show()

        vHt_g += dvHt_g
        return iters

    def _solve(self, phi, rho, charge=None, eps=None, maxcharge=1e-6,
               zero_initial_phi=False):

        from gpaw.utilities.grid_redistribute import AlignedGridRedistributor
        #print('System charge: {}\nIntegrated charge: {}'.
              #format(charge, self.gd.integrate(rho))
        
        # Redistribute grid distriptor to reduce communication later.
        # Aka each rank has whole z-domain.
        gd = self.gd
        redist = AlignedGridRedistributor(gd, 0, 2)
        rho_xy = redist.forth(rho)

        # XXX ugly
        extend2_Nz_d = np.zeros(3, dtype=int)
        extend2_Nz_d[0] = gd.extend_N_cd[2, 0]
        extend2_Nz_d[1] = self.gd_original.N_c[2]
        extend2_Nz_d[2] = gd.extend_N_cd[2, 1]

        assert extend2_Nz_d.sum() == gd.N_c[2]
        
        oldcellstart = extend2_Nz_d[0]
        oldcellstop = extend2_Nz_d[0] + extend2_Nz_d[1]
        #print('start/stop: {} {}'.format(oldcellstart, oldcellstop))

        # XXX there is some problem with the fact, that the last element in rho
        # is always zero and acutally not part of the original array????????
        rho_central = rho_xy[:, :, oldcellstart:oldcellstop]
        if self.side == 'left':
            rho_xy[:, :, :oldcellstart] -= rho_central[:, :, ::-1]
        elif self.side == 'right':
            # Have to cut first element...
            rho_central = rho_central[:, :, 1:]
            rho_xy[:, :, oldcellstop:] -= rho_central[:, :, ::-1]
            print rho_xy.sum(0).sum(0)
        else:
            raise NotImplementedError
                
        mirror_rho = redist.back(rho_xy)
        rho = mirror_rho

        #from matplotlib import pyplot as pl
        #pl.plot(rho.sum(0).sum(0))
        #pl.show()
        
        x = super(EPS2, self)._solve(phi, rho, charge=0, eps=None,
                                     maxcharge=1e-6,
                                     zero_initial_phi=False)

        return x

