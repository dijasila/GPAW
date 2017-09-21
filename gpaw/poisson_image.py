import numpy as np
from gpaw.poisson_extended import ExtendedPoissonSolver as EPS
from gpaw.poisson import FDPoissonSolver
from gpaw.dipole_correction import dipole_correction
from gpaw.utilities.extend_grid import extended_grid_descriptor
from gpaw.utilities.timing import nulltimer

# TODO
# 1. Documentation
# 2. Working examples (there is a test already)
# 3. 'right' gives very wrong results, why? implicit zero of omitted first array
#    element?
# 4. allow arbitrary directions?

class ImagePoissonSolver(EPS):
    """ Extended Poisson solver with image/mirror charges
    and dipole corrections

    This class is intended for calculations of charged unit cells with mixed
    boundary conditions. A typical application would be capacitor half cells or
    surface slabs with absorbed ions.

    Workings: First the dipole correction is applied to remove dipole
    components. Then the cell is extended in a given directions (currently
    fixed to z). Then the image charge is appiled in one of the extended
    regions, either 'left' or 'right' of the actual cell. Currently only 'left'
    is giving correct results though. The corresponding face of the cell is
    acting like a perfect absorber. This system is then given to the Poisson
    solver. The results is de-extended and the dipole component are added back
    after.

    Usage:
    gpts = h2gpts(0.2, atoms.cell, idiv=16)
    calc = GPAW(mode='lcao',
            gpts=gpts,
            charge=1,
            basis='dzp',
            poissonsolver=ImagePoissonSolver(direction=2, side='left'),
               )

    """

    def __init__(self, direction, side, nn=3, relax='J', eps=2e-10,
                 maxiter=1000, extended=None, timer=nulltimer):
        # super(ImagePoissonSolver, self).__init__(*args, **kwargs)
        FDPoissonSolver.__init__(self, nn=nn, relax=relax,
                                 eps=eps, maxiter=maxiter,
                                 remove_moment=None)

        self.timer = timer
        self.moment_corrections = None
        # Broadcast over band, kpt, etc. communicators required?
        self.requires_broadcast = False
        self.is_extended = True
        if extended is None:
            extended={'useprev': True}
        self.extended = extended

        assert 'useprev' in extended.keys(), 'useprev parameter is missing'
        if self.extended.get('comm') is not None:
            self.requires_broadcast = True

        # XXX Currently z-direction only. Fix this
        assert direction == 2
        self.direction = direction
        self.c = direction
        # left or right side for mirror charge?
        assert side == 'left' or side == 'right'
        self.side = side
        self.correction = None

    def todict(self):
        dct = super(ImagePoissonSolver, self).todict()
        dct['name'] = 'ifd'
        dct['direction'] = self.c
        dct['side'] = self.side
        return dct

    def set_grid_descriptor(self, gd):
        # super(ImagePoissonSolver, self).set_grid_descriptor(gd)
        self.gd_original = gd
        finegpts = self.gd_original.N_c.copy()
        finegpts[self.c] *= 2
        self.extended['finegpts'] = finegpts
        self.extended['gpts'] = finegpts // 2
        extend_N_cd = np.zeros((3,2), dtype=int)
        if  self.side == 'left':
            extend_N_cd[self.c,0] = self.gd_original.N_c[self.c]
        else:
            extend_N_cd[self.c,1] = self.gd_original.N_c[self.c]
        gd, _, _ = extended_grid_descriptor(gd, extend_N_cd=extend_N_cd,
                                            extcomm=self.extended.get('comm'))

        FDPoissonSolver.set_grid_descriptor(self, gd)

        # Allow only orthogonal cells for sanities sake
        assert gd.orthogonal
        # This is not supposed to be used with all periodic boundary conditions...
        assert not self.gd.pbc_c[self.direction]

    def get_description(self):
        description = super(ImagePoissonSolver, self).get_description()

        lines = [description]
        lines.extend(['    Adding mirror charges to neutralize the system'])
        return '\n'.join(lines)

    # Stolen from dipole_correction.fdsolve
    def solve(self, vHt_g, rhot_g, **kwargs):
        drhot_g, dvHt_g, self.correction = dipole_correction(self.direction,
                                                             self.gd_original,
                                                             rhot_g)

        # shift dipole potential to be zero at "electrode"
        
        # XXX Use self.correction to shift the potential. The question is 
        # is just, how do you figure out, if it's + or - 
        if self.side == 'right':
            # XXX "right" doesn't work. This ugly fix helps, but it's still wrong.
            drhot_g *= -1.
            dvHt_g *= -1.
            #self.correction *= -1. 

        dvHt_g -= self.correction
        # XXX hamiltonian.get_electrostatic_potential() uses this one. We don't like that.
        self.correction = 0.0
        vHt_g -= dvHt_g

        iters = super(ImagePoissonSolver, self).solve(vHt_g, rhot_g + drhot_g,
                                                      **kwargs)
        
        vHt_g += dvHt_g
        return iters

    def _solve(self, phi, rho, charge=None, eps=None, maxcharge=1e-6,
               zero_initial_phi=False):

        from gpaw.utilities.grid_redistribute import AlignedGridRedistributor
        
        # Redistribute grid distriptor to reduce communication later.
        # Aka each rank has whole z-domain.
        gd = self.gd
        redist = AlignedGridRedistributor(gd, 0, 2) # XXX direction
        rho_xy = redist.forth(rho)

        # Our direction is non-periodic, so we have one element less in array.
        # First element of original array is implicit zero.
        # use [1,2,3,4] -> [4,3,2,1,0,1,2,3,4] or [1,2,3,4,0,4,3,2,1]
        if self.side == 'left':
            startpoint = gd.extend_N_cd[self.c, 0]
        else:
            startpoint = 0
        stoppoint = startpoint + self.gd_original.N_c[self.c] - 1

        rho_central = rho_xy[:, :, startpoint:stoppoint] # XXX direction

        if self.side == 'left':
            rho_xy[:, :, 0:startpoint-1] -= rho_central[:, :, ::-1]
            rho_xy[:, :, startpoint-1] = 0.0
        else:
            rho_xy[:, :, stoppoint+1:] -= rho_central[:, :, ::-1]
            rho_xy[:, :, stoppoint] = 0.0

        rho = redist.back(rho_xy)

        x = super(ImagePoissonSolver, self)._solve(phi, rho, charge=0,
                                                   eps=None,
                                                   maxcharge=1e-6,
                                                   zero_initial_phi=False)

        return x
