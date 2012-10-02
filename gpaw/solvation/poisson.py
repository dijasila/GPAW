from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.fd_operators import Laplace, Gradient
from gpaw.wfd_operators import WeightedFDOperator
import numpy


class SolvationPoissonSolver(PoissonSolver):
    """
    Poisson solver including an electrostatic solvation model

    following Sanchez et al J. Chem. Phys. 131 (2009) 174108
    """

    def __init__(self, nn=3, relax='J', eps=2e-10, op_weights=None):
        self.op_weights = op_weights
        PoissonSolver.__init__(self, nn, relax, eps)

    def solve(self, phi, rho, charge=None, eps=None,
              #maxcharge=1e-6,
              maxcharge=1000.,
              zero_initial_phi=False):
        if not abs(charge) <= maxcharge:
            raise NotImplementedError(
                'SolvationPoissonSolver supports only '
                'neutral systems up to now.'
                )
        self.restrict_op_weights()
        ret = PoissonSolver.solve(self, phi, rho, charge, eps, maxcharge,
                                  zero_initial_phi)
        assert numpy.isfinite(phi).all() # XXX
        return ret

    def restrict_op_weights(self):
        weights = [self.op_weights] + self.op_coarse_weights
        for i, res in enumerate(self.restrictors):
            for j in xrange(4):
                res.apply(weights[i][j], weights[i + 1][j])
        self.step = 0.66666666 / self.operators[0].get_diagonal_element()

    def set_grid_descriptor(self, gd):
        if gd.pbc_c.any():
            raise NotImplementedError(
                'SolvationPoissonSolver supports only '
                'non-periodic boundary conditions up to now.'
                )
        self.gd = gd
        self.gds = [gd]
        self.dv = gd.dv
        gd = self.gd
        self.B = None
        self.interpolators = []
        self.restrictors = []
        self.operators = []
        level = 0
        self.presmooths = [2]
        self.postsmooths = [1]
        self.weights = [2. / 3.]
        while level < 4:
            try:
                gd2 = gd.coarsen()
            except ValueError:
                break
            self.gds.append(gd2)
            self.interpolators.append(Transformer(gd2, gd))
            self.restrictors.append(Transformer(gd, gd2))
            self.presmooths.append(4)
            self.postsmooths.append(4)
            self.weights.append(1.0)
            level += 1
            gd = gd2
        self.levels = level

    def initialize(self, load_gauss=False):
        self.presmooths[self.levels] = 8
        self.postsmooths[self.levels] = 8
        self.phis = [None] + [gd.zeros() for gd in self.gds[1:]]
        self.residuals = [gd.zeros() for gd in self.gds]
        self.rhos = [gd.zeros() for gd in self.gds]
        self.op_coarse_weights = [[g.empty() for g in (gd, ) * 4] \
                               for gd in self.gds[1:]]
        scale = -0.25 / numpy.pi
        for i, gd in enumerate(self.gds):
            if i == 0:
                nn = self.nn
                weights = self.op_weights
            else:
                nn = 1
                weights = self.op_coarse_weights[i - 1]
            operators = [Laplace(gd, scale, nn)] + \
                        [Gradient(gd, j, scale, nn) for j in (0, 1, 2)]
            self.operators.append(WeightedFDOperator(weights, operators))
        if load_gauss:
            self.load_gauss()
        if self.relax_method == 1:
            self.description = 'Gauss-Seidel'
        else:
            self.description = 'Jacobi'
        self.description += ' solver with dielectric and ' \
                            '%d multi-grid levels' % (self.levels + 1, )
        self.description += '\nStencil: ' + self.operators[0].description
