from math import nan

# import numpy as np
# from ase.utils import seterr
# from scipy.special import erf

# from gpaw.typing import Array1D
from gpaw.core import PlaneWaves


class PoissonSolver:
    def solve(self,
              vHt,
              rhot) -> float:
        raise NotImplementedError


class PoissonSolverWrapper(PoissonSolver):
    def __init__(self, solver):
        self.description = solver.get_description()
        self.solver = solver

    def solve(self,
              vHt,
              rhot) -> float:
        self.solver.solve(vHt.data, rhot.data)
        return nan
