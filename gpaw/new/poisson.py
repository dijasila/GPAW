from math import nan


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
