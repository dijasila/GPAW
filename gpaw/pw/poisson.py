from math import pi


class ReciprocalSpacePoissonSolver:
    def __init__(self, pd, realpbc_c):
        self.pd = pd
        self.realpbc_c = realpbc_c
        self.G2_q = pd.G2_qG[0]
        if pd.gd.comm.rank == 0:
            # Avoid division by zero:
            self.G2_q[0] = 1.0

    def initialize(self):
        pass

    def get_stencil(self):
        return '????'

    def estimate_memory(self, mem):
        pass

    def todict(self):
        return {}

    def solve(self, vHt_q, dens):
        vHt_q[:] = 4 * pi * dens.rhot_q
        vHt_q /= self.G2_q
