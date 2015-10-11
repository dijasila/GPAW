import numpy as np

from functools import partial
from gpaw.utilities.grid_redistribute import general_redistribute
from gpaw.grid_descriptor import GridDescriptor

class MoreParallelPoissonSolver:
    def __init__(self, solver):
        self.solver = solver

    def get_stencil(self):
        return self.solver.get_stencil()

    def set_grid_descriptor(self, gd):
        big_gd = gd.new_descriptor(comm=gd.comm.parent)
        
        if big_gd.comm.rank == 0:
            big_ranks = gd.comm.translate_ranks(big_gd.comm,
                                                np.arange(gd.comm.size))
            big_ranks = big_ranks.astype(int) # XXXX WTF np.int32
        else:
            big_ranks = np.empty(gd.comm.size, dtype=int)
        big_gd.comm.broadcast(big_ranks, 0)
        
        bigrank2rank = dict(zip(big_ranks, np.arange(gd.comm.size)))
        def rank2parpos1(rank):
            if rank in bigrank2rank:
                return gd.get_processor_position_from_rank(bigrank2rank[rank])
            else:
                return None

        rank2parpos2 = big_gd.get_processor_position_from_rank

        self.distribute = partial(general_redistribute, big_gd.comm,
                                  gd.n_cp, big_gd.n_cp,
                                  rank2parpos1, rank2parpos2)
        self.collect = partial(general_redistribute, big_gd.comm,
                               big_gd.n_cp, gd.n_cp,
                               rank2parpos2, rank2parpos1)
            
        self.solver.set_grid_descriptor(big_gd)

    def get_description(self):
        orig_txt = self.solver.get_description()
        nxnxn = ' x '.join(str(n) for n in self.solver.gd.parsize_c)
        desc = 'Extended parallelization of Poisson solver: %s' % nxnxn
        return '\n'.join([orig_txt, desc])

    def initialize(self):
        self.solver.initialize()

    def solve(self, phi, rho, **kwargs):
        dist_rho = self.solver.gd.empty()
        dist_phi = self.solver.gd.empty()
        self.distribute(rho, dist_rho)
        self.distribute(phi, dist_phi)
        niter = self.solver.solve(dist_phi, dist_rho, **kwargs)
        self.collect(dist_phi, phi)
        return niter

    def estimate_memory(self, mem):
        return self.solver.estimate_memory(mem)
