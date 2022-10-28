import pytest
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.fd_operators import Laplace
from gpaw.transformers import Transformer
from gpaw.mpi import world

def test_fd_ops(gpu):
    if world.size > 4:
        # Grid is so small that domain decomposition cannot exceed 4 domains
        assert world.size % 4 == 0
        group, other = divmod(world.rank, 4)
        ranks = np.arange(4 * group, 4 * (group + 1))
        domain_comm = world.new_communicator(ranks)
    else:
        domain_comm = world

    lat = 8.0
    for pbc in (True, False):
        gd = GridDescriptor((32, 32, 32), (lat, lat, lat), pbc_c=pbc, comm=domain_comm)

        if pbc:
            dtype = complex
            phase = np.ones((3, 2), complex)
        else:
            dtype = float
            phase = None

        # Use Gaussian as input
        x, y, z = gd.get_grid_point_coordinates()
        sigma = 1.5
        mu = lat / 2.0
     
        a = gd.zeros(dtype=dtype)
        a[:] = np.exp(-((x-mu)**2 + (y-mu)**2 + (z-mu)**2) / (2.0*sigma))
        # analytic solution
        b_analytic = np.zeros_like(a)
        b_analytic[:] = (((x-mu)**2 + (y-mu)**2 + (z-mu)**2)/sigma**2 - 3.0/sigma) * a

        b = np.zeros_like(a)
        a_gpu = gpu.copy_to_device(a)
        b_gpu = gpu.array.zeros_like(a_gpu)

        # Laplace
        Laplace(gd, 1.0, 3, dtype=dtype).apply(a, b, phase_cd=phase)
        Laplace(gd, 1.0, 3, dtype=dtype, cuda=True).apply(a_gpu, b_gpu, phase_cd=phase)
        b_ref = gpu.copy_to_host(b_gpu)

        assert b == pytest.approx(b_ref, abs=1e-12)
        # Neglect boundaries in check to analytic solution
        assert b_analytic[2:-2, 2:-2, 2:-2] == pytest.approx(b_ref[2:-2, 2:-2, 2:-2], abs=1e-2)

        # Transformers
        coarsegd = gd.coarsen()
        a_coarse = coarsegd.zeros(dtype=dtype)
        a_coarse_gpu = gpu.array.zeros_like(a_coarse)

        # Restrict
        Transformer(gd, coarsegd, 1, dtype=dtype).apply(a, a_coarse, phases=phase)
        # Interpolate
        Transformer(coarsegd, gd, 1, dtype=dtype).apply(a_coarse, a, phases=phase)

        # Restrict
        Transformer(gd, coarsegd, 1, dtype=dtype, cuda=True).apply(a_gpu, a_coarse_gpu, phases=phase)
        # Interpolate
        Transformer(coarsegd, gd, 1, dtype=dtype, cuda=True).apply(a_coarse_gpu, a_gpu, phases=phase)
        a_ref = gpu.copy_to_host(a_gpu)
    
        assert a == pytest.approx(a_ref, abs=1e-14)
