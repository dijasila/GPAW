import numpy as np
from functools import partial

from ase.parallel import parprint

import gpaw.mpi as mpi

def memory_estimate(mynw, mynG, nG, optical_limit):
    """Estimate memory requirements of matrices."""

    chi0_wGG_size = mynw*mynG*nG
    chi0_wxvG_size = mynw*2*3*nG*optical_limit
    chi0_wvv_size = mynw*3*3*optical_limit

    return tuple([s*16/1024**2 for s in [chi0_wGG_size, chi0_wxvG_size, chi0_wvv_size]])

def check_grid(nw, nG, mempc, q_c, nblocks=1, response='density', f=None):
    """Check if grid in response calculation fit the parallelization
    
    nw : int
       Number of frequency points
    nG : int
       Number of G-vectors in calculation
    mempc : float
       Current memory per cpu
    q_c : list of ndarray
       Momentum vector
    nblocks : int
       Number of memory blocks
    response : str
       Type of response function, 'density' or 'spin'
    f : file handle
       Output file
    """

    if f:
        p = partial(parprint, file=f)
    else:
        p = partial(parprint)
    
    optical_limit = np.allclose(q_c, 0.0) and response == 'density'
    
    (chi0_wGG_mem, chi0_wxvG_mem, chi0_wvv_mem) = memory_estimate(nw, nG, nG, optical_limit)
    totmem = chi0_wGG_mem + chi0_wxvG_mem + chi0_wvv_mem
    
    
    p("--------------------------------------------------")
    p("Checking memory requirements for chi0 calculation")
    p("\n")
    p("q_c: %s\n" % str([e for e in q_c]))
    p("Number of frequency points: %d" % nw)
    p("Number of G-vectors: %d" % nG)
    p("Memory estimates:")
    p("\tchi0_wGG: %f MiB" % chi0_wGG_mem)
    p("\tchi0_wxvG: %f MiB" % chi0_wxvG_mem)
    p("\tchi0_wvv: %f MiB" % chi0_wvv_mem)
    p("subtotal: %f MiB" % totmem)
    p("\tworld.size: %d" % mpi.world.size)
    p("\tMemory usage before allocation: %f MiB / cpu" % mempc)
    totmem += mempc
    p("Total: %f MiB / cpu" % totmem)

    (addG, addw) = (0, 0)
    if nblocks > 1:
        totmem = 0.
        sG = (nG + nblocks - 1) // nblocks
        sw = (nw + nblocks - 1) // nblocks
        p("\nDistributing data in %d blocks" % nblocks)
        for rank in range(nblocks):
            Ga = min(rank*sG,nG)
            mynG = min(Ga+sG,nG) - Ga
            wa = min(rank*sw,nw)
            mynw = min(wa+sw,nw) - wa
            (chi0_wmyGG_mem, chi0_wxvG_mem, chi0_wvv_mem) = memory_estimate(nw, mynG, nG, optical_limit)
            (chi0_mywGG_mem, chi0_mywxvG_mem, chi0_mywvv_mem) = memory_estimate(mynw, nG, nG, optical_limit)
            p("\nBlock %d" % (rank+1))
            p("Number of frequency points: %d" % mynw)
            p("Number of G-vectors: %d" % mynG)
            p("Total memory estimates in block for the two steps:")
            Gtotmem = mempc + chi0_wmyGG_mem + chi0_wxvG_mem + chi0_wvv_mem
            ftotmem = mempc + chi0_mywGG_mem + chi0_mywxvG_mem + chi0_mywvv_mem
            p("\tWhen distributing over frequencies: %f MiB / cpu" % ftotmem)
            p("\tWhen distributing over G-vectors: %f MiB / cpu" % Gtotmem)

            if mynG == 0:
                addG += sG
            if mynw == 0:
                addw += sw
            
            if Gtotmem > totmem:
                totmem = Gtotmem
            if ftotmem > totmem:
                totmem = ftotmem
        
        p("\n")
        if addG:
            p("\nWARNING: parallelization will fail with these settings")
            p("\tAdd %d G-vectors to allocate computations among all blocks" % addG)
            if addw:
                p("\tAdd %d frequencies to allocate computations among all blocks" % addw)
        elif addw:
            p("Add %d frequencies to allocate computations among all blocks" % addw)
        else:
            p("Computations have been allocated among all blocks")

    return (addG, addw, totmem)



def check_scrf_mem(scrf, q_c, response, f=None):
   """Check if input parameters in response calculation fit the parallelization
       
   scrf : obj
       The SpinChargeResponseFunction object
   q_c : list of ndarray
       Momentum vector
   response : str
       Type of response function, 'density' or 'spin'
   f : file handle
       Output file
   
   - omegacutlower and omegacutupper and domega0, omega2 and omegamax or frequencies
       will determine number of frequencies in calculation
    - cell, q_c and ecut will determine the number of G-vectors in calculation
    - nblocks will determine how memory and computations are distributed
    
    - if G and w grids are dense, parameters have to be chosen in such a way that the calculation
       has enough memory, and each rank in the G-vector parallelization has computations to do
   """
   
   scrf.chi0.set_response(response)
   (nw, nG, mempc) = scrf.chi0.get_chi0_mem_data(q_c)

   return check_grid(nw, nG, mempc, q_c, nblocks=scrf.chi0.nblocks, response=response, f=f)


if __name__ == "__main__":
    from gpaw.response.scrf import SpinChargeResponseFunction
    
    gs = '/home/niflheim/s134264/magnetic_response/calculations/code_development_test/Febcc/ground_state/Febcc_gs_xc_LDA_gc_False_kd_24_nbd_18_nbc_-4_pw_1000_oc_0.001_kf_24_nbf_18_a_2.867_mm_2.21/'
    gs += 'Febcc_ground_state.gpw'
    
    scrf = SpinChargeResponseFunction(gs, frequencies=np.linspace(0., 0.5, 201), ecut=1000, 
                                      nblocks=12,
                                      hilbert=False, disable_point_group=True, disable_time_reversal=True)
    
    
    addG, addw, totmem = check_scrf_mem(scrf, [0,0,0], 'spin')
