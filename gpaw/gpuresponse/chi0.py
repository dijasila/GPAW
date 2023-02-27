from gpaw.mpi import world
#import cupy as xp
import numpy as xp
import numpy as np

class WGGData:
    def __init__(self, wggdesc, *, xp, dtype):
        self.wggdesc = wggdesc
        self.xp = xp
        self.data_wgg = xp.zeros(wggdesc.localshape, dtype=dtype)

class StreamModule:
    def __init__(self):
        pass

    def Stream(self, non_blocking=False):
        return self

    def synchronize(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        pass

class Runtime():
    def deviceSynchronize(self):
        pass

class Cuda:
    def __init__(self):
        self.stream = StreamModule()
        self.runtime = Runtime()

xp.cuda = Cuda()


class WGGDescriptor:
    def __init__(self, shape, *, dist_c, comm):
        print(comm.size,'size')
        # Global shape of the WGG-data
        self.shape = np.array(shape)
        # Distribution, for example (2,3,3)
        self.dist_c = np.array(dist_c)
        print(comm.size, self.dist_c.prod())
        assert self.dist_c.prod() == comm.size
        self.comm = comm

        # Parallel stride
        self.pstride_c = (self.shape + self.dist_c - 1) // self.dist_c

        # All ranks, the way they are distributed is just a decision
        ranks_wGG = np.ravel_multi_index(np.indices(self.dist_c), self.dist_c)
        # Get my ranks in each communicator
        wgg_c = np.unravel_index(comm.rank, self.dist_c)
        print(wgg_c)
        wgg_c = np.array(wgg_c)
        Wrank, G1rank, G2rank = wgg_c

        # Get global positions of this array
        self.beg_c = np.minimum(self.pstride_c * wgg_c, self.shape)
        self.end_c = np.minimum(self.pstride_c * (wgg_c + 1), self.shape)
        self.localshape = self.end_c - self.beg_c

        # Create axial communicators
        self.Wcomm = comm.new_communicator(list(ranks_wGG[:, G1rank, G2rank]))
        self.G1comm = comm.new_communicator(list(ranks_wGG[Wrank, :, G2rank]))
        self.G2comm = comm.new_communicator(list(ranks_wGG[Wrank, G1rank, :]))
 
        # Create w=x plane communicator
        self.GGcomm = comm.new_communicator(list(ranks_wGG[Wrank, :, :].ravel()))

        # Ensure that we know our own rank
        assert self.Wcomm.rank == Wrank
        assert self.G1comm.rank == G1rank
        assert self.G2comm.rank == G2rank

    @property
    def comms(self):
        return self.Wcomm, self.G1comm, self.G2comm, self.GGcomm

    def zeros(self, dtype=complex, *, xp):
        return WGGData(self, dtype=dtype, xp=xp)

def chi0_brute_force_update(localn_mG, chi0_wGG):
    # At w=fixed plane:
    # Start. I have my local n_mG, with full G.
    # Other ranks in my plane have their local n_mG's, with full G.
    # What each core needs, is n_mG[:,x] and n_mG[:,y] according to their rank
    print('enter chi0')
    desc = chi0_wGG.wggdesc
    xp = chi0_wGG.xp
    comm = desc.comm
    Wcomm, G1comm, G2comm, GGcomm = desc.comms
    stream = xp.cuda.stream.Stream(non_blocking=True)
    with stream:
        Wbeg,  Wend  = desc.beg_c[0], desc.end_c[0]
        G1beg, G1end = desc.beg_c[1], desc.end_c[1]
        G2beg, G2end = desc.beg_c[2], desc.end_c[2]


        # Each core will have two slices, one for G1 and one for G2.
        # The number of pair densities will be local pairdensities * GGcomm.size
        mstride = len(localn_mG)
        n1_Mg = xp.empty((G1comm.size * mstride, G1end-G1beg), dtype=complex)
        n2_Mg = xp.empty((G2comm.size * mstride, G2end-G2beg), dtype=complex)
        n1_Mg[G1comm.rank*mstride:(G1comm.rank+1)*mstride, :] = localn_mG[:, G1beg:G1end]
        n2_Mg[G2comm.rank*mstride:(G2comm.rank+1)*mstride, :] = localn_mG[:, G2beg:G2end].conj() 
        stream.synchronize()
        # Step 1a: Send G1 slices to their respective ranks with G1comm
        G1sends = []
        for rank in range(G1comm.size):
            if rank != G1comm.rank: 
                n1_mg = xp.ascontiguousarray(localn_mG[:, rank*desc.pstride_c[1]:(rank + 1)*desc.pstride_c[1]])
                #xp.cuda.runtime.deviceSynchronize()
                stream.synchronize()
                G1sends.append(G1comm.send(n1_mg, rank, tag=1, block=False))
        # Step 2a: Receive G1 slices from different communicators 
        G1receives = []
        for rank in range(G1comm.size):
            if rank != G1comm.rank:
                G1receives.append(G1comm.receive(n1_Mg[rank*mstride:(rank+1)*mstride], rank, 1, block=False))
        print('rank', comm.rank, 'A', flush=True)
        if len(G1receives):
            G1comm.waitall(G1receives)
        print('rank', comm.rank, 'B', flush=True)
        if len(G1sends):
            G1comm.waitall(G1sends)
        # Step 1b: Send G2 slices to their respective ranks with G2comm
        G2sends = []
        for rank in range(G2comm.size):
            if rank != G2comm.rank: 
                n2_mg = xp.ascontiguousarray(localn_mG[:, rank*desc.pstride_c[2]:(rank + 1)*desc.pstride_c[2]].conj())
                #xp.cuda.runtime.deviceSynchronize()
                stream.synchronize()
                G2sends.append(G2comm.send(n2_mg, rank, tag=2, block=False))
                print('G2comm', G2comm.rank, 'sending to', rank)
        
    
        G2receives = []
        # Step 2b: Receive G2 slices from different communicators 
        for rank in range(G2comm.size):
            if rank != G2comm.rank:
                print('G2comm rank', G2comm.rank, 'receiving from', rank)
                G2receives.append(G2comm.receive(n2_Mg[rank*mstride:(rank+1)*mstride], rank, 2, block=False))
        print('rank', comm.rank, 'C', flush=True)
        if len(G2receives):
            G2comm.waitall(G2receives)
        print('rank', comm.rank, 'D', flush=True)
        if len(G2sends):
            G2comm.waitall(G2sends)
        print('rank', comm.rank, 'E', flush=True)
        stream.synchronize()
        comm.barrier()
        n1_Ng = xp.empty((GGcomm.size * mstride, G1end-G1beg), dtype=complex)
        n2_Ng = xp.empty((GGcomm.size * mstride, G2end-G2beg), dtype=complex)
        stream.synchronize()

        # Step 2a: Allgather G1 slices with G2-comm, so that each rank has their G1 slices
        G2comm.all_gather(n1_Mg, n1_Ng)
        # Step 2b: Allgather G2 slices with G1-comm, so that each rank has their G2 slices
        G1comm.all_gather(n2_Mg, n2_Ng)
        
        #n2_Mg = xp.transpose(n2_Mg.reshape((2, 2, -1, G2end-G2beg)), axes=(1,0,2,3)).ravel().reshape((-1, G2end-G2beg))
        #n1_Ng = xp.transpose(n1_Ng.reshape((G2comm.size, G1comm.size, -1, G1end-G1beg)), axes=(1,0,2,3)).ravel().reshape((-1, G1end-G1beg))
        n2_Ng = xp.transpose(n2_Ng.reshape((G2comm.size, G1comm.size, -1, G2end-G2beg)), axes=(1,0,2,3)).ravel().reshape((-1, G2end-G2beg))
        #n1_Mg = xp.transpose(n1_Mg.reshape((2, 2, -1, G1end-G1beg)), axes=(1,0,2,3)).copy().reshape((-1, G1end-G1beg))
        #GGsends = []
        #if GGcomm.rank != 0:
        #    n1_mG = xp.ascontiguousarray(localn_mG[:, G1beg:G1end])
        #    GGsends.append(GGcomm.send(n1_mG, 0, tag=1, block=False))
        #
        #G2master = GGcomm.size // 2
        #if G2comm.rank != G2master:
        #    n2_mG = xp.ascontiguousarray(localn_mG[:, G2beg:G2end].conj())
        #    GGsends.append(GGcomm.send(n2_mG, G2master, tag=2, block=False))

        if Wcomm.size > 1:
            newn1_Ng = xp.empty((GGcomm.size * mstride, G1end-G1beg), dtype=complex)
            newn2_Ng = xp.empty((GGcomm.size * mstride, G2end-G2beg), dtype=complex)

        #GGreceives = []
        #if GGcomm.rank == 0:
        #    print(n1_Mg.shape, mstride, localn_mG.shape)
        #    n1_Mg[:mstride, :] = localn_mG[:, G1beg:G1end]
        #    for rank in range(1, GGcomm.size): 
        #        GGreceives.append(GGcomm.receive(n1_Mg[mstride * rank:(mstride + 1)*rank, :], rank, tag=1, block=False))

        #if GGcomm.rank == G2master:
        #    n2_Mg[G2master * mstride:(G2master + 1) * mstride, :] = localn_mG[:, G2beg:G2end].conj()
        #    for rank in range(GGcomm.size): 
        #        if rank != G2master:
        #            GGreceives.append(GGcomm.receive(n2_Mg[mstride * rank:(mstride + 1)*rank, :], rank, tag=2, block=False))
        #print('doing waitalls rank', world.rank, flush=True) 
        #G1comm.waitall(GGreceives)
        #print('A doing waitalls rank', world.rank, flush=True) 
        #G2comm.waitall(GGsends)
        #print('done doing waitalls rank', world.rank, flush=True) 
        #G1comm.broadcast(n1_Mg, 0)
        #G2comm.broadcast(n2_Mg, G2master)
        w_Nw = np.ones((mstride * GGcomm.size, Wend-Wbeg), dtype=complex)
        # Start rolling to right
        for witer in range(Wcomm.size):
            stream.synchronize()
            # Initialize send
            receives = []
            sends = []
            if witer != Wcomm.size - 1:
                # Send to right
                sends.append(Wcomm.send(n1_Ng, (Wcomm.rank+1) % Wcomm.size, tag=11, block=False))
                sends.append(Wcomm.send(n2_Ng, (Wcomm.rank+1) % Wcomm.size, tag=12, block=False))
                # Receive from left
                receives.append(Wcomm.receive(newn1_Ng, (Wcomm.rank-1) % Wcomm.size, tag=11, block=False))
                receives.append(Wcomm.receive(newn2_Ng, (Wcomm.rank-1) % Wcomm.size, tag=12, block=False))

            # Work
            print(n1_Mg.shape, w_Nw.shape, n2_Mg.shape)
            print('Summing data ', n1_Ng, w_Nw, n2_Ng)
            stream.synchronize()
            chi0_wGG.data_wgg += xp.einsum('NA,NW,NB->WAB', n1_Ng, w_Nw, n2_Ng, optimize=True)
            stream.synchronize()
            print('chi0_wGG at this stage', chi0_wGG.data_wgg, world.rank, flush=True)

            # Finalize receive for the next iteration
            if len(receives):
                print('waitall receives rank', comm.rank, flush=True)
                Wcomm.waitall(receives)
                print('waitall receives finished rank', comm.rank, flush=True)
            if len(sends):
                print('waitall sends rank', comm.rank, flush=True)
                Wcomm.waitall(sends)
                print('waitall sendsends rank', comm.rank, flush=True)
            # Update loop variables
            if Wcomm.size > 1:
                n1_Ng = newn1_Ng
                n2_Ng = newn2_Ng
        print('exit chi0')

 
from gpaw.gpu import setup
setup()
print('Hello', world.rank, world.size)
NW, NG = 5, 6000
NM = 3000
serial_comm = world.new_communicator([world.rank])
n_mG = xp.zeros((NM, NG), dtype=complex) # + 1j*xp.random.rand(NM, NG)
n_mG[:] = xp.random.rand(NM, NG) + 1j*xp.random.rand(NM, NG)
#for m in range(120):
#    n_mG[m,m] = (m+1)
#    #n_mG[m,m+1] = (m+1) + (m+1)*1j
xp.cuda.runtime.deviceSynchronize()
world.broadcast(n_mG, 0)
xp.cuda.runtime.deviceSynchronize()
if 1:
    wggdesc = WGGDescriptor((NW, NG, NG), dist_c=(1,1,1), comm=serial_comm)
    chi0s_wGG = wggdesc.zeros(xp=xp)
    import time
    world.barrier()
    start = time.time()
    chi0_brute_force_update(n_mG, chi0s_wGG)
    end = time.time()
    print(end-start, 'serial')
print(chi0s_wGG.data_wgg)

xp.cuda.runtime.deviceSynchronize()
world.barrier()
wggdesc = WGGDescriptor((NW, NG, NG), dist_c=(1,2,2), comm=world)
chi0_wGG = wggdesc.zeros(xp=xp)
pstride = NM // world.size
assert NM % world.size == 0 # TODO: Remove this restriction by allocating empties
n_mG = n_mG[world.rank * pstride:(world.rank+1)*pstride] 
#n_mG = xp.ones((NM, NG), dtype=complex)
import time
xp.cuda.runtime.deviceSynchronize()
start = time.time()
chi0_brute_force_update(n_mG, chi0_wGG)
end = time.time()
print(end-start, 'parallel')
flops = 4 * NW * NM * NG**2
print(flops / 1024**3 / (end-start), 'parallel Gflops')

beg_c = chi0_wGG.wggdesc.beg_c
end_c = chi0_wGG.wggdesc.end_c
if not np.allclose(chi0s_wGG.data_wgg[beg_c[0]:end_c[0], beg_c[1]:end_c[1], beg_c[2]:end_c[2]], chi0_wGG.data_wgg):
    print('Mismatch of data')
    A = chi0s_wGG.data_wgg[beg_c[0]:end_c[0], beg_c[1]:end_c[1], beg_c[2]:end_c[2]]
    B = chi0_wGG.data_wgg
    for x in range(beg_c[0], end_c[0]):
        for y in range(beg_c[1], end_c[1]):
            for z in range(beg_c[2], end_c[2]):
                a = A[x-beg_c[0], y-beg_c[1], z-beg_c[2]]
                b = B[x-beg_c[0], y-beg_c[1], z-beg_c[2]]
                if a != b:
                    print(x,y,z,'rank=',world.rank, 'ref=', a, 'par=', b)
