import numpy as np
class ParallelArrayDistribution:
    """
        Provides a general distribution for arrays
    """
    def __init__(self, shape, dist_n=None, *, comm):
        pass
        #if dist_n is None:
             
class ParallelArrayDescriptor:
    def __init__(self, shape, verbose=True, *, comm, xp):
        self.shape = np.array(shape)
        self.comm = comm
        self.xp = xp
        #self.dist = ParallelArrayDistribution(shape, dist_n=dist_n, comm=comm)
        if verbose:
            print('Parallel array descriptor', shape, comm.rank, xp.__name__)

    def new_array(self):
        return ParallelArray(self)

    def create_from_blocks(self):
        array = self.new_array()
        return ParallelArrayBuildFromBlocksContextManager(array)

    def create_from_global(self, A, dist_n):
        """
            Creates a blocked array from a global array. Note: all ranks must
            get exactly the same array.
        """
        dist_n = np.asarray(dist_n)
        assert A.shape == tuple(self.shape), (A.shape, self.shape)
        assert self.comm.size == dist_n.prod()
        pstride_n = (self.shape + dist_n - 1) // dist_n
        coord_n = np.asarray(np.unravel_index(self.comm.rank, dist_n))
        #print('coord_n=',coord_n, self.comm.rank)
        with self.create_from_blocks() as array:
            start_n = np.minimum(coord_n * pstride_n, self.shape)
            end_n = np.minimum((coord_n + 1) * pstride_n, self.shape)
            slices = tuple([slice(start, end) for start, end in zip(start_n, end_n)])
            array[slices] = A[slices]
        return array
         

class ParallelArrayReadOnlyContextManager:
    def __init__(self, array, slice_n):
        self.array = array
        self.slice_n = slice_n
        self.local_X = None

    def __enter__(self):
        # Do MPI communication to fill local_X with the required slice
        self.local_X = np.zeros(...)
        return self.file_obj

    def __exit__(self, type, value, traceback):
        print(type, value, traceback)

class ParallelArrayBuildFromBlocksContextManager:
    def __init__(self, array):

        self.array = array

    def __enter__(self):
        array = self.array
        assert array.data is None
        array.data = ParallelArrayBlockedData()
        array.context = self
        return array

    def __setitem__(self, item, value):
        self.array.data.__setitem__(item, value)

    def __exit__(self, type, value, traceback):
        self.array.data.collect_blocks(self.array.pdd.comm)
        assert self.array.context == self
        self.array.context = None
        print(type, value, traceback)

class ParallelArrayReadWriteContextManager(ParallelArrayReadOnlyContextManager):
    def __init__(self, array, slice_n):
        self.array = array
        self.slice_n = slice_n

    def __exit__(self, type, value, traceback):
        ParallelArrayReadOnlyContextManager.__exit__(self, type, value, traceback)
        # Do MPI communication to write back self.local_X to proper place
        print(type, value, traceback)

class ParallelArrayData:
    pass

from gpaw.mpi import broadcast

class ParallelArrayBlockedData(ParallelArrayData):
    def __init__(self):
        self.myblocks = []
        self.all_items = []

    def __setitem__(self, item, value):
        self.myblocks.append((item, value))

    def collect_blocks(self, comm):
        items = []
        myitems = [(comm.rank, localindex, item) for localindex, (item, value) in enumerate(self.myblocks)]
        for rank in range(comm.size):
            items.extend(broadcast(myitems if rank == comm.rank else None, root=rank, comm=comm))
        self.all_items = items
        #print(comm.rank, 'collected', self.all_items)

    def fill(self, number):
        # Terrible consequences, if blocks are not defined
        for item, value in self.myblocks:
            value[:] = number 
   
    def print(self, pdd):
        for item, value in self.myblocks: 
            print(f'rank={pdd.comm.rank}', item, value.shape)


class ParallelArray:
    def __init__(self, pdd):
        self.pdd = pdd
        self.context = None
        self.data = None
        self.shape = pdd.shape

    def __setitem__(self, item, value):
        self.context.__setitem__(item, value)

    def fill(self, value):
        self.data.fill(value)

    def print(self):
        self.data.print(self.pdd)

    def access(self, slice_n, mode='r'):
        if mode == 'r':
            return ParallelArrayReadOnlyContextManager(self, slice_n)
        elif mode == 'rw':
            return ParallelArrayReadWriteContextManager(self, slice_n)
        raise ValueError(f'Unknown access mode: {mode}')

    def collect(self):
        comm = self.pdd.comm
        array = np.zeros(self.shape)
        for item, value in self.data.myblocks:
            b1, e1, b2, e2 = item[0].start, item[0].stop, item[1].start, item[1].stop
            array[b1:e1, b2:e2] += value
        comm.sum(array)
        return array    


class NDSlice:
    def __init__(self, item=None, beg_n=None, end_n=None):
       if beg_n is None:
           self.beg_n = np.array([s.start for s in item])
       if end_n is None:
           self.end_n = np.array([s.stop for s in item])

    def intersect(self, other):
        beg_n = np.maximum(self.beg_n, other.beg_n)
        end_n = np.minimum(self.beg_n, other.beg_n)
        if np.any(end_n-beg_n<=0):
            return None
        return NDSlice(beg_n=beg_n, end_n=end_n)
       
    def __getitem__(self, item):
        if isinstance(item, slice):
            return NDSlice(beg_n=self.beg_n[item], end_n=self.end_n[item])
        raise ValueError(f'Do not know how to handle: {item}.')

    def __add__(self, other):
        return NDSlice(beg_n=np.concatenate((self.beg_n, other.beg_n)),
                       end_n=np.concatenate((self.end_n, other.end_n)))

def gemm(pA, pB, pC):
    # Assertions for matrix compatibility
    assert len(pA.shape) == 2
    assert len(pB.shape) == 2
    assert len(pB.shape) == 2
    assert pA.shape[1] == pB.shape[0]
    assert pA.shape[0] == pC.shape[0]
    assert pB.shape[1] == pC.shape[1]
    assert pA.pdd.comm == pB.pdd.comm
    assert pA.pdd.comm == pC.pdd.comm
    assert pB.pdd.comm == pC.pdd.comm
    comm = pA.pdd.comm
    def generate_work():
        for rankA, localindexA, itemA in pA.data.all_items:
             sA = NDSlice(itemA)
             for rankB, localindexB, itemB in pB.data.all_items:
                 sB = NDSlice(itemB)
                 # Intersect the contracted dimension
                 beg = np.maximum(sA.beg_n[1], sB.beg_n[0])
                 end = np.minimum(sA.end_n[1], sB.end_n[0])
                 if end-beg <= 0:
                     continue
                 for rankC, localindexC, itemC in pC.data.all_items:
                     sC = NDSlice(itemC)
                     beg1 = np.maximum(sA.beg_n[0], sC.beg_n[0])
                     end1 = np.minimum(sA.end_n[0], sC.end_n[0])
                     if end1 - beg1 <= 0:
                          continue
                     beg2 = np.maximum(sB.beg_n[1], sC.beg_n[1])
                     end2 = np.minimum(sB.end_n[1], sC.end_n[1])
                     if end2 - beg2 <= 0:
                          continue
                     yield (rankA, localindexA, rankB, localindexB, rankC, localindexC, beg1, end1, beg, end, beg2, end2)
    for rankA, localindexA, rankB, localindexB, rankC, localindexC, beg1, end1, beg, end, beg2, end2 in generate_work():
        if comm.rank == rankA:
            if rankA != rankC:
                item, value = pA.data.myblocks[localindexA]
                b1, e1, b2, e2 = item[0].start, item[0].stop, item[1].start, item[1].stop
                A = value[beg1-b1:end1-b1, beg-b2:end-b2].copy() # Copy to be contiguous
                comm.send(A, rankC)
        if comm.rank == rankB:
            if rankB != rankC:
                item, value = pB.data.myblocks[localindexB]
                b1, e1, b2, e2 = item[0].start, item[0].stop, item[1].start, item[1].stop
                A = value[beg-b1:end-b1, beg2-b2:end2-b2].copy() # Copy to be contiguous
                comm.send(A, rankC)
        if comm.rank == rankC:
            if rankA != rankC:
                A = np.empty((end1-beg1, end-beg))
                comm.receive(A, rankA)
            else:
                item, value = pA.data.myblocks[localindexA]
                b1, e1, b2, e2 = item[0].start, item[0].stop, item[1].start, item[1].stop
                A = value[beg1-b1:end1-b1, beg-b2:end-b2]
            if rankB != rankC:
                B = np.empty((end-beg, end2-beg2))
                comm.receive(B, rankB)
            else:
                item, value = pB.data.myblocks[localindexB]
                b1, e1, b2, e2 = item[0].start, item[0].stop, item[1].start, item[1].stop
                B = value[beg-b1:end-b1, beg2-b2:end2-b2]

            C = A @ B
            item, value = pC.data.myblocks[localindexC]
            b1, e1, b2, e2 = item[0].start, item[0].stop, item[1].start, item[1].stop
            #vslice = value[beg1-b1:end1-b1, beg2-b2:end2-b2]
            #print(f'C={C.shape} value={value.shape} value_slice={vslice.shape}', beg1, end1, beg, end, beg2, end2, 'C', b1, e1, b2, e2)
            value[beg1-b1:end1-b1, beg2-b2:end2-b2] += C
            


