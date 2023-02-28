from gpaw.gpu import cupy as cp
import numpy as np


class CuPyMPI:
    """Quick'n'dirty wrapper to make things work without a GPU-aware MPI."""
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size

    def sum(self, array, root=-1):
        if isinstance(array, (float, int)):
            return self.comm.sum(array, root)
        if isinstance(array, np.ndarray):
            self.comm.sum(array, root)
            return
        a = array.get()
        self.comm.sum(a, root)
        array[:] = cp.asarray(a)

    def max(self, array):
        self.comm.max(array)

    def all_gather(self, a, b):
        self.comm.all_gather(a, b)

    def gather(self, a, rank, b):
        if isinstance(a, np.ndarray):
            self.comm.gather(a, rank, b)
        else:
            if rank == self.rank:
                c = np.empty(b.shape, b.dtype)
            else:
                c = None
            self.comm.gather(a.get(), rank, c)
            if rank == self.rank:
                b[:] = cp.asarray(c)

    def scatter(self, fro, to, root=0):
        if isinstance(fro, np.ndarray):
            1 / 0
        b = np.empty(to.shape, to.dtype)
        if self.rank == root:
            a = fro.get()
        else:
            a = None
        self.comm.scatter(a, b, root)
        to[:] = cp.asarray(b)

    def broadcast(self, a, root):
        if isinstance(a, np.ndarray):
            self.comm.broadcast(a, root)
            return
        b = a.get()
        self.comm.broadcast(b, root)
        a[:] = cp.asarray(b)

    def receive(self, a, rank, tag=0):
        if isinstance(a, np.ndarray):
            return self.comm.receive(a, rank, tag)
        b = np.empty(a.shape, a.dtype)
        self.comm.receive(b, rank, tag)
        a[:] = cp.asarray(b)

    def ssend(self, a, rank, tag):
        self.comm.send(a.get(), rank, tag)

    def send(self, a, rank, tag=0, block=True):
        if isinstance(a, np.ndarray):
            return self.comm.send(a, rank, tag, block)
        b = a.get()
        request = self.comm.send(b, rank, tag, block)
        if not block:
            return CuPyRequest(request, b)

    def wait(self, request):
        self.comm.wait(request.request)

    def get_c_object(self):
        return self.comm.get_c_object()


class CuPyRequest:
    def __init__(self, request, array):
        self.request = request
        self.array = array
