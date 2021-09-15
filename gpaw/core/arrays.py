from __future__ import annotations
import numpy as np
from gpaw.mpi import MPIComm, serial_comm
from gpaw.core.layout import Layout
from gpaw.matrix import Matrix


class DistributedArrays:
    def __init__(self,
                 layout: Layout,
                 shape: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None,
                 dtype=None,
                 layout_last: bool = True):
        self.layout = layout
        self.comm = comm
        self.layout_last = layout_last

        self.shape = shape if isinstance(shape, tuple) else (shape,)

        if self.shape:
            myshape0 = (self.shape[0] + comm.size - 1) // comm.size
            self.myshape = (myshape0,) + self.shape[1:]
        else:
            self.myshape = ()

        if layout_last:
            fullshape = self.myshape + layout.myshape
        else:
            fullshape = layout.myshape + self.myshape

        dtype = dtype or layout.dtype

        if data is not None:
            assert data.shape == fullshape
            assert data.dtype == dtype
        else:
            data = np.empty(fullshape, dtype)

        self.data = data

    def matrix_view(self):
        if self.layout_last:
            shape = (np.prod(self.shape), np.prod(self.layout.shape))
            myshape = (np.prod(self.myshape), np.prod(self.layout.myshape))
            dist = (self.comm, -1, 1)
        else:
            shape = (np.prod(self.layout.shape), np.prod(self.shape))
            myshape = (np.prod(self.layout.myshape), np.prod(self.myshape))
            dist = (self.comm, 1, -1)
        return Matrix(*shape,
                      data=self.data.reshape(myshape),
                      dist=dist)

    def matrix_elements(self, other, *, symmetric=None, function=None,
                        out=None, add_to_out=False, domain_sum=True):
        assert out is not None
        assert not domain_sum
        if symmetric is None:
            symmetric = self is other
        if function:
            other = function(other)
        M1 = self.matrix_view()
        M2 = other.matrix_view()
        if self.layout_last and other.layout_last:
            assert not add_to_out
            M1.multiply(M2, opb='C', alpha=self.layout.dv, symmetric=symmetric,
                        out=out)
            out.complex_conjugate()
        elif not self.layout_last and not other.layout_last:
            assert add_to_out
            M1.multiply(M2, opa='C', symmetric=symmetric, out=out, beta=1.0)
        else:
            1 / 0
        #operate_and_multiply(self, self.layout.dv, out, function, ...)
        return out

    def __iadd__(self, other):
        other.acfs.add_to(self, other.coefs)
        return self


def operate_and_multiply(psit1, dv, out, operator, psit2):
    if psit1.comm:
        if psit2 is not None:
            assert psit2.comm is psit1.comm
        if psit1.comm.size > 1:
            out.comm = psit1.comm
            out.state = 'a sum is needed'

    comm = psit1.matrix.dist.comm
    N = len(psit1)
    n = (N + comm.size - 1) // comm.size
    mynbands = len(psit1.matrix.array)

    buf1 = psit1.new(nbands=n, dist=None)
    buf2 = psit1.new(nbands=n, dist=None)
    half = comm.size // 2
    psit = psit1.view(0, mynbands)
    if psit2 is not None:
        psit2 = psit2.view(0, mynbands)

    for r in range(half + 1):
        rrequest = None
        srequest = None

        if r < half:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            skip = (comm.size % 2 == 0 and r == half - 1)
            n1 = min(rrank * n, N)
            n2 = min(n1 + n, N)
            if not (skip and comm.rank < half) and n2 > n1:
                rrequest = comm.receive(buf1.array[:n2 - n1], rrank, 11, False)
            if not (skip and comm.rank >= half) and len(psit1.array) > 0:
                srequest = comm.send(psit1.array, srank, 11, False)

        if r == 0:
            if operator:
                operator(psit1.array, psit2.array)
            else:
                psit2 = psit

        if not (comm.size % 2 == 0 and r == half and comm.rank < half):
            m12 = psit2.matrix_elements(psit, symmetric=(r == 0), cc=True,
                                        serial=True)
            n1 = min(((comm.rank - r) % comm.size) * n, N)
            n2 = min(n1 + n, N)
            out.array[:, n1:n2] = m12.array[:, :n2 - n1]

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        psit = buf1
        buf1, buf2 = buf2, buf1

    requests = []
    blocks = []
    nrows = (comm.size - 1) // 2
    for row in range(nrows):
        for column in range(comm.size - nrows + row, comm.size):
            if comm.rank == row:
                n1 = min(column * n, N)
                n2 = min(n1 + n, N)
                if mynbands > 0 and n2 > n1:
                    requests.append(
                        comm.send(out.array[:, n1:n2].T.conj().copy(),
                                  column, 12, False))
            elif comm.rank == column:
                n1 = min(row * n, N)
                n2 = min(n1 + n, N)
                if mynbands > 0 and n2 > n1:
                    block = np.empty((mynbands, n2 - n1), out.dtype)
                    blocks.append((n1, n2, block))
                    requests.append(comm.receive(block, row, 12, False))

    comm.waitall(requests)
    for n1, n2, block in blocks:
        out.array[:, n1:n2] = block
