import mpi4py.MPI as mpi
import numpy as np


class Communicator:
    def __init__(self, comm, parent=None):
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.parent = parent  # XXX check C-object against comm.parent?

    def new_communicator(self, ranks):
        """Create a new MPI communicator for a subset of ranks in a group.
        Must be called with identical arguments by all relevant processes.

        Note that a valid communicator is only returned to the processes
        which are included in the new group; other ranks get None returned.

        Parameters:

        ranks: ndarray (type int)
            List of integers of the ranks to include in the new group.
            Note that these ranks correspond to indices in the current
            group whereas the rank attribute in the new communicators
            correspond to their respective index in the subset.

        """
        comm = self.comm.Create(self.comm.group.Incl(ranks))
        if self.comm.rank in ranks:
            return Communicator(comm, parent=self)
        else:
            # This cpu is not in the new communicator:
            return None

    def sum(self, a, root=-1, op=mpi.SUM):
        if isinstance(a, (int, float, complex)):
            if root == -1:
                return self.comm.allreduce(a, op=op)
            else:
                return self.comm.reduce(a, root=root, op=op)
        else:
            if root == -1:
                self.comm.Allreduce(a, a, op=op)
            else:
                self.comm.Reduce(a, a, root=root, op=op)

    def product(self, a, root=-1):
        return self.sum(a, root, mpi.PROD)

    def max(self, a, root=-1):
        return self.sum(a, root, mpi.MAX)

    def min(self, a, root=-1):
        return self.sum(a, root, mpi.MIN)

    def scatter(self, a, b, root):
        self.comm.Scatter(a, b, root)

    def alltoallv(self, sbuffer, scounts, sdispls, rbuffer, rcounts, rdispls):
        """All-to-all in a group.

        Parameters:

        sbuffer: ndarray
            Source of the data to distribute, i.e., send buffers on all ranks
        scounts: ndarray
            Integer array equal to the group size specifying the number of
            elements to send to each processor
        sdispls: ndarray
            Integer array (of length group size). Entry j specifies the
            displacement (relative to sendbuf from which to take the
            outgoing data destined for process j)
        rbuffer: ndarray
            Destination of the distributed data, i.e., local receive buffer.
        rcounts: ndarray
            Integer array equal to the group size specifying the maximum
            number of elements that can be received from each processor.
        rdispls:
            Integer array (of length group size). Entry i specifies the
            displacement (relative to recvbuf at which to place the incoming
            data from process i
        """
        assert sbuffer.flags.contiguous
        assert scounts.flags.contiguous
        assert sdispls.flags.contiguous
        assert rbuffer.flags.contiguous
        assert rcounts.flags.contiguous
        assert rdispls.flags.contiguous
        assert sbuffer.dtype == rbuffer.dtype

        for arr in [scounts, sdispls, rcounts, rdispls]:
            assert arr.dtype == np.int, arr.dtype
            assert len(arr) == self.size

        assert np.all(0 <= sdispls)
        assert np.all(0 <= rdispls)
        assert np.all(sdispls + scounts <= sbuffer.size)
        assert np.all(rdispls + rcounts <= rbuffer.size)
        self.comm.alltoallv(sbuffer, scounts, sdispls,
                            rbuffer, rcounts, rdispls)

    def all_gather(self, a, b):
        """Gather data from all ranks onto all processes in a group.

        Parameters:

        a: ndarray
            Source of the data to gather, i.e. send buffer of this rank.
        b: ndarray
            Destination of the distributed data, i.e. receive buffer.
            The size of this array must match the size of the distributed
            source arrays multiplied by the number of process in the group.

        Example::

          # All ranks have parts of interesting data. Gather on all ranks.
          mydata = np.random.normal(size=N)
          data = np.empty(N*comm.size, dtype=float)
          comm.all_gather(mydata, data)

          # .. which is equivalent to ..

          if comm.rank == 0:
              # Insert my part directly
              data[0:N] = mydata
              # Gather parts from the slaves
              buf = np.empty(N, dtype=float)
              for rank in range(1, comm.size):
                  comm.receive(buf, rank, tag=123)
                  data[rank*N:(rank+1)*N] = buf
          else:
              # Send to the master
              comm.send(mydata, 0, tag=123)
          # Broadcast from master to all slaves
          comm.broadcast(data, 0)

        """
        assert a.flags.contiguous
        assert b.flags.contiguous
        assert b.dtype == a.dtype
        assert (b.shape[0] == self.size and a.shape == b.shape[1:] or
                a.size * self.size == b.size)
        self.comm.all_gather(a, b)

    def gather(self, a, root, b=None):
        """Gather data from all ranks onto a single process in a group.

        Parameters:

        a: ndarray
            Source of the data to gather, i.e. send buffer of this rank.
        root: int
            Rank of the root process, on which the data is to be gathered.
        b: ndarray (ignored on all ranks different from root; default None)
            Destination of the distributed data, i.e. root's receive buffer.
            The size of this array must match the size of the distributed
            source arrays multiplied by the number of process in the group.

        The reverse operation is ``scatter``.

        Example::

          # All ranks have parts of interesting data. Gather it on master.
          mydata = np.random.normal(size=N)
          if comm.rank == 0:
              data = np.empty(N*comm.size, dtype=float)
          else:
              data = None
          comm.gather(mydata, 0, data)

          # .. which is equivalent to ..

          if comm.rank == 0:
              # Extract my part directly
              data[0:N] = mydata
              # Gather parts from the slaves
              buf = np.empty(N, dtype=float)
              for rank in range(1, comm.size):
                  comm.receive(buf, rank, tag=123)
                  data[rank*N:(rank+1)*N] = buf
          else:
              # Send to the master
              comm.send(mydata, 0, tag=123)

        """
        assert a.flags.contiguous
        assert 0 <= root < self.size
        if root == self.rank:
            assert b.flags.contiguous and b.dtype == a.dtype
            assert (b.shape[0] == self.size and a.shape == b.shape[1:] or
                    a.size * self.size == b.size)
            self.comm.gather(a, root, b)
        else:
            assert b is None
            self.comm.gather(a, root)

    def broadcast(self, a, root):
        """Share data from a single process to all ranks in a group.

        Parameters:

        a: ndarray
            Data, i.e. send buffer on root rank, receive buffer elsewhere.
            Note that after the broadcast, all ranks have the same data.
        root: int
            Rank of the root process, from which the data is to be shared.

        Example::

          # All ranks have parts of interesting data. Take a given index.
          mydata[:] = np.random.normal(size=N)

          # Who has the element at global index 13? Everybody needs it!
          index = 13
          root, myindex = divmod(index, N)
          element = np.empty(1, dtype=float)
          if comm.rank == root:
              # This process has the requested element so extract it
              element[:] = mydata[myindex]

          # Broadcast from owner to everyone else
          comm.broadcast(element, root)

          # .. which is equivalent to ..

          if comm.rank == root:
              # We are root so send it to the other ranks
              for rank in range(comm.size):
                  if rank != root:
                      comm.send(element, rank, tag=123)
          else:
              # We don't have it so receive from root
              comm.receive(element, root, tag=123)

        """
        assert 0 <= root < self.size
        assert is_contiguous(a)
        self.comm.broadcast(a, root)

    def sendreceive(self, a, dest, b, src, sendtag=123, recvtag=123):
        assert 0 <= dest < self.size
        assert dest != self.rank
        assert is_contiguous(a)
        assert 0 <= src < self.size
        assert src != self.rank
        assert is_contiguous(b)
        return self.comm.sendreceive(a, dest, b, src, sendtag, recvtag)

    def send(self, a, dest, tag=123, block=True):
        assert 0 <= dest < self.size
        assert dest != self.rank
        assert is_contiguous(a)
        if not block:
            pass  # assert sys.getrefcount(a) > 3
        return self.comm.send(a, dest, tag, block)

    def ssend(self, a, dest, tag=123):
        assert 0 <= dest < self.size
        assert dest != self.rank
        assert is_contiguous(a)
        return self.comm.ssend(a, dest, tag)

    def receive(self, a, src, tag=123, block=True):
        assert 0 <= src < self.size
        assert src != self.rank
        assert is_contiguous(a)
        return self.comm.receive(a, src, tag, block)

    def test(self, request):
        """Test whether a non-blocking MPI operation has completed. A boolean
        is returned immediately and the request is not modified in any way.

        Parameters:

        request: MPI request
            Request e.g. returned from send/receive when block=False is used.

        """
        return self.comm.test(request)

    def testall(self, requests):
        """Test whether non-blocking MPI operations have completed. A boolean
        is returned immediately but requests may have been deallocated as a
        result, provided they have completed before or during this invokation.

        Parameters:

        request: MPI request
            Request e.g. returned from send/receive when block=False is used.

        """
        return self.comm.testall(requests)  # may deallocate requests!

    def wait(self, request):
        """Wait for a non-blocking MPI operation to complete before returning.

        Parameters:

        request: MPI request
            Request e.g. returned from send/receive when block=False is used.

        """
        self.comm.wait(request)

    def waitall(self, requests):
        """Wait for non-blocking MPI operations to complete before returning.

        Parameters:

        requests: list
            List of MPI requests e.g. aggregated from returned requests of
            multiple send/receive calls where block=False was used.

        """
        self.comm.waitall(requests)

    def abort(self, errcode):
        """Terminate MPI execution environment of all tasks in the group.
        This function only returns in the advent of an error occurring.

        Parameters:

        errcode: int
            Error code to return to the invoking environment.

        """
        return self.comm.abort(errcode)

    def name(self):
        """Return the name of the processor as a string."""
        return self.comm.name()

    def barrier(self):
        """Block execution until all process have reached this point."""
        self.comm.barrier()

    def compare(self, othercomm):
        """Compare communicator to other.

        Returns 'ident' if they are identical, 'congruent' if they are
        copies of each other, 'similar' if they are permutations of
        each other, and otherwise 'unequal'.

        This method corresponds to MPI_Comm_compare."""
        if isinstance(self.comm, SerialCommunicator):
            return self.comm.compare(othercomm.comm)  # argh!
        result = self.comm.compare(othercomm.get_c_object())
        assert result in ['ident', 'congruent', 'similar', 'unequal']
        return result

    def translate_ranks(self, other, ranks):
        """"Translate ranks from communicator to other.

        ranks must be valid on this communicator.  Returns ranks
        on other communicator corresponding to the same processes.
        Ranks that are not defined on the other communicator are
        assigned values of -1.  (In contrast to MPI which would
        assign MPI_UNDEFINED)."""
        assert hasattr(other, 'translate_ranks'), \
            'Excpected communicator, got %s' % other
        assert all(0 <= rank for rank in ranks)
        assert all(rank < self.size for rank in ranks)
        if isinstance(self.comm, SerialCommunicator):
            return self.comm.translate_ranks(other.comm, ranks)  # argh!
        otherranks = self.comm.translate_ranks(other.get_c_object(), ranks)
        assert all(-1 <= rank for rank in otherranks)
        assert ranks.dtype == otherranks.dtype
        return otherranks

    def get_members(self):
        """Return the subset of processes which are members of this MPI group
        in terms of the ranks they are assigned on the parent communicator.
        For the world communicator, this is all integers up to ``size``.

        Example::

          >>> world.rank, world.size
          (3, 4)
          >>> world.get_members()
          array([0, 1, 2, 3])
          >>> comm = world.new_communicator(array([2, 3]))
          >>> comm.rank, comm.size
          (1, 2)
          >>> comm.get_members()
          array([2, 3])
          >>> comm.get_members()[comm.rank] == world.rank
          True

        """
        return self.comm.get_members()

    def get_c_object(self):
        """Return the C-object wrapped by this debug interface.

        Whenever a communicator object is passed to C code, that object
        must be a proper C-object - *not* e.g. this debug wrapper.  For
        this reason.  The C-communicator object has a get_c_object()
        implementation which returns itself; thus, always call
        comm.get_c_object() and pass the resulting object to the C code.
        """
        c_obj = self.comm.get_c_object()
        assert isinstance(c_obj, _gpaw.Communicator)
        return c_obj


serial_comm = Communicator(mpi.COMM_SELF)
