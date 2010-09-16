"""Module for high-level BLACS interface.

Usage
=====

A BLACS grid is a logical grid of processors.  To use BLACS, first
create a BLACS grid.  If comm contains 8 or more ranks, this example
will work::

  from gpaw.mpi import world
  from gpaw.blacs import BlacsGrid
  grid = BlacsGrid(world, 4, 2)

Use the processor grid to create various descriptors for distributed
arrays::

  block_desc = grid.new_descriptor(500, 500, 64, 64)
  local_desc = grid.new_descriptor(500, 500, 500, 500)

The first descriptor describes 500 by 500 arrays distributed amongst
the 8 CPUs of the BLACS grid in blocks of 64 by 64 elements (which is
a sensible block size).  That means each CPU has many blocks located
all over the array::

  print world.rank, block_desc.shape, block_desc.gshape

Here block_desc.shape is the local array shape while gshape is the
global shape.  The local array shape varies a bit on each CPU as the
block distribution may be slightly uneven.

The second descriptor, local_desc, has a block size equal to the
global size of the array, and will therefore only have one block.
This block will then reside on the first CPU -- local_desc therefore
represents non-distributed arrays.  Let us instantiate some arrays::

  H_MM = local_desc.empty()
  
  if world.rank == 0:
      assert H_MM.shape == (500, 500)
      H_MM[:, :] = calculate_hamiltonian_or_something()
  else:
      assert H_MM.shape[0] == 0 or H_MM.shape[1] == 0

  H_mm = block_desc.empty()
  print H_mm.shape # many elements on all CPUs

We can then redistribute the local H_MM into H_mm::

  from gpaw.blacs import Redistributor
  redistributor = Redistributor(world, local_desc, block_desc)
  redistributor.redistribute(H_MM, H_mm)
  
Now we can run parallel linear algebra on H_mm.  This will diagonalize
H_mm, place the eigenvectors in C_mm and the eigenvalues globally in
eps_M::

  eps_M = np.empty(500)
  C_mm = block_desc.empty()
  block_desc.diagonalize_ex(H_mm, C_mm, eps_M)

We can redistribute C_mm back to the master process if we want::

  C_MM = local_desc.empty()
  redistributor2 = Redistributor(world, block_desc, local_desc)
  redistributor2.redistribute(C_mm, C_MM)

If somebody wants to do all this more easily, they will probably write
a function for that.

List of interesting classes
===========================

 * BlacsGrid
 * BlacsDescriptor
 * Redistributor

The other classes in this module are coded specifically for GPAW and
are inconvenient to use otherwise.

The module gpaw.utilities.blacs contains several functions like gemm,
gemv and r2k.  These functions may or may not have appropriate
docstings, and may use Fortran-like variable naming.  Also, either
this module or gpaw.utilities.blacs will be renamed at some point.

"""

import numpy as np

from gpaw.mpi import SerialCommunicator, serial_comm
from gpaw.matrix_descriptor import MatrixDescriptor
from gpaw.utilities import uncamelcase
from gpaw.utilities.blas import gemm, r2k, gemmdot
from gpaw.utilities.lapack import diagonalize, sldiagonalize, \
    general_diagonalize, slgeneral_diagonalize, \
    inverse_cholesky, slinverse_cholesky
from gpaw.utilities.blacs import scalapack_inverse_cholesky, \
    scalapack_diagonalize_ex, scalapack_general_diagonalize_ex, \
    scalapack_diagonalize_dc, scalapack_general_diagonalize_dc, \
    scalapack_diagonalize_mr3, scalapack_general_diagonalize_mr3, \
    pblas_simple_gemm
from gpaw.utilities.timing import nulltimer
from gpaw.utilities.tools import tri2full
import _gpaw


INACTIVE = -1
BLOCK_CYCLIC_2D = 1


class BlacsGrid:
    """Class representing a 2D grid of processors sharing a Blacs context.
        
    A BLACS grid defines a logical M by N ordering of a collection of
    CPUs.  A BLACS grid can be used to create BLACS descriptors.  On
    an npcol by nprow BLACS grid, a matrix is distributed amongst M by
    N CPUs along columns and rows, respectively, while the matrix
    shape and blocking properties are determined by the descriptors.

    Use the method new_descriptor() to create any number of BLACS
    descriptors sharing the same CPU layout.
    
    Most matrix operations require the involved matrices to all be on
    the same BlacsGrid.  Use a Redistributor to redistribute matrices
    from one BLACS grid to another if necessary.

    Parameters::

      * comm:  MPI communicator for CPUs of the BLACS grid or None.  A BLACS
        grid may use all or some of the CPUs of the communicator.
      * nprow:  Number of CPU rows.
      * npcol: Number of CPU columns.
      * order: 'R' or 'C', meaning rows or columns.  I'm not sure what this 
        does, it probably interchanges the meaning of rows and columns. XXX

    Complicated stuff
    -----------------

    It may be useful to know that a BLACS grid is said to be active
    and will evaluate to True on any process where comm is not None
    *and* comm.rank < nprow * npcol.  Otherwise it is considered
    inactive and evaluates to False.  Ranks where a grid is inactive
    never do anything at all.

    BLACS identifies each grid by a unique ID number called the
    context (frequently abbreviated ConTxt).  Grids on inactive ranks
    have context -1."""
    def __init__(self, comm, nprow, npcol, order='R'):
        assert nprow > 0
        assert npcol > 0
        assert len(order) == 1
        assert order in 'CcRr'

        if isinstance(comm, SerialCommunicator):
            raise ValueError('Instance of SerialCommunicator not supported')
        if comm is None: # if and only if rank is not part of the communicator
            context = INACTIVE
        else:
            if nprow * npcol > comm.size:
                raise ValueError('Impossible: %dx%d Blacs grid with %d CPUs'
                                 % (nprow, npcol, comm.size))
            context = _gpaw.new_blacs_context(comm.get_c_object(),
                                              npcol, nprow, order)
            assert (context != INACTIVE) == (comm.rank < nprow * npcol)

        self.mycol, self.myrow = _gpaw.get_blacs_gridinfo(context, nprow,
                                                          npcol)
        
        self.context = context
        self.comm = comm
        self.nprow = nprow
        self.npcol = npcol
        self.ncpus = nprow * npcol
        self.order = order

    def new_descriptor(self, M, N, mb, nb, rsrc=0, csrc=0):
        """Create a new descriptor from this BLACS grid.

        See documentation for BlacsDescriptor.__init__."""
        return BlacsDescriptor(self, M, N, mb, nb, rsrc, csrc)

    def is_active(self):
        """Whether context is active on this rank."""
        return self.context != INACTIVE

    def __nonzero__(self):
        return self.is_active()

    def __str__(self):
        classname = self.__class__.__name__
        template = '%s[comm:size=%d,rank=%d; context=%d; %dx%d]'
        string = template % (classname, self.comm.size, self.comm.rank, 
                             self.context, self.nprow, self.npcol)
        return string
    
    def __del__(self):
        if self.is_active():
            _gpaw.blacs_destroy(self.context)


class BlacsDescriptor(MatrixDescriptor):
    """Class representing a 2D matrix distribution on a blacs grid.

    A BlacsDescriptor represents a particular shape and distribution
    of matrices.  A BlacsDescriptor has a global matrix shape and a
    rank-dependent local matrix shape.  The local shape is not
    necessarily equal on all ranks.

    A numpy array is said to be compatible with a BlacsDescriptor if,
    on all ranks, the shape of the numpy array is equal to the local
    shape of the BlacsDescriptor.  Compatible arrays can be created
    conveniently with the zeros() and empty() methods.

    An array with a global shape of M by N is distributed such that
    each process gets a number of distinct blocks of size mb by nb.
    The blocks on one process generally reside in very different areas
    of the matrix to improve load balance.
    
    The following chart describes how different ranks (there are 4
    ranks in this example, 0 through 3) divide the matrix into blocks.
    This is called 2D block cyclic distribution::

        +--+--+--+--+..+--+
        | 0| 1| 0| 1|..| 1|
        +--+--+--+--+..+--+
        | 2| 3| 2| 3|..| 3|
        +--+--+--+--+..+--+
        | 0| 1| 0| 1|..| 1|
        +--+--+--+--+..+--+
        | 2| 3| 2| 3|..| 3|
        +--+--+--+--+..+--+
        ...................
        ...................
        +--+--+--+--+..+--+
        | 2| 3| 2| 3|..| 3|
        +--+--+--+--+..+--+

    Also refer to:
    http://acts.nersc.gov/scalapack/hands-on/datadist.html

    Parameters:
     * blacsgrid: the BLACS grid of processors to distribute matrices.
     * M: global row count
     * N: global column count
     * mb: number of rows per block
     * nb: number of columns per block
     * rsrc: rank on which the first row is stored
     * csrc: rank on which the first column is stored

    Complicated stuff
    -----------------
    
    If there is trouble with matrix shapes, the below caveats are
    probably the reason.

    Depending on layout, a descriptor may have a local shape of zero
    by N or something similar.  If the row blocksize is 7, the global
    row count is 10, and the blacs grid contains 3 row processes: The
    first process will have 7 rows, the next will have 3, and the last
    will have 0.  The shapes in this case must still be correctly
    given to BLACS functions, which can be confusing.
    
    A blacs descriptor must also give the correct local leading
    dimension (lld), which is the local array size along the
    memory-contiguous direction in the matrix, and thus equal to the
    local column number, *except* when local shape is zero, but the
    implementation probably works.

    """
    def __init__(self, blacsgrid, M, N, mb, nb, rsrc, csrc):
        assert M > 0
        assert N > 0
        assert 1 <= mb <= M
        assert 1 <= nb <= N
        assert 0 <= rsrc < blacsgrid.nprow
        assert 0 <= csrc < blacsgrid.npcol
        
        self.blacsgrid = blacsgrid
        self.M = M # global size 1
        self.N = N # global size 2
        self.mb = mb # block cyclic distr dim 1
        self.nb = nb # and 2.  How many rows or columns are on this processor
        # more info:
        # http://www.netlib.org/scalapack/slug/node75.html
        self.rsrc = rsrc
        self.csrc = csrc
        
        if 1:#blacsgrid.is_active():
            locN, locM = _gpaw.get_blacs_local_shape(self.blacsgrid.context,
                                                     self.N, self.M,
                                                     self.nb, self.mb, 
                                                     self.csrc, self.rsrc)
            self.lld  = max(1, locN) # max 1 is nonsensical, but appears
                                     # to be required by PBLAS
        else:
            locN, locM = 0, 0
            self.lld = 0
        
        MatrixDescriptor.__init__(self, max(0, locM), max(0, locN))
        
        self.active = locM > 0 and locN > 0 # inactive descriptor can
                                            # exist on an active OR
                                            # inactive blacs grid
        
        self.bshape = (self.mb, self.nb) # Shape of one block
        self.gshape = (M, N) # Global shape of array


    def asarray(self):
        """Return a nine-element array representing this descriptor.
        
        In the C/Fortran code, a BLACS descriptor is represented by a
        special array of arcane nature.  The value of asarray() must 
        generally be passed to BLACS functions in the C code."""
        arr = np.array([BLOCK_CYCLIC_2D, self.blacsgrid.context, 
                        self.N, self.M, self.nb, self.mb, self.csrc, self.rsrc,
                        self.lld], np.int32)
        return arr

    def __repr__(self):
        classname = self.__class__.__name__
        template = '%s[context=%d, glob %s, block %s, lld %d, loc %s]'
        string = template % (classname, self.blacsgrid.context,
                             self.gshape,
                             self.bshape, self.lld, self.shape)
        return string

    def diagonalize_dc(self, H_nn, C_nn, eps_N, UL='L'):
        """See documentation in gpaw/utilities/blacs.py."""
        scalapack_diagonalize_dc(self, H_nn, C_nn, eps_N, UL)

    def diagonalize_ex(self, H_nn, C_nn, eps_N, UL='L', iu=None):
        """See documentation in gpaw/utilities/blacs.py."""
        scalapack_diagonalize_ex(self, H_nn, C_nn, eps_N, UL, iu=iu)

    def diagonalize_mr3(self, H_nn, C_nn, eps_N, UL='L', iu=None):
        """See documentation in gpaw/utilities/blacs.py."""
        scalapack_diagonalize_mr3(self, H_nn, C_nn, eps_N, UL, iu=iu)

    def general_diagonalize_dc(self, H_mm, S_mm, C_mm, eps_M,
                               UL='L'):
        """See documentation in gpaw/utilities/blacs.py."""
        scalapack_general_diagonalize_dc(self, H_mm, S_mm, C_mm, eps_M,
                                         UL)

    def general_diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M,
                               UL='L', iu=None):
        """See documentation in gpaw/utilities/blacs.py."""
        scalapack_general_diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M,
                                         UL, iu=iu)

    def general_diagonalize_mr3(self, H_mm, S_mm, C_mm, eps_M,
                                UL='L', iu=None):
        """See documentation in gpaw/utilities/blacs.py."""
        scalapack_general_diagonalize_mr3(self, H_mm, S_mm, C_mm, eps_M,
                                          UL, iu=iu)

    def inverse_cholesky(self, S_nn, UL='L'):
        """See documentation in gpaw/utilities/blacs.py."""
        scalapack_inverse_cholesky(self, S_nn, UL)

    def my_blocks(self, array_mn):
        """Yield the local blocks and their global index limits.
        
        Yields tuples of the form (Mstart, Mstop, Nstart, Nstop, block),
        for each locally stored block of the array.
        """
        if not self.check(array_mn):
            raise ValueError('Bad array shape (%s vs %s)' % (self,
                                                             array_mn.shape))
        
        grid = self.blacsgrid
        mb = self.mb
        nb = self.nb
        myrow = grid.myrow
        mycol = grid.mycol
        nprow = grid.nprow
        npcol = grid.npcol
        M, N = self.gshape

        Mmyblocks = -(-self.shape[0] // mb)
        Nmyblocks = -(-self.shape[1] // nb)
        for Mblock in range(Mmyblocks):
            for Nblock in range(Nmyblocks):
                myMstart = Mblock * mb
                myNstart = Nblock * nb
                Mstart = myrow * mb + Mblock * mb * nprow
                Nstart = mycol * mb + Nblock * nb * npcol
                Mstop = min(Mstart + mb, M)
                Nstop = min(Nstart + nb, N)
                block = array_mn[myMstart:myMstart + mb,
                                 myNstart:myNstart + mb]
                
                yield Mstart, Mstop, Nstart, Nstop, block

    def as_serial(self):
        return self.blacsgrid.new_descriptor(self.M, self.N, self.M, self.N)

    def redistribute(self, otherdesc, src_mn, dst_mn=None):
        if self.blacsgrid != otherdesc.blacsgrid:
            raise ValueError('Cannot redistribute to other BLACS grid.  '
                             'Requires using Redistributor class explicitly')
        if dst_mn is None:
            dst_mn = otherdesc.empty(dtype=src_mn.dtype)
        r = Redistributor(self.blacsgrid.comm, self, otherdesc)
        r.redistribute(src_mn, dst_mn)
        return dst_mn

    def collect_on_master(self, src_mn, dst_mn=None):
        desc = self.as_serial()
        return self.redistribute(desc, src_mn, dst_mn)

    def distribute_from_master(self, src_mn, dst_mn=None):
        desc = self.as_serial()
        return desc.redistribute(self, src_mn, dst_mn)


class Redistributor:
    """Class for redistributing BLACS matrices on different contexts."""
    def __init__(self, supercomm, srcdescriptor, dstdescriptor, uplo='G'):
        """Create redistributor.

        Source and destination descriptors may reside on different
        BLACS grids, but the descriptors should describe arrays with
        the same number of elements.  

        The communicators of the BLACS grid of srcdescriptor as well
        as that of dstdescriptor *must* both be subcommunicators of
        supercomm.

        Allowed values of UPLO are: G for general matrix, U for upper
        triangular and L for lower triangular. The latter two are useful
        for symmetric matrices."""
        self.supercomm = supercomm
        self.supercomm_bg = BlacsGrid(self.supercomm, self.supercomm.size, 1)
        self.srcdescriptor = srcdescriptor
        self.dstdescriptor = dstdescriptor
        assert uplo in ['G', 'U', 'L'] 
        self.uplo = uplo
    
    def redistribute_submatrix(self, src_mn, dst_mn, subM, subN):
        """Redistribute submatrix into other submatrix.  

        A bit more general than redistribute().  See also redistribute()."""
        # self.supercomm must be a supercommunicator of the communicators
        # corresponding to the context of srcmatrix as well as dstmatrix.
        # We should verify this somehow.
        dtype = src_mn.dtype
        assert dtype == dst_mn.dtype
        
        isreal = (dtype == float)
        assert dtype == float or dtype == complex

        # Check to make sure the submatrix of the source
        # matrix will fit into the destination matrix
        # plus standard BLACS matrix checks.
        srcdescriptor = self.srcdescriptor
        dstdescriptor = self.dstdescriptor
        srcdescriptor.checkassert(src_mn)
        dstdescriptor.checkassert(dst_mn)
        assert srcdescriptor.gshape[0] >= subM
        assert srcdescriptor.gshape[1] >= subN
        assert dstdescriptor.gshape[0] >= subM
        assert dstdescriptor.gshape[1] >= subN

        # Switch to Fortran conventions
        uplo = {'U': 'L', 'L': 'U', 'G': 'G'}[self.uplo]
        
        _gpaw.scalapack_redist(srcdescriptor.asarray(), 
                               dstdescriptor.asarray(),
                               src_mn, dst_mn,
                               self.supercomm_bg.context,
                               subN, subM, isreal, uplo)
    
    def redistribute(self, src_mn, dst_mn):
        """Redistribute src_mn to dst_mn.

        src_mn must be compatible with the source descriptor of this 
        redistributor, while dst_mn must be compatible with the 
        destination descriptor.""" 
        subM, subN = self.srcdescriptor.gshape 
        self.redistribute_submatrix(src_mn, dst_mn, subM, subN)


def parallelprint(comm, obj):
    import sys
    for a in range(comm.size):
        if a == comm.rank:
            print 'rank=%d' % a
            print obj
            print
            sys.stdout.flush()
        comm.barrier()

# -------------------------------------------------------------------

# We should probably move everything below into a seperate file...

from gpaw.matrix_descriptor import BandMatrixDescriptor, \
                                   BlacsBandMatrixDescriptor

def get_kohn_sham_layouts(sl, mode, use_blacs, gd, bd, **kwargs):
    """Create Kohn-Sham layouts object."""
    # Not needed for AtomPAW special mode, as usual we just provide whatever
    # happens to make the code not crash
    if not isinstance(mode, str):
        return None #XXX
    name = {'fd': 'BandLayouts', 'lcao': 'OrbitalLayouts'}[mode]
    args = (gd, bd)
    if use_blacs:
        name = 'Blacs' + name
        assert len(sl) == 3
        args += tuple(sl)
    elif sl is not None: #TODO deprecate
        name = 'SL' + name
    ksl = {'BandLayouts':         BandLayouts,
           'BlacsBandLayouts':    BlacsBandLayouts,
           'SLBandLayouts':       OldSLBandLayouts, #TODO deprecate
           'BlacsOrbitalLayouts': BlacsOrbitalLayouts,
           'OrbitalLayouts':      OrbitalLayouts,
           'SLOrbitalLayouts':    OldSLOrbitalLayouts, #TODO deprecate
            }[name](*args, **kwargs)
    if 0: #XXX debug
        print 'USING KSL: %s' % repr(ksl)
    assert isinstance(ksl, KohnShamLayouts)
    assert isinstance(ksl, BlacsLayouts) == use_blacs, (ksl, use_blacs)
    return ksl


class KohnShamLayouts:
    using_blacs = False
    matrix_descriptor_class = None

    def __init__(self, gd, bd, timer=nulltimer):
        assert gd.comm.parent is bd.comm.parent # must have same parent comm
        self.world = bd.comm.parent
        self.gd = gd
        self.bd = bd
        self.timer = timer
        self._kwargs = {'timer': timer}

    def get_keywords(self):
        return self._kwargs.copy() # just a shallow copy...

    def diagonalize(self, *args, **kwargs):
        raise RuntimeError('Virtual member function should not be called.')

    def inverse_cholesky(self, *args, **kwargs):
        raise RuntimeError('Virtual member function should not be called.')

    def new_descriptor(self):
        return self.matrix_descriptor_class(self.bd, self.gd, self)

    def __repr__(self):
        return uncamelcase(self.__class__.__name__)

    def get_description(self):
        """Description of this object in prose, e.g. for logging.

        Subclasses are expected to override this with something useful."""
        return repr(self)


class BlacsLayouts(KohnShamLayouts):
    using_blacs = True

    def __init__(self, gd, bd, mcpus, ncpus, blocksize, timer=nulltimer):
        KohnShamLayouts.__init__(self, gd, bd, timer)
        self._kwargs.update({'mcpus': mcpus, 'ncpus': ncpus, 'blocksize': blocksize})

        bcommsize = self.bd.comm.size
        gcommsize = self.gd.comm.size
        shiftks = self.world.rank - self.world.rank % (bcommsize * gcommsize)
        column_ranks = shiftks + np.arange(bcommsize) * gcommsize
        block_ranks = shiftks + np.arange(bcommsize * gcommsize)
        self.columncomm = self.world.new_communicator(column_ranks) #XXX rename?
        self.blockcomm = self.world.new_communicator(block_ranks)

        assert mcpus * ncpus <= bcommsize * gcommsize
        self.blockgrid = BlacsGrid(self.blockcomm, mcpus, ncpus)

    def get_description(self):
        title = 'BLACS'
        template = '%d x %d grid with %d x %d blocksize'
        return (title, template)


class BandLayouts(KohnShamLayouts):
    matrix_descriptor_class = BandMatrixDescriptor

    def diagonalize(self, H_NN, eps_n):
        nbands = self.bd.nbands
        eps_N = np.empty(nbands)
        self.timer.start('Diagonalize')
        info = self._diagonalize(H_NN, eps_N)
        self.timer.stop('Diagonalize')
        if info != 0:
            raise RuntimeError('Failed to diagonalize: %d' % info)

        self.timer.start('Distribute results')
        if self.gd.comm.rank == 0:
            self.bd.distribute(eps_N, eps_n)
            self.bd.comm.broadcast(H_NN, 0)
        self.gd.comm.broadcast(H_NN, 0)
        self.gd.comm.broadcast(eps_n, 0)
        self.timer.stop('Distribute results')

    def _diagonalize(self, H_NN, eps_N):
        """Serial diagonalizer."""
        # Only one processor really does any work.
        if self.gd.comm.rank == 0 and self.bd.comm.rank == 0:
            return diagonalize(H_NN, eps_N)
        else:
            return 0

    def inverse_cholesky(self, S_NN):
        self.timer.start('Inverse Cholesky')
        info = self._inverse_cholesky(S_NN)
        self.timer.stop('Inverse Cholesky')
        if info != 0:
            raise RuntimeError('Failed to orthogonalize: %d' % info)

        self.timer.start('Distribute results')
        if self.bd.comm.rank == 0:
            self.gd.comm.broadcast(S_NN, 0)
        self.bd.comm.broadcast(S_NN, 0)
        self.timer.stop('Distribute results')

    def _inverse_cholesky(self, S_NN):
        """Serial inverse cholesky."""
        # Only one processor really does any work.
        if self.gd.comm.rank == 0 and self.bd.comm.rank == 0:
            return inverse_cholesky(S_NN)
        else:
            return 0

    def get_description(self):
        return 'Serial LAPACK'


class OldSLBandLayouts(BandLayouts): #old SL before BLACS grids. TODO delete!
    """Original ScaLAPACK diagonalizer using 
    redundantly distributed arrays."""
    def __init__(self, gd, bd, timer=nulltimer, root=0):
        BandLayouts.__init__(self, gd, bd, timer)
        bcommsize = self.bd.comm.size
        gcommsize = self.gd.comm.size
        shiftks = self.world.rank - self.world.rank % (bcommsize * gcommsize)
        block_ranks = shiftks + np.arange(bcommsize * gcommsize)
        self.blockcomm = self.world.new_communicator(block_ranks)
        self.root = root
        # Keep buffers?

    def _diagonalize(self, H_NN, eps_N):
        # Work is done on BLACS grid, but one processor still collects
        # all eigenvectors. Only processors on the BLACS grid return
        # meaningful values of info.
        return sldiagonalize(H_NN, eps_N, self.blockcomm, root=self.root)

    def _inverse_cholesky(self, S_NN):
        return slinverse_cholesky(S_NN, self.blockcomm, self.root)

    def get_description(self):
        return 'Old ScaLAPACK'


class BlacsBandLayouts(BlacsLayouts): #XXX should derive from BandLayouts too!
    """ScaLAPACK Dense Linear Algebra.

    This class is instantiated in the real-space code.  Not for
    casual use, at least for now.
    
    Requires two distributors and three descriptors for initialization
    as well as grid descriptors and band descriptors. Distributors are
    for cols2blocks (1D -> 2D BLACS grid) and blocks2rows (2D -> 1D
    BLACS grid). ScaLAPACK operations must occur on a 2D BLACS grid for
    performance and scalability. Redistribute of 1D *column* layout
    matrix will operate only on lower half of H or S. Redistribute of
    2D block will operate on entire matrix for U, but only lower half
    of C.

    inverse_cholesky is "hard-coded" for real-space code.
    Expects overlap matrix (S) and the coefficient matrix (C) to be a
    replicated data structures and *not* created by the BLACS descriptor class. 
    This is due to the MPI_Reduce and MPI_Broadcast that will occur
    in the parallel matrix multiply. Input matrices should be:
    S = np.empty((nbands, mybands), dtype)
    C = np.empty((mybands, nbands), dtype)

    
    _standard_diagonalize is "hard-coded" for the real-space code.
    Expects both hamiltonian (H) and eigenvector matrix (U) to be a
    replicated data structures and not created by the BLACS descriptor class.
    This is due to the MPI_Reduce and MPI_Broadcast that will occur
    in the parallel matrix multiply. Input matrices should be:
    H = np.empty((nbands, mynbands), dtype)
    U = np.empty((mynbands, nbands), dtype)
    eps_n = np.empty(mynbands, dtype = float)
    """ #XXX rewrite this docstring a bit!

    matrix_descriptor_class = BlacsBandMatrixDescriptor

    # This class 'describes' all the realspace Blacs-related layouts
    def __init__(self, gd, bd, mcpus, ncpus, blocksize, timer=nulltimer):
        BlacsLayouts.__init__(self, gd, bd, mcpus, ncpus, blocksize, timer)

        nbands = bd.nbands
        mynbands = bd.mynbands

        # 1D layout - columns
        self.columngrid = BlacsGrid(self.columncomm, 1, bd.comm.size)
        self.Nndescriptor = self.columngrid.new_descriptor(nbands, nbands,
                                                           nbands, mynbands)

        # 2D layout
        self.nndescriptor = self.blockgrid.new_descriptor(nbands, nbands,
                                                          blocksize, blocksize)

        # 1D layout - rows
        self.rowgrid = BlacsGrid(self.columncomm, bd.comm.size, 1)
        self.nNdescriptor = self.rowgrid.new_descriptor(nbands, nbands,
                                                        mynbands, nbands)

        # Only redistribute filled out half for Hermitian matrices
        self.Nn2nn = Redistributor(self.blockcomm, self.Nndescriptor,
                                   self.nndescriptor)
        #self.Nn2nn = Redistributor(self.blockcomm, self.Nndescriptor,
        #                           self.nndescriptor, 'L') #XXX faster but...

        # Resulting matrix will be used in dgemm which is symmetry obvlious
        self.nn2nN = Redistributor(self.blockcomm, self.nndescriptor,
                                   self.nNdescriptor)
        
    def diagonalize(self, H_nn, eps_n):
        nbands = self.bd.nbands
        eps_N = np.empty(nbands)
        self.timer.start('Diagonalize')
        info = self._diagonalize(H_nn, eps_N)
        self.timer.stop('Diagonalize')
        if info != 0:
            raise RuntimeError('Failed to diagonalize: %d' % info)

        self.timer.start('Distribute results')
        if self.gd.comm.rank == 0:
            # grid master with bd.rank = 0 
            # scatters to other grid masters
            # NOTE: If the origin of the blacs grid
            # ever shifts this will not work
            self.bd.distribute(eps_N, eps_n)
        self.gd.comm.broadcast(eps_n, 0)
        self.timer.stop('Distribute results')

    def _diagonalize(self, H_nn, eps_N):
        """Parallel diagonalizer."""
        self.nndescriptor.diagonalize_dc(H_nn.copy(), H_nn, eps_N, 'L')
        return 0 #XXX scalapack_diagonalize_dc doesn't return this info!!!

    def inverse_cholesky(self, S_nn):
        self.timer.start('Inverse Cholesky')
        info = self._inverse_cholesky(S_nn)
        self.timer.stop('Inverse Cholesky')
        if info != 0:
            raise RuntimeError('Failed to orthogonalize: %d' % info)

    def _inverse_cholesky(self, S_nn):
        self.nndescriptor.inverse_cholesky(S_nn, 'L')
        return 0 #XXX scalapack_inverse_cholesky doesn't return this info!!!

    def get_description(self):
        (title, template) = BlacsLayouts.get_description(self)
        bg = self.blockgrid
        desc = self.nndescriptor
        s = template % (bg.nprow, bg.npcol, desc.mb, desc.nb)
        return ' '.join([title, s])


class BlacsOrbitalLayouts(BlacsLayouts):
    """ScaLAPACK Dense Linear Algebra.

    This class is instantiated in LCAO.  Not for casual use, at least for now.
    
    Requires two distributors and three descriptors for initialization
    as well as grid descriptors and band descriptors. Distributors are
    for cols2blocks (1D -> 2D BLACS grid) and blocks2cols (2D -> 1D
    BLACS grid). ScaLAPACK operations must occur on 2D BLACS grid for
    performance and scalability.

    _general_diagonalize is "hard-coded" for LCAO.
    Expects both Hamiltonian and Overlap matrix to be on the 2D BLACS grid. 
    This is done early on to save memory.
    """ #XXX rewrite this docstring a bit!

    # This class 'describes' all the LCAO Blacs-related layouts
    def __init__(self, gd, bd, mcpus, ncpus, blocksize, nao, timer=nulltimer):
        BlacsLayouts.__init__(self, gd, bd, mcpus, ncpus, blocksize, timer)
        self._kwargs.update(nao=nao) #XXX should only be done by OrbitalLayouts!

        nbands = bd.nbands
        mynbands = bd.mynbands
        self.orbital_comm = self.bd.comm
        naoblocksize = -((-nao) // self.orbital_comm.size)
        self.nao = nao

        # Range of basis functions for BLACS distribution of matrices:
        self.Mmax = nao
        self.Mstart = bd.comm.rank * naoblocksize
        self.Mstop = min(self.Mstart + naoblocksize, self.Mmax)
        self.mynao = self.Mstop - self.Mstart

        # Column layout for one matrix per band rank:
        self.columngrid = BlacsGrid(bd.comm, bd.comm.size, 1)
        self.mMdescriptor = self.columngrid.new_descriptor(nao, nao,
                                                           naoblocksize, nao)
        self.nMdescriptor = self.columngrid.new_descriptor(nbands, nao,
                                                           mynbands, nao)
        assert self.mMdescriptor.shape == (self.mynao, nao)

        #parallelprint(world, (mynao, self.mMdescriptor.shape))

        # Column layout for one matrix in total (only on grid masters):
        self.single_column_grid = BlacsGrid(self.columncomm, bd.comm.size, 1)
        self.mM_unique_descriptor = self.single_column_grid.new_descriptor( \
            nao, nao, naoblocksize, nao)

        # nM_unique_descriptor is meant to hold the coefficients after
        # diagonalization.  BLACS requires it to be nao-by-nao, but
        # we only fill meaningful data into the first nbands columns.
        #
        # The array will then be trimmed and broadcast across
        # the grid descriptor's communicator.
        self.nM_unique_descriptor = self.single_column_grid.new_descriptor( \
            nbands, nao, mynbands, nao)

        # Fully blocked grid for diagonalization with many CPUs:
        self.mmdescriptor = self.blockgrid.new_descriptor(nao, nao, blocksize,
                                                          blocksize)

        #self.nMdescriptor = nMdescriptor
        self.mM2mm = Redistributor(self.blockcomm, self.mM_unique_descriptor,
                                   self.mmdescriptor)
        self.mm2nM = Redistributor(self.blockcomm, self.mmdescriptor,
                                   self.nM_unique_descriptor)

    def diagonalize(self, H_mm, C_nM, eps_n, S_mm):
        # C_nM needs to be simultaneously compatible with:
        # 1. outdescriptor
        # 2. broadcast with gd.comm
        # We will does this with a dummy buffer C2_nM
        indescriptor = self.mM2mm.srcdescriptor #cols2blocks
        outdescriptor = self.mm2nM.dstdescriptor #blocks2cols
        blockdescriptor = self.mM2mm.dstdescriptor #cols2blocks

        dtype = S_mm.dtype
        eps_M = np.empty(C_nM.shape[-1]) # empty helps us debug
        subM, subN = outdescriptor.gshape
        
        C_mm = blockdescriptor.zeros(dtype=dtype)
        self.timer.start('General diagonalize')
        blockdescriptor.general_diagonalize_ex(H_mm, S_mm.copy(), C_mm, eps_M,
                                               UL='L', iu=self.bd.nbands)
        self.timer.stop('General diagonalize')
 
       # Make C_nM compatible with the redistributor
        self.timer.start('Redistribute coefs')
        if outdescriptor:
            C2_nM = C_nM
        else:
            C2_nM = outdescriptor.empty(dtype=dtype)
        assert outdescriptor.check(C2_nM)
        self.mm2nM.redistribute_submatrix(C_mm, C2_nM, subM, subN) #blocks2cols
        self.timer.stop('Redistribute coefs')

        self.timer.start('Send coefs to domains')
        if outdescriptor: # grid masters only
            assert self.gd.comm.rank == 0
            # grid master with bd.rank = 0 
            # scatters to other grid masters
            # NOTE: If the origin of the blacs grid
            # ever shifts this will not work
            self.bd.distribute(eps_M[:self.bd.nbands], eps_n)
        else:
            assert self.gd.comm.rank != 0

        self.gd.comm.broadcast(C_nM, 0)
        self.gd.comm.broadcast(eps_n, 0)
        self.timer.stop('Send coefs to domains')

    def distribute_overlap_matrix(self, S_qmM, root=0):
        # Some MPI implementations need a lot of memory to do large
        # reductions.  To avoid trouble, we do comm.sum on smaller blocks
        # of S (this code is also safe for arrays smaller than blocksize)
        Sflat_x = S_qmM.ravel()
        blocksize = 2**23 // Sflat_x.itemsize # 8 MiB
        nblocks = -(-len(Sflat_x) // blocksize)
        Mstart = 0
        for i in range(nblocks):
            self.gd.comm.sum(Sflat_x[Mstart:Mstart + blocksize])
            Mstart += blocksize
        assert Mstart + blocksize >= len(Sflat_x)

        xshape = S_qmM.shape[:-2]
        nm, nM = S_qmM.shape[-2:]
        S_qmM = S_qmM.reshape(-1, nm, nM)
        
        blockdesc = self.mmdescriptor
        coldesc = self.mM_unique_descriptor
        S_qmm = blockdesc.zeros(len(S_qmM), S_qmM.dtype)

        if not coldesc: # XXX ugly way to sort out inactive ranks
            S_qmM = coldesc.zeros(len(S_qmM), S_qmM.dtype)
        
        self.timer.start('Distribute overlap matrix')
        for S_mM, S_mm in zip(S_qmM, S_qmm):
            self.mM2mm.redistribute(S_mM, S_mm)
        self.timer.stop('Distribute overlap matrix')
        return S_qmm.reshape(xshape + blockdesc.shape)

    def get_overlap_matrix_shape(self):
        return self.mmdescriptor.shape

    def calculate_density_matrix(self, f_n, C_nM, rho_mM=None):
        nbands = self.bd.nbands
        mynbands = self.bd.mynbands
        nao = self.nao
        
        if rho_mM is None:
            rho_mM = self.mMdescriptor.zeros(dtype=C_nM.dtype)
        
        Cf_nM = (C_nM * f_n[:, None]).conj()
        pblas_simple_gemm(self.nMdescriptor, self.nMdescriptor,
                          self.mMdescriptor, Cf_nM, C_nM, rho_mM, transa='T')
        return rho_mM

    def get_transposed_density_matrix(self, f_n, C_nM, rho_mM=None):
        # XXX for the complex case, find out whether this or the other
        # method should be changed
        return self.calculate_density_matrix(f_n, C_nM, rho_mM)

    def get_description(self):
        (title, template) = BlacsLayouts.get_description(self)
        bg = self.blockgrid
        desc = self.mmdescriptor
        s = template % (bg.nprow, bg.npcol, desc.mb, desc.nb)
        return ' '.join([title, s])


class OrbitalLayouts(KohnShamLayouts):
    def __init__(self, gd, bd, nao, timer=nulltimer):
        KohnShamLayouts.__init__(self, gd, bd, timer)
        self._kwargs.update({'nao': nao})
        self.mMdescriptor = MatrixDescriptor(nao, nao)
        self.nMdescriptor = MatrixDescriptor(bd.mynbands, nao)
        
        self.Mstart = 0
        self.Mstop = nao
        self.Mmax = nao
        self.mynao = nao
        self.nao = nao
        self.orbital_comm = serial_comm

    def diagonalize(self, H_MM, C_nM, eps_n, S_MM):
        eps_M = np.empty(C_nM.shape[-1])
        info = self._diagonalize(H_MM, S_MM.copy(), eps_M)
        if info != 0:
            raise RuntimeError('Failed to diagonalize: %d' % info)
        
        nbands = self.bd.nbands
        if self.bd.rank == 0:
            self.gd.comm.broadcast(H_MM[:nbands], 0)
            self.gd.comm.broadcast(eps_M[:nbands], 0)
        self.bd.distribute(H_MM[:nbands], C_nM)
        self.bd.distribute(eps_M[:nbands], eps_n)
    
    def _diagonalize(self, H_MM, S_MM, eps_M):
        # Only one processor really does any work.
        if self.gd.comm.rank == 0 and self.bd.comm.rank == 0:
            return general_diagonalize(H_MM, eps_M, S_MM)
        else:
            return 0

    def estimate_memory(self, mem, dtype):
        nao = self.setups.nao
        itemsize = mem.itemsize[dtype]
        mem.subnode('eps [M]', self.nao * mem.floatsize)
        mem.subnode('H [MM]', self.nao * self.nao * itemsize)

    def distribute_overlap_matrix(self, S_qMM, root=0):
        self.gd.comm.sum(S_qMM, root)
        return S_qMM

    def get_overlap_matrix_shape(self):
        return self.nao, self.nao

    def calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        # Only a madman would use a non-transposed density matrix.
        # Maybe we should use the get_transposed_density_matrix instead
        if rho_MM is None:
            rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        # XXX Should not conjugate, but call gemm(..., 'c')
        # Although that requires knowing C_Mn and not C_nM.
        # that also conforms better to the usual conventions in literature
        Cf_Mn = C_nM.T.conj() * f_n
        gemm(1.0, C_nM, Cf_Mn, 0.0, rho_MM, 'n')
        self.bd.comm.sum(rho_MM)
        return rho_MM

    def get_transposed_density_matrix(self, f_n, C_nM, rho_MM=None):
        return self.calculate_density_matrix(f_n, C_nM, rho_MM).T.copy()

        #if rho_MM is None:
        #    rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        #C_Mn = C_nM.T.copy()
        #gemm(1.0, C_Mn, f_n[np.newaxis, :] * C_Mn, 0.0, rho_MM, 'c')
        #self.bd.comm.sum(rho_MM)
        #return rho_MM

    def alternative_calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        if rho_MM is None:
            rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        # Alternative suggestion. Might be faster. Someone should test this
        C_Mn = C_nM.T.copy()
        r2k(0.5, C_Mn, f_n * C_Mn, 0.0, rho_MM)
        tri2full(rho_MM)
        return rho_MM

    def get_description(self):
        return 'Serial LAPACK'


class OldSLOrbitalLayouts(OrbitalLayouts): #old SL before BLACS grids. TODO delete!
    """Original ScaLAPACK diagonalizer using 
    redundantly distributed arrays."""
    def __init__(self, gd, bd, nao, timer=nulltimer, root=0):
        OrbitalLayouts.__init__(self, gd, bd, nao, timer)
        bcommsize = self.bd.comm.size
        gcommsize = self.gd.comm.size
        shiftks = self.world.rank - self.world.rank % (bcommsize * gcommsize)
        block_ranks = shiftks + np.arange(bcommsize * gcommsize)
        self.blockcomm = self.world.new_communicator(block_ranks)
        self.root = root
        # Keep buffers?

    def _diagonalize(self, H_MM, S_MM, eps_M):
        # Work is done on BLACS grid, but one processor still collects
        # all eigenvectors. Only processors on the BLACS grid return
        # meaningful values of info.
        return slgeneral_diagonalize(H_MM, eps_M, S_MM, self.blockcomm,
                                     root=self.root)

    def get_description(self):
        return 'Old ScaLAPACK'


