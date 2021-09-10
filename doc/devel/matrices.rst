.. module:: gpaw.matrix


GPAW's Matrix object
====================

A simple example that we can run with MPI on 4 cores::

    from gpaw.matrix import Matrix
    from gpaw.mpi import world
    a = Matrix(5, 5, dist=(world, 2, 2, 2))
    a.data[:] = world.rank
    print(world.rank, a.array.shape)

Here, we have created a 5x5 :class:`Matrix` of floats distributed on a 2x2
BLACS gris with a blocksize 2 and we then print the shapes of the ndarrays,
which looks like this (in random order)::

    1 (2, 3)
    2 (3, 2)
    3 (2, 2)
    0 (3, 3)

Let's create a new matrix ``b`` and :meth:`redistribute <Matrix.redist>` from
``a`` to ``b``::

    b = a.new(dist=(None, 1, 1, None))
    a.redist(b)
    if world.rank == 0:
        print(b.data)

This will output::

    [[ 0.  0.  2.  2.  0.]
     [ 0.  0.  2.  2.  0.]
     [ 1.  1.  3.  3.  1.]
     [ 1.  1.  3.  3.  1.]
     [ 0.  0.  2.  2.  0.]]

Matrix-matrix :meth:`multiplication <matrix_matrix_multiply>`
works like this::

    c = (a @ a.H).eval()
    c = a.new()
    c[:] = a @ a.H
    c[:] = 0.5 * a @ a.H

.. autofunction:: matrix_matrix_multiply

.. autoclass:: Matrix
    :members:
