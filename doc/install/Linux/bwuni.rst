=====
bwuni
=====

Information about `bwUniCluster <http://www.scc.kit.edu/dienste/bwUniCluster.php>`__.

Building GPAW
=============

We assume that the installation will be located in ``$HOME/source``.

libxc
--------

GPAW relies on libxc (see the `libxc web site <http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:download>`__). 
To install libxc we assume that ``MYLIBXCDIR`` is set to 
the directory where you want to install 
(e.g. ``MYLIBXCDIR=$HOME/source/libxc``)::

 $ cd $MYLIBXCDIR
 $ wget http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.0.2.tar.gz
 $ tar -xzvf libxc-2.0.2.tar.gz
 $ cd libxc-2.0.2/
 $ mkdir install
 $ ./configure CFLAGS="-fPIC" --prefix=$PWD/install -enable-shared
 $ make |tee make.log
 $ make install

This will have installed the libs ``$MYLIBXCDIR/libxc-2.0.2/install/lib`` 
and the C header
files to ``$MYLIBXCDIR/libxc-2.0.2/install/include``.

Building GPAW
-------------

We first create a place for gpaw and get the trunk version::

 GPAW_SOURCE=$PWD/source/gpaw
 mkdir -p $GPAW_SOURCE
 cd $GPAW_SOURCE
 svn checkout https://svn.fysik.dtu.dk/projects/gpaw/trunk trunk

The current trunk version can then be updated by::

 cd $GPAW_SOURCE/trunk
 svn up

We have to modify the file :file:`customize.py` to
:svn:`~doc/install/Linux/customize_bwuni.py`

.. literalinclude:: customize_bwuni.py

To build GPAW use::

 module purge
 module load mpi/openmpi
 module load numlib/python_scipy
 module unload numlib/mkl/11.0.5
 module load numlib/mkl/11.1.4 lib/hdf5

 module load libxc
 module load ase

 cd $GPAW_SOURCE/trunk
 mkdir install
 python setup.py install --prefix=$PWD/install
 python setup.py build_ext 2>&1 | tee build_ext.log

which installs GPAW to ``$GPAW_SOURCE/trunk/install``.

Running GPAW
------------

A gpaw script :file:`test.py` can be submitted to run on 8 cpus like this::

  > gpaw-runscript test.py 8
  using bwg
  run.bwg written
  > qsub run.bwg

