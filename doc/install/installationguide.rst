.. _installationguide:

==================
Installation guide
==================

Requirements
============

1) Python 2.3 or later.  Python is available from http://www.python.org.

2) NumPy_.

3) Atomic Simulation Environment (:ase:`ASE <>`).

4) C compiler - preferably gcc.

5) BLAS and LAPACK libraries. Start with your system provided defaults or e.g. http://www.amd.com/acml.
   Consult the :ref:`best_performance` page.

6) An MPI library required for parallel calculations.

.. note::

   In order to use the code, you need also the setups for all your atoms (:ref:`setups`).

.. _NumPy: http://numpy.scipy.org/

Installation from the fys package repository
============================================

GPAW and dependencies will be installed system-wide. This procedure requires root acccess.

.. note::

   Most of the fys packages are relocatable, therefore suitable for installing also on a cluster.

The packages are provided for the following RPMS-based systems:

- Fedora 13, 12: :ref:`Fedora`,

- RedHat Enterprise Linux (or CentOS) 5: :ref:`CentOS`,

- Ubuntu 9.10 or newer: :ref:`Ubuntupackage`.

If you prefer to install manually proceed to the next section.

Standard installation
=====================

1) :ref:`download` the code.

.. note::

   **CAMd users** installing on ``Niflheim``: please follow instructions for :ref:`Niflheim`.

2) Go to the :file:`gpaw` directory::

     [~]$ cd gpaw

3) If you install on a cluster,
   take a look at :ref:`install_custom_installation`.
   Then follow :ref:`developer_installation`.

   .. note::

       Take a look at :ref:`platforms_and_architectures` - it provides
       installation instructions for different platforms.

   **Note**: alternatively (**not recommended**) to
   :ref:`developer_installation`, install with the standard (using bash)::

     [gpaw]$ python setup.py install --home=<my-directory>  2>&1 | tee install.log

   and put :file:`{<my-directory>}/lib/python` (or
   :file:`{<my-directory>}/lib64/python`) in your :envvar:`$PYTHONPATH` 
   environment variable.  Moreover, if parallel environment is found on your system,
   a special :program:`gpaw-python` python-interpreter is created under
   :file:`{<my-directory>}/bin`. Please add
   :file:`{<my-directory>}/bin` to :envvar:`PATH`. Alternatively, the full pathname
   :file:`{<my-directory}>/bin/gpaw-python` can be used when executing
   parallel runs. See :ref:`parallel_installation` for more details about
   parallel runs.

   .. note::

     Usually :envvar:`$HOME` is a good choice for :file:`{<my-directory>}`.

   If you have root-permissions, you can install GPAW system-wide (using bash)::

     [gpaw]$ python setup.py install 2>&1 | tee install.log

4) Get the tar file :file:`gpaw-setups-{<version>}.tar.gz` from the 
   :ref:`setups` page
   and unpack it somewhere, preferably in :envvar:`${HOME}`
   (``cd; tar zxf gpaw-setups-<version>.tar.gz``) - it could
   also be somewhere global where
   many users can access it like in :file:`/usr/share/gpaw-setups/`.  There will
   now be a directory :file:`gpaw-setups-{<version>}/` containing all the
   atomic data needed for doing LDA, PBE, and RPBE calculations.  Set the
   environment variable :envvar:`GPAW_SETUP_PATH` to point to the directory
   :file:`gpaw-setups-{<version>}/`, e.g. put into :file:`~/.tcshrc`::

    setenv GPAW_SETUP_PATH ${HOME}/gpaw-setups-<version>

   or if you use bash, put these lines into :file:`~/.bashrc`::

    export GPAW_SETUP_PATH=${HOME}/gpaw-setups-<version>

.. _running_tests:

Run the tests
=============

Make sure that everything works by running the test suite (using bash)::

  [gpaw]$ gpaw-python `which gpaw-test` 2>&1 | tee test.log

This will take around 40 minutes.  If you have a multicore CPU, you
can speed up the test by using ``gpaw-test -j <number-of-cores>``.
This will run tests simultaneously (**not** employing MPI parallelization)
on the requested `<number-of-cores>`.
Please report errors to the ``gridpaw-developer`` mailing list (see
:ref:`mailing_lists`) Send us :file:`test.log`, as well as the
information about your environment (processor architecture, versions
of python and numpy, C-compiler, BLAS and LAPACK libraries, MPI
library), and (only when requested) :file:`build_ext.log`
(or :file:`install.log`).

If tests pass, and the parallel version is built, test the parallel code::

  [gpaw]$ mpirun -np 2 gpaw-python -c "import gpaw.mpi as mpi; print mpi.rank"
  1
  0

.. note::

   Many MPI versions have their own `-c` option which may
   invalidate python command line options. In this case
   test the parallel code as in the example below.

Try also::

  [test]$ cd ~/gpaw/examples
  [examples]$ mpirun -np 2 gpaw-python H.py

This will do a calculation for a single hydrogen atom parallelized
with spin up on one processor and spin down on the other.  If you run
the example on 4 processors, you should get parallelization over both
spins and the domain.

If you enabled ScaLAPACK, do::

  [examples]$ mpirun -np 2 gpaw-python ~/gpaw/test/CH4.py --sl_diagonalize=1,2,2

This will enable ScaLAPACK's diagonalization on a 1x2 BLACS grid
with the block size of 2. ScaLAPACK can be currently used
only in cases **without** k-points.

.. _install_custom_installation:

Custom installation
===================

The install script does its best when trying to guess proper libraries
and commands to build gpaw. However, if the standard procedure fails
or user wants to override default values it is possible to customize
the setup with :svn:`customize.py` file which is located in the gpaw base
directory. As an example, :svn:`customize.py` might contain the following
lines::

  libraries = ['myblas', 'mylapack']
  library_dirs = ['path_to_myblas']

Now, gpaw would be built with "``-Lpath_to_myblas -lmyblas
-lmylapack``" linker flags. Look at the file :svn:`customize.py`
itself for more possible options.  :ref:`platforms_and_architectures`
provides examples of :file:`customize.py` for different platforms.
After editing :svn:`customize.py`, follow the instructions for the
:ref:`installationguide` from step 3 on.

.. _PGO:

Profile guided optimization
===========================

Some compilers allow one to use
`profile guided optimization <http://en.wikipedia.org/wiki/Profile-guided_optimization>`_ (PGO).
See :ref:`PGO_gcc_EL5` for an example how use PGO to compile GPAW on CentOS.

.. _parallel_installation:

Parallel installation
=====================

By default, setup looks if :program:`mpicc` is available, and if setup
finds one, a parallel version is build. If the setup does not find
mpicc, a user can specify one in the :svn:`customize.py` file.

Additionally a user may want to enable ScaLAPACK, setting in
:file:`customize.py`::

 scalapack = True

and, if needed, providing BLACS/ScaLAPACK `libraries` and `library_dirs`
as described in :ref:`install_custom_installation`.

Instructions for running parallel calculations can be found in the
:ref:`user manual <manual_parallel_calculations>`.
