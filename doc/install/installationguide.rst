.. _installationguide:

==================
Installation guide
==================

.. note::

   **CAMd users** installing on ``Niflheim``: please follow the
   instructions for :ref:`Niflheim`.

In order to choose the right method for installing GPAW
identify your computer system and your goals related to GPAW.
Is it a personal laptop or maybe an HPC cluster?
Are you just trying out GPAW or need a full development environment
in order to participate in developing GPAW?

For the installation on personal laptops we recommend
the binary packages provided for popular Linux distributions
(:ref:`installationguide_package`)
and MS Windows (:ref:`installationguide_windows`).

Please skip to :ref:`installationguide_developer` if you prefer
to install from sources.

If you are on Mac OSX, please follow :ref:`installationguide_macosx`.


.. _installationguide_package:

Installation with package manager on Linux
==========================================

This is **the preferred** way to install on a Linux system.
If you prefer to install from sources follow :ref:`installationguide_developer`.

Configure the package repositories as described at
`Installation with package manager on Linux <https://wiki.fysik.dtu.dk/ase/download.html#installation-with-package-manager-on-linux>`_,
and install GPAW with:

- on RHEL/CentOS/Fedora::

    yum -u install gpaw

- on openSUSE::

    yast -i gpaw

- on Debian/Ubuntu::

    sudo apt-get update
    sudo apt-get -y install gpaw

For the full list of supported distributions check
https://build.opensuse.org/package/show?package=gpaw&project=home%3Adtufys

After performing the installation do not forget to :ref:`running_tests`!


.. _installationguide_macosx:

Installation on OS X
====================

For installation with http://brew.sh/ please follow
instructions at :ref:`homebrew`.

After performing the installation do not forget to :ref:`running_tests`!


.. _installationguide_windows:

Installation on Windows
=======================

.. note::

   GPAW is not yet fully functional on Windows! See
   http://listserv.fysik.dtu.dk/pipermail/gpaw-users/2013-August/002264.html

On Windows install ASE dependencies as described at
https://wiki.fysik.dtu.dk/ase/download.html#windows.

Download the gpaw.win32-py2.7.msi_ installer and install with::

   gpaw.win32-py2.7.msi /l*vx "%TMP%\gpaw_install.log" /passive

.. _gpaw.win32-py2.7.msi:
       https://wiki.fysik.dtu.dk/gpaw-files/gpaw.win32-py2.7.msi

.. note::

    Unpack gpaw-setups under C:\gpaw-setups (see :ref:`setups`).

As the last step (this is important) install the ASE
(see https://wiki.fysik.dtu.dk/ase/download.html#windows).

After performing the installation do not forget to :ref:`running_tests`!


.. _installationguide_developer:

Manual installation
===================

GPAW binaries are available only for the :ref:`latest_stable_release`,
and all available GPAW releases are listed at the :ref:`download` page.

If you need a development version (or a historic version) of GPAW
perform a manual installation according to instructions below.
Follow the same instructions if you are configuring GPAW on an HPC cluster.


This is the **preferred** way of manually installing GPAW.
It offers the following advantages:

- installation is limited to standard user's account:
  it does not pollute the root filesystem,

- user gains access to version control updates, if necessary.

Requirements
------------

1) Python2 version 2.6 or newer. Python3 is not supported yet.
   Python is available from http://www.python.org.

2) NumPy_ 1.6.1 or newer.  Earlier versions may work for basic operations.

3) Atomic Simulation Environment (:ase:`ASE <>`).

4) C compiler - preferably gcc.

5) Libxc version 2.0.1 or newer (libxc-download_).

6) BLAS and LAPACK libraries. Start with your system provided defaults or
   e.g. acml_ or openblas_. Multithreading is not supported.

7) SciPy_ 0.7.0 or newer

Optionally:

8) an MPI library (required for parallel calculations).

9) HDF5 (> 1.8.0) library for parallel I/O and for saving files in HDF5 format


.. _NumPy: http://numpy.org/
.. _SciPy: http://scipy.org/
.. _libxc-download: http://www.tddft.org/programs/octopus/wiki/index.php/
                    Libxc:download
.. _acml: http://developer.amd.com/tools-and-sdks/cpu-development/
          amd-core-math-library-acml/
.. _openblas: http://www.openblas.net/

Installation process
--------------------

After having installed the dependencies do:

1) Perform :ref:`developer_installation`.

   .. note::

       If you install on a cluster,
       take a look at :ref:`install_custom_installation` - it provides
       installation instructions for different platforms.

2) Perform :ref:`installationguide_setup_files`.

3) :ref:`running_tests`.

The following environment variables are involved when installing GPAW:

.. envvar:: HOME

  The path to your home directory.

.. envvar:: GPAW_HOME

  Optional: points to the root directory of your GPAW installation, e.g.:
  ``~/gpaw``.

.. envvar:: PATH

  The ``$PATH`` environment variable should contain the paths to directory
  containing the ``gpaw-python`` executable and additional scripts.

.. envvar:: PYTHONPATH

  The ``PYTHONPATH`` should point to the directory containing the GPAW Python
  module and the ``_gpaw.so`` shared library.

.. envvar:: OMP_NUM_THREADS
  
  Currently should be set to 1.

.. envvar:: GPAW_SETUP_PATH

  Points to the directory containing the PAW datasets.

See below for hints how to customize your installation.

Installation tricks
-------------------

.. _install_custom_installation:

Customizing installation
++++++++++++++++++++++++

The install script does its best when trying to guess proper libraries
and commands to build GPAW. However, if the standard procedure fails
or user wants to override default values it is possible to customize
the setup with :svn:`customize.py` file which is located in the GPAW base
directory. As an example, :svn:`customize.py` might contain the following
lines::

  libraries = ['myblas', 'mylapack']
  library_dirs = ['path_to_myblas']

Now, GPAW would be built with "``-Lpath_to_myblas -lmyblas
-lmylapack``" linker flags. Look at the file :svn:`customize.py`
itself for more possible options.  :ref:`platforms_and_architectures`
provides examples of :file:`customize.py` for different platforms.
After editing :svn:`customize.py`, follow the instructions for the
:ref:`developer_installation`.

Installation with HDF5 support
++++++++++++++++++++++++++++++

HDF5 support can be enabled by setting in :file:`customize.py`::

 hdf5 = True

and, in this case, provide HDF5 ``include_dirs``, ``libraries``, and
``library_dirs`` as described in :ref:`install_custom_installation`.

.. _parallel_installation:

Parallel installation
+++++++++++++++++++++

By default, setup looks if :program:`mpicc` is available, and if setup
finds one, a parallel version is build. If the setup does not find
mpicc, a user can specify one in the :svn:`customize.py` file.

Additionally a user may want to enable ScaLAPACK, setting in
:file:`customize.py`::

 scalapack = True

and, in this case, provide BLACS/ScaLAPACK ``libraries`` and ``library_dirs``
as described in :ref:`install_custom_installation`.

Instructions for running parallel calculations can be found in the
:ref:`user manual <manual_parallel_calculations>`.


.. _PGO:

Profile guided optimization
+++++++++++++++++++++++++++

Some compilers allow one to use
`profile guided optimization <http://en.wikipedia.org/wiki/Profile-guided_optimization>`_ (PGO).
See :ref:`PGO_gcc_EL5` for an example how use PGO to compile GPAW on CentOS.

Libxc Installation
++++++++++++++++++

Libxc download/install instructions can be found `here <http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:download>`_.  A few extra tips:

- Libxc installation requires both a C compiler and a fortran compiler.

- We've tried intel and gnu compilers and haven't noticed much of a
  performance difference.  Use whatever is easiest.

- Libxc shared libraries can be built with the "--enable-shared" option
  to configure.  This might be slightly preferred because it reduces
  memory footprints for executables.

- Typically when building GPAW one has to modify customize.py in a manner
  similar to the following::

    library_dirs += ['/my/path/to/libxc/2.0.2/install/lib']
    include_dirs += ['/my/path/to/libxc/2.0.2/install/include']

  or if you don't want to modify your customize.py, you can add these lines to
  your .bashrc::
  
    export C_INCLUDE_PATH=/my/path/to/libxc/2.0.2/install/include
    export LIBRARY_PATH=/my/path/to/libxc/2.0.2/install/lib
    export LD_LIBRARY_PATH=/my/path/to/libxc/2.0.2/install/lib

Example::
    
    wget http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.0.2.tar.gz -O libxc-2.0.2.tar.gz
    tar -xf libxc-2.0.2.tar.gz
    cd libxc-2.0.2
    ./configure --enable-shared --prefix=$HOME/xc
    make
    make install
    
    # add these to your .bashrc:
    export C_INCLUDE_PATH=~/xc/include
    export LIBRARY_PATH=~/xc/lib
    export LD_LIBRARY_PATH=~/xc/lib


.. _installationguide_setup_files:

Installation of PAW datasets
============================

1) Get the tar file :file:`gpaw-setups-{<version>}.tar.gz`
   of the <version> of PAW datasets from the :ref:`setups` page
   and unpack it somewhere, preferably in :envvar:`HOME`
   (``cd; tar -xf gpaw-setups-<version>.tar.gz``) - it could
   also be somewhere global where
   many users can access it like in :file:`/usr/share/gpaw-setups/`.
   There will now be a subdirectory :file:`gpaw-setups-{<version>}/`
   containing all the atomic data for the most commonly used functionals.

2) Set the environment variable :envvar:`GPAW_SETUP_PATH`
   to point to the directory
   :file:`gpaw-setups-{<version>}/`, e.g. put into :file:`~/.tcshrc`::

    setenv GPAW_SETUP_PATH ${HOME}/gpaw-setups-<version>

   or if you use bash, put these lines into :file:`~/.bashrc`::

    export GPAW_SETUP_PATH=${HOME}/gpaw-setups-<version>

   Refer to :ref:`using_your_own_setups` for alternative way of
   setting the location of PAW datasets.

   .. note::

     In case of several locations of PAW datasets the first found setup
     file is used.


.. _running_tests:

Run the tests
=============

Make sure that everything works by running the test suite
in serial (using bash)::

  [gpaw]$ python `which gpaw-test` 2>&1 | tee test.log

If you compiled the custom interpreter (needed to running calculations
in parallel), test it too, in serial::

  [gpaw]$ gpaw-python `which gpaw-test` 2>&1 | tee test1.log

This will take a couple of hours.
Please report errors to the ``gpaw-developers`` mailing list (see
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

   Many MPI versions have their own ``-c`` option which may
   invalidate python command line options. In this case
   test the parallel code as in the example below.

Try also::

  [gpaw]$ mpirun -np 2 gpaw-python gpaw/test/spinpol.py

This will perform a calculation for a single hydrogen atom.
First spin-paired then spin-polarized case, the latter parallelized
over spin up on one processor and spin down on the other.  If you run
the example on 4 processors, you get parallelization over both
spins and the domain.

If you enabled ScaLAPACK, do::

  [examples]$ mpirun -np 2 gpaw-python ~/gpaw/test/CH4.py --sl_default=1,2,2

This will enable ScaLAPACK's diagonalization on a 1x2 BLACS grid
with the block size of 2.

Finally run the tests in parallel on 2, 4 and 8 cores::

  [gpaw]$ mpirun -np 4 gpaw-python `which gpaw-test` 2>&1 | tee test4.log

