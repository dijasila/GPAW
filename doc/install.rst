.. _installation:

============
Installation
============

GPAW relies on the Python library *atomic simulation environment* (ASE_),
so you need to :ref`install ASE <ase:install>` first.  GPAW itself is written
mostly in the Python programming language, but there are also some
C-code used for:
    
* performance critical parts
* allowing Python to talk to external numerical libraries (BLAS_, LAPACK_,
  LibXC_, MPI_ and ScaLAPACK_)

So, in order to make GPAW work, you need to compile some C-code.  For serial
calculations, you will need to build a dynamically linked library
(``_gpaw.so``) that the standard Python interpreter can load.  For parallel
calculations, you need to build a new Python interpreter (``gpaw-python``)
that has MPI_ functionality built in.


Requirements
============

* Python_ 2.6-3.5
* NumPy_ 1.6.1 or later (base N-dimensional array package)
* `Atomic Simulation Environment <https://wiki.fysi.dtu.dk/ase>`_
* a C compiler
* LibXC_ 2.0.1 or later
* BLAS_ and LAPACK_ libraries

Optional, but highly recommended:

* SciPy_ 0.7 or later (library for scientific computing, requirered for
  some features)
* an MPI_ library (required for parallel calculations)
* FFTw_ (for increased performance)
* BLACS_ and ScaLAPACK_

Optional (maybe not needed):
    
* HDF5 1.8.0 or later (library for parallel I/O and for saving files in HDF5
  format)


.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _LibXC: http://www.tddft.org/programs/octopus/wiki/index.php/Libxc


Installation using ``pip``
==========================

.. highlight:: bash

The simplest way to install GPAW is using pip_ and the GPAW package from the
Python package index (PyPI_)::
    
    $ pip install --upgrade --user gpaw
    
This will compile and install GPAW (both ``_gpaw.so`` and all the Python files) in your ``~/.local/lib/pythonX.Y/site-packages`` folder where Python
can automatically find it.  The ``pip`` command will also place the command
line tool :command:`gpaw` in the ``~/.local/bin`` folder, so make sure you
have that in your :envvar:`PATH` environment variable.  If you have an
``mpicc`` command on your system then there will also be a ``gpaw-python``
executable in ``~/.local/bin``.

Check that you have installed everything in the correct places::
    
    $ gpaw status
    
.. note::

    Some Linux distributions have a GPAW package (named ``gpaw``),
    that you can install on your system so that it is avilable for all
    users.

    
Install PAW datasets
====================

This is done using this command::
    
    $ gpaw install-data
    
See :ref:`installation of paw datasets` for more details.

Now you should be ready to use GPAW, but before you start, please run the
tests as described below.


.. index:: test
.. _running tests:

Test your installation
======================

Run the tests like this::
    
    $ gpaw test -j 4  # takes 1 hour!

and send us the output if there are failing tests.


.. _download:

Installation from source
========================

As an alternative to ``pip``, you can also get the source from a tar-file or
from Git.


:Tar-file:

    You can get the source as a tar-file for the
    latest stable release (gpaw-1.0.0.tar.gz_) or the latest
    development snapshot (`<snapshot.tar.gz>`_).

    Unpack and make a soft link::
    
        $ tar -xf python-gpaw-3.9.1.4567.tar.gz
        $ ln -s python-gpaw-3.9.1.4567 gpaw

:Git clone:

    Alternatively, you can get the source for the latest stable release from
    https://gitlab.com/gpaw/gpaw like this::
    
        $ git clone -b 3.9.1 https://gitlab.com/gpaw/gpaw.git

    or if you want the development version::

        $ git clone https://gitlab.com/gpaw/gpaw.git
    
Add ``~/gpaw`` to your :envvar:`PYTHONPATH` environment variable and add
``~/gpaw/tools`` to :envvar:`PATH` (assuming ``~/gpaw`` is where your GPAW
folder is).
    
.. note::
    
    We also have Git tags for older stable versions of GPAW.
    See the :ref:`releasenotes` for which tags are available.  Also the
    dates of older releases can be found there.


.. _gpaw-1.0.0.tar.gz:
    https://pypi.python.org/packages/source/g/gpaw/gpaw-1.0.0.tar.gz

Niflheim, datasets, platforms, devel-mode





See below for hints how to customize your installation.

Installation tricks
-------------------

.. _install_custom_installation:

Customizing installation
++++++++++++++++++++++++

The install script does its best when trying to guess proper libraries
and commands to build GPAW. However, if the standard procedure fails
or user wants to override default values it is possible to customize
the setup with :git:`customize.py` file which is located in the GPAW base
directory. As an example, :git:`customize.py` might contain the following
lines::

  libraries = ['myblas', 'mylapack']
  library_dirs = ['path_to_myblas']

Now, GPAW would be built with "``-Lpath_to_myblas -lmyblas
-lmylapack``" linker flags. Look at the file :git:`customize.py`
itself for more possible options.  :ref:`platforms_and_architectures`
provides examples of :file:`customize.py` for different platforms.
After editing :git:`customize.py`, follow the instructions for the
:ref:`developer installation`.

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
mpicc, a user can specify one in the :git:`customize.py` file.

Additionally a user may want to enable ScaLAPACK, setting in
:file:`customize.py`::

 scalapack = True

and, in this case, provide BLACS/ScaLAPACK ``libraries`` and ``library_dirs``
as described in :ref:`install_custom_installation`.

Instructions for running parallel calculations can be found in the
:ref:`user manual <manual_parallel_calculations>`.


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
:ref:`mail lists`) Send us :file:`test.log`, as well as the
information about your environment (processor architecture, versions
of python and numpy, C-compiler, BLAS and LAPACK libraries, MPI
library), and (only when requested) :file:`build_ext.log`
(or :file:`install.log`).

If tests pass, and the parallel version is built, test the parallel code::

  [gpaw]$ mpirun -np 2 gpaw-python -c "import gpaw.mpi as mpi; print(mpi.rank)"
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

On Windows install Python(x,y) as described at
https://wiki.fysik.dtu.dk/ase/download.html#windows.

Download the gpaw.win32-py2.7.msi_ installer
(fix the incorrect *man* extension while downloading) and install with::

   gpaw.win32-py2.7.msi /l*vx "%TMP%\gpaw_install.log" /passive

.. _gpaw.win32-py2.7.msi:
       https://wiki.fysik.dtu.dk/gpaw-files/gpaw.win32-py2.7.msi

.. note::

    Unpack gpaw-setups under C:\gpaw-setups (see :ref:`setups`).

As the last step (this is important) install the ASE msi
(see https://wiki.fysik.dtu.dk/ase/download.html#windows).

After performing the installation do not forget to :ref:`running_tests`!
    