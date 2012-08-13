.. _installationguide:

==================
Installation guide
==================


Requirements
============

1) Python 2.4 - 2.7.  Python is available from http://www.python.org.

2) NumPy_ 1.1 - 1.5.

3) Atomic Simulation Environment (:ase:`ASE <>`).

4) C compiler - preferably gcc.

5) BLAS and LAPACK libraries. Start with your system provided defaults or
   e.g. http://www.amd.com/acml.

6) An MPI library (required for parallel calculations).

7) (Optional) HDF5 (> 1.8.0) library for parallel I/O and for saving files in HDF5 format


.. note::

   In order to use the code, you need also the setups for all your
   atoms (:ref:`setups`).

.. _NumPy: http://numpy.scipy.org/


Installation
============

Below the recommended ways of installing GPAW
are described, in order of preference.

.. note::

   **CAMd users** installing on ``Niflheim``: please follow instructions
   for :ref:`Niflheim`.


.. _installationguide_package:

Installation with package manager on Linux
------------------------------------------

Install the binaries with the software package manager of your Linux distribution.
This is **the preferred** way to install on a Linux system.
If you prefer to install from sources follow :ref:`installationguide_developer`.

The currently supported systems include (issue the commands below **as root**):

- RHEL/CentOS 6::

    yum install wget
    cd /etc/yum.repos.d/
    wget http://download.opensuse.org/repositories/home:/dtufys/CentOS_CentOS-6/home:dtufys.repo
    yum install gpaw

- Fedora 17::

    yum install wget
    cd /etc/yum.repos.d/
    wget http://download.opensuse.org/repositories/home:/dtufys/Fedora_17/home:dtufys.repo
    yum install gpaw

- openSUSE 12.2::

    zypper ar -f http://download.opensuse.org/repositories/home:/dtufys/openSUSE_12.2/home:dtufys.repo
    yast -i gpaw

- Debian 6.0::

    sudo bash -c 'echo "deb http://widehat.opensuse.org/repositories/home:/dtufys/Debian_6.0 /" > /etc/apt/sources.list.d/home_dtufys.sources.list'
    wget http://widehat.opensuse.org/repositories/home:/dtufys/Debian_6.0/Release.key && sudo apt-key add Release.key && rm Release.key
    sudo apt-get update
    sudo apt-get install gpaw

- Ubuntu. Install `Ubuntu package <https://wiki.fysik.dtu.dk/gpaw/install/Linux/Ubuntu_ppa.html#ubuntupackage>`_.

For the full list of supported distributions check
https://build.opensuse.org/package/show?package=gpaw&project=home%3Adtufys

.. note::

   If you prefer to install manually proceed to the next section, or
   alternatively, manually unpack the RPMS, e.g. (RHEL/CentOS 6)::

     # download the packages + dependencies (you can do that also manually!)
     $ yumdownloader --resolve gpaw
     # unpack into the current directory
     $ find . -name "*.rpm" | xargs -t -I file sh -c "rpm2cpio file | cpio -idm"
     # modify profile.d scripts
     $ find . -name "*.*sh" | xargs -t -I file sh -c "sed -i "s#PA=/usr#PA=$PWD/usr#" file"
     # make scripts executable
     $ find . -name "*.*sh" | xargs -t -I file sh -c "chmod u+x file"
     # source the scripts (example for bash)
     $ for f in `find . -name "*.sh"`; do source $f; done
     # test the installation
     $ gpaw-python -c "import gpaw; print gpaw.__file__, gpaw.mpi.rank"

.. _installationguide_developer:

Developer installation
----------------------

This is the **preferred** way of manually installing GPAW.
It offers the following advantages:

- installation is limited to standard user's account:
  it does not pollute the root filesystem,

- user gains access to svn updates, if necessary.

1) Perform :ref:`developer_installation`.

   .. note::

       If you install on a cluster,
       take a look at :ref:`install_custom_installation` - it provides
       installation instructions for different platforms.

2) Perform :ref:`installationguide_setup_files`.

3) :ref:`running_tests`.


.. _installationguide_standard:

Standard installation
---------------------

This is the standard way of installing python modules.
Avoid it as it does **not** offer advantages of
the :ref:`installationguide_developer`.

.. note::

   The standard installation must
   always be preceded by a well tested :ref:`installationguide_developer`!

1) :ref:`download` the code.

2) Go to the :file:`gpaw` directory::

     [~]$ cd gpaw

3) Install with the standard (using bash)::

     [gpaw]$ python setup.py install --home=<my-directory>  2>&1 | tee install.log

   and put :file:`{<my-directory>}/lib/python` (or
   :file:`{<my-directory>}/lib64/python`) in your :envvar:`PYTHONPATH` 
   environment variable.

   .. note::

     Usually :envvar:`HOME` is a good choice for :file:`{<my-directory>}`.

   Moreover, if :file:`setup.py` finds an ``mpicc`` compiler,
   a special :program:`gpaw-python` python-interpreter is created under
   :file:`{<my-directory>}/bin`.
   Please add :file:`{<my-directory>}/bin` to :envvar:`PATH`.
   Alternatively, the full pathname
   :file:`{<my-directory}>/bin/gpaw-python` can be used when executing
   parallel runs. See :ref:`parallel_installation` for more details about
   parallel runs.

   If you have root permissions, you can install GPAW system-wide
   (example below assumes bash)::

     [gpaw]# python setup.py install 2>&1 | tee install.log

4) :ref:`running_tests`.


Installation tricks
-------------------

.. _install_custom_installation:

Custom installation
+++++++++++++++++++

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
:ref:`installationguide_developer`.

.. _parallel_installation:


Installation with HDF5 support
++++++++++++++++++++++++++++++

HDF5 support can be enabled by setting in :file:`customize.py`::

 hdf5 = True

and, in this case, provide HDF5 `include_dirs`, `libraries`, and `library_dirs`
as described in :ref:`install_custom_installation`.


Parallel installation
+++++++++++++++++++++

By default, setup looks if :program:`mpicc` is available, and if setup
finds one, a parallel version is build. If the setup does not find
mpicc, a user can specify one in the :svn:`customize.py` file.

Additionally a user may want to enable ScaLAPACK, setting in
:file:`customize.py`::

 scalapack = True

and, in this case, provide BLACS/ScaLAPACK `libraries` and `library_dirs`
as described in :ref:`install_custom_installation`.

Instructions for running parallel calculations can be found in the
:ref:`user manual <manual_parallel_calculations>`.


.. _PGO:

Profile guided optimization
+++++++++++++++++++++++++++

Some compilers allow one to use
`profile guided optimization <http://en.wikipedia.org/wiki/Profile-guided_optimization>`_ (PGO).
See :ref:`PGO_gcc_EL5` for an example how use PGO to compile GPAW on CentOS.


.. _installationguide_setup_files:

Installation of setup files
---------------------------

1) Get the tar file :file:`gpaw-setups-{<version>}.tar.gz`
   of the <version> of setups from the :ref:`setups` page
   and unpack it somewhere, preferably in :envvar:`HOME`
   (``cd; tar zxf gpaw-setups-<version>.tar.gz``) - it could
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
   setting the location of setups.

   .. note::

     In case of several locations of setups the first found setup file is used.


.. _running_tests:

Run the tests
=============

Make sure that everything works by running the test suite (using bash)::

  [gpaw]$ gpaw-python `which gpaw-test` 2>&1 | tee test.log

This will a couple of hours.  If you have a multicore CPU, you
can speed up the test by using ``gpaw-test -j <number-of-cores>``.
This will run tests simultaneously (**not** employing MPI parallelization)
on the requested `<number-of-cores>`.
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

   Many MPI versions have their own `-c` option which may
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

  [examples]$ mpirun -np 2 gpaw-python ~/gpaw/test/CH4.py --gpaw=blacs=1 --sl_default=1,2,2

This will enable ScaLAPACK's diagonalization on a 1x2 BLACS grid
with the block size of 2.
