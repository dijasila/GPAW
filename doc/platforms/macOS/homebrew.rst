.. _homebrew:

=================
Homebrew on macOS
=================

These instructions are for Intel-based Macs running macOS Big Sur (11.2). Big Sur uses uses zsh 
as the default shell, but you can revert to use bash if desired.

.. highlight:: zsh

Install the Xcode command line tools::

    $ xcode-select --install

Install Homebrew::

    $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    $ echo 'export PATH=/usr/local/bin:$PATH' >> ~/.zprofile
    $ source ~/.zprofile

Install Python with interface to Tcl/Tk (only possible with Python 3.9 with latest Homebrew(::

    $ brew install python-tk

(Optional: default to using Python 3, so 'pip' can be usead instead of 'pip3'::

    $ echo 'export PATH=/usr/local/opt/python/libexec/bin:$PATH' >> ~/.zprofile 
    $ source ~/.zprofile
)

Install dependencies required by numpy >=1.20::

    $ brew install openblas
    $ brew install lapack

(Optional: manually install ASE dependencies::

    $ pip3 install numpy scipy matplotlib
)

Install pytest::

    $ pip3 install pytest, pytest-mock

**Option 1:** Clone and install latest git development version of ASE::

    $ git clone https://gitlab.com/ase/ase.git
    $ cd ase
    $ pip3 install --editable .

**Option 2:** Alternatively for latest stable release::

    $ pip3 install --upgrade ase

Run ASE tests::

    $ ase test

Install GPAW dependencies::

    $ brew install libxc
    $ brew install open-mpi
    $ brew install fftw
    $ brew install scalapack

**Option 1:** Clone git development version of GPAW::

    $ git clone https://gitlab.com/gpaw/gpaw.git

Configure GPAW for FFTW and Scalapack::

    $ cd gpaw

Use this :ref:`siteconfig.py <siteconfig>` file (save in the gpaw root directory):

.. literalinclude:: siteconfig.py

Install editable GPAW with pip::

    $ pip3 install --editable .

**Option 2:** Alternatively for latest stable release of GPAW::

    $ pip3 install --upgrade gpaw

Install GPAW setups::

    $ cd ~
    $ curl -O https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.20000.tar.gz
    $ tar -zxf gpaw-setups-0.9.20000.tar.gz
    $ echo 'export GPAW_SETUP_PATH=~/gpaw-setups-0.9.20000' >> ~/.zprofile
    $ source ~/.zprofile

Install parallel pytest::

    $ pip3 install pytest-xdist

Test GPAW (here for 4 cores)::

    $ gpaw test
    $ gpaw -P 4 test
    $ pytest --pyargs gpaw -n 4

As of gpaw-21.1.1b1-6d0953a81b, a few tests can be expected to
segfault due to crashes with scipy/linalg/decomp.py.
