.. _conda-forge:

===============================
Conda-forge + Homebrew on macOS
===============================

These instructions are for Apple Silicon-based Macs (tested on the M1) running macOS Big Sur (11.2). 
Big Sur uses uses zsh as the default shell, but you can revert to use bash if desired.

.. highlight:: zsh

Get the Xcode Command Line Tools with the command::

    $ xcode-select --install

Install Miniconda::

    $ curl -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
    $ chmod u+x Miniforge3-MacOSX-arm64.sh
    $ ./Miniforge3-MacOSX-arm64.sh
    $ $HOME/miniconda/bin/conda init zsh
    $ source ~/.zshrc

Configure conda and create an environment::

    $ conda config --set auto_activate_base false
    $ conda create --name gpaw python=3.8
    $ conda activate gpaw

Install ASE dependencies including Python interface to Tcl/Tk::

    $ conda install tk
    $ conda install matplotlib
    $ conda install scipy=1.5.3
    $ conda install pillow
    $ conda install pytest pytest-mock

**Option 1:** Install git development version of ASE::

    $ git clone https://gitlab.com/ase/ase.git
    $ cd ase
    $ pip3 install --editable .

**Option 2:** Alternatively for the latest stable release of ASE::

    $ pip3 install --upgrade ase

Run ASE tests::

    $ ase test

Install Homebrew::

    $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Install GPAW dependencies::

    $ brew install libxc
    $ brew install fftw
    $ brew install open-mpi
    $ brew install scalapack

Set paths – check that these match your system::

    $ export C_INCLUDE_PATH=/opt/homebrew/Cellar/libxc/4.3.4_1/include
    $ export LIBRARY_PATH=/opt/homebrew/Cellar/libxc/4.3.4_1/lib
    $ export LD_LIBRARY_PATH=/opt/homebrew/Cellar/libxc/4.3.4_1/lib
    $ export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
    $ export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"

**Option 1:** Clone git development version of GPAW::

    $ git clone https://gitlab.com/gpaw/gpaw.git

Configure GPAW for FFTW and Scalapack::

    $ cd gpaw
    $ cp siteconfig_example.py siteconfig.py

Use these in the :ref:`siteconfig.py <siteconfig>` file:

.. literalinclude:: siteconfig.py

Install GPAW::

    $ pip3 install --editable .

Option 2:** Alternatively for the latest stable release of GPAW::

    $ pip3 install --upgrade gpaw

Install GPAW setups::

    $ cd ~
    $ curl -O https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.20000.tar.gz
    $ tar -zxf gpaw-setups-0.9.20000.tar.gz
    $ echo 'export GPAW_SETUP_PATH=~/gpaw-setups-0.9.20000' >> ~/.zprofile
    $ source ~/.zprofile

Run GPAW tests (here for 4 cores)::

    $ conda install pytest-xdist
    $ gpaw test
    $ gpaw -P 4 test
    $ pytest --pyargs gpaw -n 4
