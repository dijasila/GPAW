.. _homebrew:

========
Homebrew
========

These instructions are for a developer installation with the latest dev version of GPAW from git. 
They have been updated for macOS 13.2 Big Sur, which uses zsh as the default shell, but it is 
straightforward to revert to use bash if desired.

.. highlight:: zsh

Get Xcode from the App Store and install it. You also need to install the
command line tools, do this with the command::

    $ xcode-select --install

Install Homebrew::

    $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    $ echo 'export PATH=/usr/local/bin:$PATH' >> ~/.zprofile

(Alternatively for bash::

    $ echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bash_profile
)

Install Python and the interface to Tcl/Tk::

    $ brew install python
    $ brew install python-tk

Make an alias to use brewed Python pip::

    $ alias pip3='/usr/local/bin/pip3'

Install dependencies required by numpy >=1.20::

    $ brew install openblas
    $ brew install lapack

Install ASE dependencies::

    $ pip3 install numpy, scipy, matplotlib

Install pytest::

    $ pip3 install pytest, pytest-mock

Clone latest git development branch for ASE::

    $ git clone https://gitlab.com/ase/ase.git

Install editable ASE with pip::

    $ cd ase
    $ pip3 install --editable .

(For latest stable release::

    $ pip3 install --upgrade ase
)

Run ASE tests::

    $ ase test

Install GPAW dependencies::

    $ brew install libxc
    $ brew install open-mpi
    $ brew install fftw
    $ brew install scalapack

Clone latest git development branch for GPAW::

    $ git clone https://gitlab.com/gpaw/gpaw.git

Configure GPAW for FFTW and Scalapack::

    $ cd gpaw
    $ cp siteconfig_example.py siteconfig.py

Use these in the :ref:`siteconfig.py <siteconfig>` file:

.. literalinclude:: siteconfig.py

Install editable GPAW with pip::

    $ pip3 install --editable gpaw

(For latest stable release::

    $ pip3 install --upgrade gpaw
) 

Install GPAW setups::

    $ cd ~
    $ curl -O https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.20000.tar.gz
    $ tar -xf gpaw-setups-0.9.20000.tar
    $ echo 'export GPAW_SETUP_PATH=~/gpaw-setups-0.9.20000' >> ~/.zprofile
    $ source ~/.zprofile

Install parallel pytest::

    $ pip3 install pytest-xdist

Test GPAW::

    $ gpaw test
    $ gpaw -P 4 test
    $ pytest -n 4 --pyargs gpaw
