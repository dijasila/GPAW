.. _miniconda:

==================
Miniconda on macOS
==================

These instructions are for Intel-based Macs running macOS Big Sur (11.2). Big Sur uses uses zsh 
as the default shell, but you can revert to use bash if desired.

.. highlight:: zsh

Get the Xcode Command Line Tools with the command::

    $ xcode-select --install

Install Miniconda::

    $ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    $ bash ~/Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda
    $ $HOME/miniconda/bin/conda init zsh
    $ source ~/.zshrc

Configure conda and create an environment::

    $ conda config --set auto_activate_base false
    $ conda create --name gpaw
    $ conda activate gpaw

Install ASE dependencies::

    $ conda install tk
    $ conda install numpy scipy matplotlig
    $ conda install pytest pytest-mock

Install ASE development version from git::

    $ git clone https://gitlab.com/ase/ase.git
    $ cd ase
    $ pip3 install --editable .

NOTE: For latest stable ASE version instead, simply use::

    $ pip3 install --upgrade ase

Run ASE tests::

    $ ase test

Install GPAW dependencies::

    $ conda install -c conda-forge libxc openmpi-mpicc fftw scalapack

Download GPAW development version from git::

    $ git clone https://gitlab.com/gpaw/gpaw.git

Configure GPAW for FFTW and Scalapack::

    $ cd gpaw
    $ cp siteconfig_example.py siteconfig.py

Use these in the :ref:`siteconfig.py <siteconfig>` file:

.. literalinclude:: siteconfig.py

Install GPAW::

    $ pip3 install --editable .

Alternatively for latest stable release::

    $ pip3 install --upgrade gpaw

Install GPAW setups::

    $ cd ~
    $ curl -O https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.20000.tar.gz
    $ tar -zxf gpaw-setups-0.9.20000.tar.gz
    $ echo 'export GPAW_SETUP_PATH=~/gpaw-setups-0.9.20000' >> ~/.zprofile
    $ source ~/.zprofile

Run GPAW tests::

    $ pip3 install pytest-xdist
    $ gpaw test
    $ gpaw -P 4 test
    $ pytest -n 4 --pyargs gpaw
