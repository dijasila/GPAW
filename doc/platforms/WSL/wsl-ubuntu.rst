=============
WSL Ubuntu 18.04+
=============
This guide assumes that you have a running Ubuntu_ installation in Windows Subsystem for Linux (WSL). If this installation is old, make sure that things are up-to-date::

    $ sudo apt update
    $ sudo apt upgrade

Install these Ubuntu_ packages::

    $ sudo apt install python3 python3-dev python3-pip python3-venv
    $ sudo apt install libopenblas-dev libxc-dev libscalapack-mpi-dev libfftw3-dev

Create a :ref:`siteconfig.py <siteconfig>` file::

    $ mkdir -p ~/.gpaw
    $ cat > ~/.gpaw/siteconfig.py
    fftw = True
    scalapack = True
    libraries = ['xc', 'blas', 'fftw3', 'scalapack-openmpi']
    ^D

Then install GPAW (and dependencies: ASE_, Numpy, SciPy)::

    $ pip install gpaw


.. _Ubuntu: http://www.ubuntu.com/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
