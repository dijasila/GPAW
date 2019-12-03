=============
Ubuntu 18.04+
=============

Install these Ubuntu_ packages::

    $ sudo apt install python3-dev libopenblas-dev libxc-dev libscalapack-mpi-dev libfftw3-dev

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
