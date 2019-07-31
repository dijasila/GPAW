=============
Ubuntu 18.04+
=============

Install these Ubuntu_ packages::

    $ sudo apt install python3-dev libopenblas-dev liblapack-dev libxc-dev libscalapack-mpi-dev libfftw3-dev

Then install ASE_, Numpy and SciPy::

    $ python3 -m pip install ase --user

And finally, GPAW with ScaLAPACK and FFTW::

    $ wget https://pypi.org/packages/source/g/gpaw/gpaw-1.5.1.tar.gz
    $ tar -xf gpaw-1.5.1.tar.gz
    $ cd gpaw
    $ sed -i "s/scalapack = False/scalapack = True/" customize.py
    $ sed -i "s/fftw = False/fftw = True/" customize.py
    $ python3 setup.py install --user


.. _Ubuntu: http://www.ubuntu.com/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
