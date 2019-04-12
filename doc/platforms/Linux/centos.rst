======
CentOS
======

Install these CentOS_ packages::

    $ yum install libxc-devel openblas-devel openmpi-devel fftw-devel
    $ yum install blacs-openmpi-devel scalapack-openmpi-devel

Add this to your ``~/.bashrc``::

    $ OPENMPI=/usr/lib64/openmpi
    $ export PATH=$OPENMPI/bin/:$PATH
    $ export LD_LIBRARY_PATH=$OPENMPI/lib:$LD_LIBRARY_PATH

Make sure you have the latest pip::

    $ python3 -m ensurepip --user
    $ python3 -m pip install pip --user

Then install ASE_, Numpy and SciPy::

    $ python3 -m pip install ase --user

And finally, GPAW with ScaLAPACK and FFTW::

    $ wget https://pypi.org/packages/source/g/gpaw/gpaw-1.5.1.tar.gz
    $ tar -xf gpaw-1.5.1.tar.gz
    $ cd gpaw
    $ cat > customize.py
    fftw = True
    scalapack = True
    libraries = ['fftw3', 'scalapack', 'mpiblacs']
    library_dirs.append('/usr/lib64/openmpi/lib/')
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
    ^D
    $ python3 setup.py install --user


.. _CentOS: http://www.centos.org/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
