.. _lumi:

============================
lumi.csc.fi
============================


.. note::
   These instructions are up-to-date as of January 2023.

GPAW for LUMI-C
===============

Following instructions are preliminary guidelines on how to install GPAW to be utilized with the
CPU only partition of LUMI-C.
To begin, add a following file to your current directory and call it `siteconfig-lumi.py`.

.. literalinclude:: siteconfig-lumi.py

Following script will create a Python virtual environment called gpaw-venv,
with latest GPAW and ASE master::
        module load cray-python/3.9.12.1
        module load PrgEnv-gnu/8.3.3
        python -m venv gpaw-venv

        # Prepend module loads (otherwise they will override venv bin folder as first)
        cp gpaw-venv/bin/activate temporary_activate
        echo "module load cray-python/3.9.12.1
        module load PrgEnv-gnu/8.3.3" > gpaw-venv/bin/activate
        cat temporary_activate >> gpaw-venv/bin/activate
        rm temporary_activate

        export CFLAGS='-fPIC -march=native -O3'
        export FFLAGS='-fPIC -march=native -O3'
        export GPAW_CONFIG=$(pwd)/siteconfig-lumi.py
        cd gpaw-venv
        source bin/activate
        pip install --upgrade pip
        git clone https://gitlab.com/ase/ase.git ase
        cd ase
        pip install --verbose -e .
        cd ..
        git clone https://gitlab.com/gpaw/gpaw.git gpaw
        cd gpaw
        pip install --verbose -e .
        cd ..
