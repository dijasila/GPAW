.. _juwels:

==================
juwels @ FZ-Jülich
==================

This installation tutorial was written for GPAW version 21.6.1b1.
In order to build GPAW manually two possibilities are provided:

Building in virtual environment
===================

First GPAW (and ASE and others) can be installed in a self contained virtual
environment. In order to install this version download :download:`gpaw-venv.sh`
and run it like this::

  bash gpaw-venv.sh <PATH_TO_INSTALL>

after the installation is complete you can load the virtual environment via::

  source <PATH_TO_INSTALL>/bin/activate

Building GPAW only from repo
===================

Alternatively GPAW can be installed separately if e.g. ASE is already
installed somewhere else. In order to perform such a installation first clone
the latest GPAW version in your directory of choice::

  git clone git@gitlab/gpaw/gpaw.git

After having downloaded the GPAW source code you can run::

    cd gpaw
    bash doc/platforms/Linux/Juwels/compile_gpaw_only.sh

This script should compile GPAW in the build directory. For running jobs with
this compilation it is recommended to create a bash script which sets all paths
needed for running GPAW e.g. the following file we call ``GPAW_env``::

    module purge
    module load StdEnv
    module load Python/3.8.5
    module load SciPy-Stack/2020-Python-3.8.5
    module load intel-para
    module load FFTW/3.3.8
    module load libxc/4.3.4
    module load ELPA/2020.05.001

    basepath=<YOUR_PATH_TO_COMPILED_CODES>

    export OMP_NUM_THREADS=1

    export PYTHONPATH=$basepath/gpaw:$PYTHONPATH
    export PATH=$basepath/gpaw/build/bin.linux-x86_64-3.8:$PATH

    export PYTHONPATH=$basepath/ase:$PYTHONPATH
    export PATH=$basepath/ase/bin:$PATH
    export PATH=$basepath/ase/tools:$PATH

    export GPAW_SETUP_PATH=$basepath/gpaw-setups-0.9.20000

This file can be sourced in a submission script or when GPAW/ASE is needed. If
ASE and/or the GPAW-setups are located in a different directory, ``$basepath``
needs to be substituted accordingly.

Running GPAW
==================

Note that in both ways of compiling GPAW the old ``gpaw-python`` was produced.
Hence, the easiest way to submit a job is to write a shell script containing
the following::

    #!/bin/bash
    #<SBATCH keys> (Don't forget the account key!)

    # if GPAW has been compiled in venv:
    source <PATH_TO_VENV>/bin/activate

    # if GPAW has been compiled as standalone
    source <PATH_TO_GPAW_env>/GPAW_env

    srun -n <CORES>*<NODES> -N <NODES>  gpaw-python <YOUR RUNSCRIPT> [OPTIONS]

The script can then be submitted via ``sbatch <SCRIPTNAME>``.
