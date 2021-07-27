.. _juwels:

==================
juwels @ FZ-Jülich
==================

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

This script should compile GPAW in the build directory.

Running GPAW
==================

Note that in both ways of compiling GPAW the old ``gpaw-python`` was produced.
Hence, the easiest way to submit a job is to write a shell script containing
the following::


    #!/bin/bash
    #<SBATCH keys> (Don't forget the account key!)

    source <PATH_TO_VENV>/bin/activate
    srun -n <CORES>*<NODES> -N <NODES>  gpaw-python <YOUR RUNSCRIPT> [OPTIONS]

The script can then be submitted via ``sbatch <SCRIPTNAME>``
