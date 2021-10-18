.. _juwels:

==================
juwels @ FZ-Jülich
==================

This installation tutorial was written for GPAW version 21.6.1b1.
In order to build GPAW manually two possibilities are provided:

Building in a virtual environment
===================

First GPAW (and ASE and others) can be installed in a self contained virtual
environment. In order to install this version download :download:`gpaw-venv.sh`
and run it like this::

  bash gpaw-venv.sh <PATH_TO_INSTALL>

after the installation is complete you can load the virtual environment via::

  source <PATH_TO_INSTALL>/bin/activate

Running GPAW
==================

The easiest way to submit a job is to write a shell script containing
the following::

    #!/bin/bash
    #<SBATCH keys> (Don't forget the account key!)

    # if GPAW has been compiled in venv:
    source <PATH_TO_VENV>/bin/activate

    srun gpaw -P <Number of $CORES*$NODES> python <YOUR RUNSCRIPT> [OPTIONS]

The script can then be submitted via ``sbatch <SCRIPTNAME>``.
