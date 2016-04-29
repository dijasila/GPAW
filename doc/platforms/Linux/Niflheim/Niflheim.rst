.. _Niflheim:

========
Niflheim
========

Information about the Niflheim cluster can be found at
`<https://wiki.fysik.dtu.dk/niflheim>`_.


Installing GPAW on all three architectures
==========================================

Here, we install the development versions of ASE and GPAW in ``~/ase`` and ``~/gpaw``.  First, make sure you have this in your ``~/.bashrc``::

    #!/bin/bash
    if [ -f /etc/bashrc ]; then . /etc/bashrc; fi
    if [ -n $PBS_ENVIRONMENT ]; then cd $PBS_O_WORKDIR; fi
    GPAW=~/gpaw
    ASE=~/ase
    source /home/opt/modulefiles/modulefiles_el6.sh
    module load GPAW
    module load NUMPY/1.7.1-1
    module load SCIPY/0.12.0-1
    module load fftw
    PLATFORM="linux-x86_64-`echo $FYS_PLATFORM | sed 's/-el6//'`-2.6"
    export PATH=$GPAW/tools:$GPAW/build/bin.$PLATFORM:$PATH
    export PYTHONPATH=$GPAW:$GPAW/build/lib.$PLATFORM:$PYTHONPATH
    export PATH=$ASE/tools:$PATH
    export PYTHONPATH=$ASE:$PYTHONPATH
    export PATH=$GPAW/doc/platforms/Linux/Niflheim:$PATH
    
Now, get the source code for ASE and GPAW and compile GPAW's C-extension::

    $ cd
    $ rm -rf gpaw ase
    $ git clone https://gitlab.com/ase/ase.git
    $ git clone https://gitlab.com/gpaw/gpaw.git
    $ cd gpaw
    $ sh doc/platforms/Linux/Niflheim/compile.sh

Submit jobs to the queue with::
    
    $ gpaw-qsub -c 16 -q long my-script.py
    
Type ``gpaw-qsub -h`` for help.


Using more than one version of GPAW
===================================

Here we install an additional version of GPAW for, say, production runs::
    
    $ cd
    $ mkdir production
    $ cd production    
    $ git clone https://gitlab.com/gpaw/gpaw.git
    $ git checkout 1.0.1
    $ cd gpaw
    $ sh doc/platforms/Linux/Niflheim/compile.sh
    
Now you can submit jobs that use this production version with::

    $ gpaw-qsub -c 16 -q long -g ~/production my-script.py
