.. _Niflheim:

========
Niflheim
========

Information about the Niflheim cluster can be found at
`<https://wiki.fysik.dtu.dk/niflheim>`_.

.. highlight:: bash

Installing GPAW on all three architectures
==========================================

Here, we install the development versions of ASE and GPAW in ``~/ase`` and
``~/gpaw``.  First, make sure you have this in your ``~/.bashrc``:

.. literalinclude:: gpaw.sh

or, just add this line::

    . ~/gpaw/doc/platforms/Linux/Niflheim/gpaw.sh

Now, get the source code for ASE and GPAW::

    $ cd
    $ rm -rf gpaw ase
    $ git clone https://gitlab.com/ase/ase.git
    $ git clone https://gitlab.com/gpaw/gpaw.git

and compile GPAW's C-extension using the :download:`compile.sh` script::

    $ cd gpaw
    $ sh doc/platforms/Linux/Niflheim/compile.sh

Submit jobs to the queue with::

    $ gpaw sbatch -- -p xeon8 -N 2 -n 16 my-script.py

Type ``gpaw sbatch -h`` for help.


Using more than one version of GPAW
===================================

Here, we install an additional version of GPAW for, say, test runs::

    $ cd ~
    $ mkdir testing
    $ cd testing
    $ ... clone gpaw and compile ...

Add this to your ``~/.bashrc``::

    if [[ $SLURM_SUBMIT_DIR/ = $HOME/test-runs* ]]; then
        GPAW=~/testing
    fi

right before sourcing the ``gpaw.sh`` script mentioned above.
Now, SLURM-jobs submitted inside your ``~/test-runs/`` folder will use the
version of GPAW from the ``~/testing/`` folder.
