==========================
Sophia cluster at HPC Risø
==========================



This document explains how to compile a developer version of GPAW on
Sophia.

.. seealso::

    * :mod:`Creation of Python virtual environments <venv>`.
    * Information about the Sophia cluster can be found at
      `<https://docs-devel.hpc.ait.dtu.dk/hardware/sophia/>`_.
      Please note that the Sophia web-page is only accessible from the DTU
      network.
    * `MyQueue <https://myqueue.readthedocs.io/>`__.


.. highlight:: bash

Creating the venv
=================

Download the :download:`sophia-venv.sh` script and run it like this::

    $ ./sophia-venv.sh <venv-name>
    ...

After a few minutes, you will have a ``<venv-name>`` folder with
a GPAW installation inside.


Using the venv
==============

In the following, we will assume that your venv folder is ``~/venv1/``.
The venv needs to be activated like this::

    $ source ~/venv1/bin/activate

and you can deactivate it when you no longer need to use it::

    $ deactivate

You will want the activation to happen automatically for the jobs you
submit to Sophia.  Here are three ways to do it:

1) If you always want to use one venv then just put the activation
   command in your ``~/.bashrc``.

2) If you only want jobs running inside a certain folder to use the venv,
   then add this to your ``~/.bashrc``::

       if [[ $SLURM_SUBMIT_DIR/ = $HOME/project-1* ]]; then
           source ~/venv1/bin/activate
       fi

   Now, SLURM-jobs submitted inside your ``~/project-1/``
   folder will use the venv.

3) Use the "automatic discovery of venv's" feature of MyQueue::

       $ cd ~/project-1
       $ ln -s ~/venv1 venv
       $ mq submit job.py

   MyQueue will look for ``venv/`` folders (or soft-links as in the example)
   in one of the parent folders and activate the venv automatically when
   your job starts running.


Full script
===========

.. literalinclude:: sophia-venv.sh
