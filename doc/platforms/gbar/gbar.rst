========
DTU GBar
========

This document shows how to build a *venv* for GPAW+ASE.

.. seealso::

    * :mod:`Creation of Python virtual environments <venv>`.
    * Information about the `GBar <http://www.gbar.dtu.dk/>`_.
    * MyQueue_.


.. _MyQueue: https://myqueue.readthedocs.io/
.. highlight:: bash


Creating the venv
=================

Download the ``gpaw-venv.sh`` script
using this link: :download:`gpaw-venv.sh` or these commands_:

    $ gpaw=https://gitlab.com/gpaw/gpaw
    $ wget $gpaw/-/raw/master/doc/platforms/gbar/gpaw-venv.sh

and run it like this::

    $ ./gpaw-venv.sh <venv-name>
    ...

After a few minutes, you will have a ``<venv-name>`` folder with
a GPAW installation inside.

In the following, we will assume that your venv folder is ``~/venv/``.


Using the venv
==============

The venv needs to be activated like this::

    $ source ~/venv/bin/activate

and you can deactivate it when you no longer need to use it::

    $ deactivate


Submitting jobs
===============

Using bsub
----------

See `here <http://www.gbar.dtu.dk/>`_.


Using MyQueue
-------------

First, configure MyQueue_::

    $ mq config --in-place -Q hpc lsf

Then you can submit jobs with::

    $ mq submit script.py -R8:4h  # 8 cores, 5 hours
    $ mq ls
    $ mq --help
