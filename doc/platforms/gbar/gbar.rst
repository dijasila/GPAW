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

Download the :download:`gpaw-venv.sh` script and run it like this::

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

    $ python3 -m myqueue.config -q hpc

Then you can submit jobs with::

    $ mq submit script.py -R8:4h  # 8 cores, 5 hours
    $ mq ls
    $ mq --help
