.. program:: gpaw
.. highlight:: bash
.. index:: gpaw, command line interface, CLI

.. _cli:

======================
Command line interface
======================

GPAW has a command line tool called :program:`gpaw` with the following
sub-commands:

==============  =====================================================
sub-command     description
==============  =====================================================
help            Help for sub-command
run             Run calculation with GPAW
info            Show versions of GPAW and its dependencies
dos             Calculate (projected) density of states from gpw-file
gpw             Write summary of GPAW-restart file
completion      Add tab-completion for Bash
atom            Solve radial equation for an atom
python          Run GPAW's parallel Python interpreter
sbatch          Submit a GPAW Python script via sbatch
dataset         Calculate density of states from gpw-file
symmetry        Analyse symmetry
install-data    Install PAW datasets, pseudopotential or basis sets
==============  =====================================================


Help
====

You can do::

    $ gpaw --help
    $ gpaw sub-command --help

to get help (or ``-h`` for short).


Other command-line tools
========================

There are also CLI tools for analysing :ref:`point groups`
and for :ref:`hyperfine`.  Try::

    $ python3 -m gpaw.point_groups --help
    $ python3 -m gpaw.hyperfine --help

See also::

    $ python3 -m gpaw.utilities.dipole --help
    $ python3 -m gpaw.utilities.ekin --help


.. _bash completion:

Bash completion
===============

You can enable bash completion like this::

    $ gpaw completions

This will append a line like this::

    complete -o default -C /path/to/gpaw/gpaw/cli/complete.py gpaw

to your ``~/.bashrc``.
