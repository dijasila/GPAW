.. _developer installation:

======================
Developer installation
======================

Start by :ref:`forking and cloning as it is done for ASE develoment
<ase:contribute>`.  Let ``<repo>`` be the folder where you have cloned to
(could be ``~/gpaw``). Then do::

    $ cd <repo>
    $ python3 -m pip install --verbose --editable .

This will compile a shared library :file:`_gpaw.cpython-<version>-<platform>.so`
that contains GPAW's C-extension module.
