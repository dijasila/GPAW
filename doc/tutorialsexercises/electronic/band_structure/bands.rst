.. _band exercise:

==============
Band structure
==============

Band diagrams are useful analysis tools.  Read :ref:`bandstructures` tutorial
and try to understand what it does.

As a next step, calculate the bandstructure of FCC silver. Here we should be
careful with the choice of exchange-correlation functional to get a good
description of the d-band, which is generally poorly described within LDA.
(Why do you think that is?).  Modify the script
:download:`bandstructure.py
<../bandstructures/bandstructure.py>` so that it will work for
Ag instead of Si.

.. image:: Ag.png

The bandstructure is plotted in the end of the script.  Where is the d-band
located? Experimentally it's found to be approximately 4 eV below the Fermi-
level.
