.. _vdwcorrection:

========================
van der Waals correction
========================

A correction on top of the PBE functional has been proposed
by Tkatchenko and Scheffler [#TS09]_. While nearly all parameters
are obtained from ab-initio calculations, the method requires
nearly no additional computational cost and performs very well:

======================= ===  ===== ====== ======= ==== =========
.                       PBE  TPSS  vdW-DF vdW-DF2 TS09 Grimme D4
======================= ===  ===== ====== ======= ==== =========
Mean absolute deviation 115  154   76     48      16   14
RMS deviation           108  128   60     42      21   14
======================= ===  ===== ====== ======= ==== =========

Error in energies compared to CCSD results of the S26 test set.
All values in meV.
GPAW calculations were done with h=0.18 and at least 4 A vacuum.
The TS09 results are in good agreement to the results obtained with
the FHI-aims code [#Hanke11jcc]_. Grimme D4 is available
at github_.


Calculating the S26 test set
============================

As an example of the usage, here the S26 (S22 plus 4 other pairs)
test set is calculated:

.. literalinclude:: s26_set.py

.. [#TS09] :doi:`Tkatchenko and Scheffler Phys. Rev. Lett. 102 (2009)
                 <10.1002/jcc.21724>`
.. [#Hanke11jcc] Felix Hanke J. Comp. Chem. 32 (2011) 1424
.. _github: https://github.com/dftd4/dftd4
