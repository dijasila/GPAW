.. _dcdft:

===============
Delta Codes DFT
===============

**Warning** the results on this page are preliminary!

Codes precision estimated for PBE exchange-correlation functional
on the database of bulk systems from http://molmod.ugent.be/DeltaCodesDFT.

The Delta values calculated based on the EOS parameters:
*V0* in Å**3/atom, bulk modulus (*B0*) in GPa, and
pressure derivative of the bulk modulus *B1* (dimensionless).
Percentage errors with respect to http://www.wien2k.at/.

GPAW
----

Calculated with: :svn:`gpaw/test/big/dcdft/pbe_gpaw_pw.py`.

EOS
+++

.. csv-table::
   :file: dcdft_pbe_gpaw_pw.csv

Delta precision measure
+++++++++++++++++++++++

Calculated accordingly to http://molmod.ugent.be/DeltaCodesDFT

.. csv-table::
   :file: dcdft_pbe_gpaw_pw_Delta.txt

Abinit
------

Abinit 5.4.4p, `GGA_FHI <http://www.abinit.org/downloads/psp-links/gga_fhi>`_
pseudopotentials, calculated with: :svn:`gpaw/test/big/dcdft/pbe_abinit_fhi.py`.

EOS
+++

.. csv-table::
   :file: dcdft_pbe_abinit_fhi.csv

Delta precision measure
+++++++++++++++++++++++

Calculated accordingly to http://molmod.ugent.be/DeltaCodesDFT

.. csv-table::
   :file: dcdft_pbe_abinit_fhi_Delta.txt
