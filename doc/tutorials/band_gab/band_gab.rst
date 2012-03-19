.. _bandgab:

=========================
Calculating band gab using the GLLB-sc functional
=========================

In this tutorial, we use the GLLB-sc to calculate the band gab of KTaO3 using the 
XC functional GLLB-sc. This functional uses the GLLB response potential to 
replace the PBEsol response potential of the exchange. [GLLB-sc]
This has been shown to improve the band gab description as shown in the figure 
below taken from [Castilli2012].

.. figure:: sodium_bands.png

A GLLB-sc band gab calculation is performed as given here: 

.. literalinclude:: gllbsc_band_gab.py

-------------

.. [GLLB-sc] M. Kuisma, J. Ojanen, J. Enkovaara, and T. T. Rantala1,
   PHYSICAL REVIEW B 82, 115106 (2010)
   *Kohn-Sham potential with discontinuity for band gap materials*,
   DOI: 10.1103/PhysRevB.82.115106

   [Castilli2012] Ivano E. Castelli, Thomas Olsen, Soumendu Datta, David D. Landis, Søren Dahl, Kristian S. Thygesena
and Karsten W. Jacobsen
   Energy Environ. Sci., 2012, 5, 5814
   *Computational screening of perovskite metal oxides for optimal solar light
capture*
   DOI: 10.1039/c1ee02717d

