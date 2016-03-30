.. _gw_tutorial:

=========================================================
Quasi-particle spectrum in the GW approximation: tutorial
=========================================================

For a brief introduction to the GW theory and the details of its implementation in GPAW, see :ref:`gw_theory`.

More information can be found here:

    \F. Hüser, T. Olsen, and K. S. Thygesen

    `Quasiparticle GW calculations for solids, molecules, and two-dimensional materials`__

    Physical Review B, Vol. **87**, 235132 (2013)

    __ http://prb.aps.org/abstract/PRB/v87/i23/e235132


Quasi-particle spectrum of bulk silicon
=======================================
In the first part of the tutorial, the G0W0 calculator is introduced and the quasi-particle spectrum of bulk silicon is calculated. 

Groundstate calculation
-----------------------

First, we need to do a regular groundstate calculation.
We do this in plane wave mode and choose the LDA exchange-correlation functional.
In order to keep the computational efforts small, we start with (3x3x3) k-points and a plane wave basis up to 200 eV.

.. literalinclude:: Si_groundstate.py

It takes a few seconds on a single CPU.
The last line in the script creates a .gpw file which contains all the informations of the system, including the wavefunctions.

.. note::

    You can change the number of bands to be written out by using ``calc.diagonalize_full_hamiltonian(nbands=...)``.
    This will be useful for higher plane wave cutoffs.

The GW calculator
-----------------

Next, we set up the G0W0 calculator and calculate the quasi-particle spectrum for all the k-points in the irreducible Brillouin zone and the specified bands. In this case, silicon has 8 valence electrons and the bands are double occupied. Setting ``bands=(3,5)`` means including band index 3 and 4 which is the highest occupied band and the lowest unoccupied band.

.. literalinclude:: Si_gw.py

The calculation takes about 5 minutes on a single CPU and returns a dict with the following key/value pairs. The dictionary is stored in ``Si-g0w0_results.pckl``.

============    =====================    ===================================
key             type                     description
============    =====================    ===================================
``f``           ``(s, k, n)`` ndarray    Occupation numbers
``eps``         ``(s, k, n)`` ndarray    Kohn-Sham eigenvalues in eV
``vxc``         ``(s, k, n)`` ndarray    Exchange-correlation 
                                         contributions in eV
``exx``         ``(s, k, n)`` ndarray    Exact exchange contributions in eV
``sigma``       ``(s, k, n)`` ndarray    Self-energy contributions in eV
``Z``           ``(s, k, n)`` ndarray    Renormalization factors
``qp``          ``(s, k, n)`` ndarray    Quasi particle energies in eV
============    =====================    ===================================

From the dict it is for example possible to extract the direct bandgap:

.. literalinclude:: get_gw_bandgap.py

with the result: 3.18 eV.

The possible input parameters of the G0W0 calculator are listed below.

Parameters
---------------

===================  =================   ===================   ====================================
keyword              type                default value         description
===================  =================   ===================   ====================================
``calc``             ``str`` or          None                  gpw filename 
                     PAW object                                (with 'all' option when writing 
                                                               the gpw file)
``filename``         ``str``             ``'gw'``              Base filename of output files.
``kpts``             ``list``            None                  List of indices of the IBZ 
                                                               k-points to calculate the quasi 
			    				       particle energies for. 
							       None means entire IBZ.
``bands``            ``tuple``           None                  Range of band indices, like 
                                                               (n1 + n2+1). Second band index 
		 					       is not included.
``nbands``           ``int``             None                  Number of bands to include. 
                                                               None defaults to all available 
 	 						       from groundstate calculation.
``ppa``              ``bool``            False                 Use the plasmon pole approximation
                                                               or not.
``truncation``       ``str``             None                  Coulomb truncation scheme: 
                                                               None, wigner-seitz, 2D, 1D or 0D. 
``integrate_gamma``  ``int``             1                     q=0 integration scheme: 0, 1 or 2. 
``ecut``             ``float``           150.0 (eV)            Planewave energy cutoff.
``eta``              ``float``           0.1 (eV)              Broadening parameter.
``E0``               ``float``           27.21 (eV)            Energy used for fitting in the 
                                                               plasmon-pole approximation.
``domega0``          ``float``           0.025                 Minimum frequency step (in eV).
``omega2``           ``float``           10.0                  Equal to the frequency where the 
                                                               grid spacing has doubled in size.
``nblocks``          ``int``             1                     Number of blocks used in block 
                                                               parallellization.
``savew``            ``bool``            False                 Option to store the screened 
                                                               Coulomb interaction.
``savepckl``         ``bool``            True                  Option to store the resulting dict 
                                                               in a .pckl file.
===================  =================   ===================   ====================================

Convergence with respect to cutoff energy and number of k-points
-----------------------------------------------------------------
Can we trust the calculated value of the direct bandgap? Not yet.
Check for convergence with respect to the plane wave cutoff energy and number of k points is necessary. 
This is done by changing the respective values in the groundstate calculation and restarting. 
Script :download:`Si_ecut_k_conv_GW.py` carries out the calculations and :download:`Si_ecut_k_conv_plot_GW.py` plots the resulting data. It takes about 8 hours on 1 xeon-8 CPU (8 cores total). The resulting figure is shown below.

.. image:: Si_GW.png
       :height: 400 px

A k-point sampling of (9x9x9) and 150 eV plane wave cutoff seems to give results converged to within 0.05 eV. 


Frequency dependence
--------------------

Next, we should check the quality of the frequency grid used in the calculation. Two parameters determine how the frequency grid looks. ``domega0`` and ``omega2``. Read more about these parameters in the tutorial for the dielectric function :ref:`df_tutorial_freq`.

Running script :download:`Si_frequency_conv.py` calculates the direct band gap using different frequency grids with ``domega0`` varying from 0.005 to 0.05 and ``omega2`` from 1 to 25. The resulting data is plotted in :download:`Si_frequency_conv_plot.py` and the figure is shown below.

.. image:: Si_freq.png
       :height: 400 px

Converged results are obtained for ``domega0=0.02`` and ``omega2=10``, which is very close to the default values. 

Final results
-------------
A full G0W0 calculation at the values found above for the plane wave cutoff, number of k-points and frequency sampling results in a direct bandgap of 3.34 eV. Hence the value of 3.18 eV calculated at first was not converged!

Another method for carrying out the frequency integration is the Plasmon Pole approximation (PPA). Read more about it here :ref:`gw_theory_ppa`. This is turned on by setting ``ppa = True`` in the G0W0 calculator (see :download:`Si_converged_ppa.py`). Carrying out a full G0W0 calculation using the converged parameters and the PPA gives a direct band gap of xxx eV, which is in very good agreement with the result for the full frequency integration but the calculation took only xxx minutes on one CPU!

.. note::

    If a calculation is very memory heavy, it is possible to set ``nblocks`` to an integer larger than 1 but less than the amount of CPU cores running the calculation. With this, the response function is divided into blocks and each core gets to store a smaller matrix. 

Quasi-particle spectrum of two-dimensional materials
====================================================
TBA

GW0 calculations
================
TBA

Details of the G0W0 object
===========================

.. autoclass:: gpaw.response.g0w0.G0W0
   :members: calculate, calculate_q, calculate_sigma, calculate_screened_potential, 
             calculate_w 
