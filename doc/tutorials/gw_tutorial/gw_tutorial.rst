.. _gw_tutorial:

=========================================================
Quasi-particle spectrum in the GW approximation: tutorial
=========================================================

For a brief introduction to the GW theory and the details of its implementation in GPAW, see :ref:`gw_theory`.

**For GW calculations, the latest development release is required.**

This tutorial and the documentation will frequently be updated.


Quasi-particle spectrum of bulk silicon
=======================================


groundstate calculation
-----------------------

First, we need to do a regular groundstate calculation.
We do this in plane wave mode and choose the LDA exchange-correlation functional.
In order to keep the computational efforts small, we start with (3x3x3) k-points and a plane wave basis up to 200 eV.

.. literalinclude:: Si_groundstate.py

It takes a few minutes on a single CPU.
The last line in the script creates a .gpw file which contains all the informations of the system, including the wavefunctions.

.. note::

    You can change the number of bands to be written out by using ``calc.diagonalize_full_hamiltonian(nbands=...)``.
    This will be useful for higher plane wave cutoffs.

the GW calculator
-----------------

Next, we set up the GW calculator, where we define all the required parameters
as well as the k-point and band indices for which we want to calculate the quasi-particle spectrum.
Here, we do this for the complete irreducible Brioullin zone and 4 bands around the Fermi level
(silicon has 8 valence electrons and the bands are double occupied, starting from band index 0,
so the corresponding band indices are 2,3,4 and 5).

.. literalinclude:: Si_gw.py
    :lines: 1-12

calculating the exact exchange contributions
--------------------------------------------

It is highly recommended (though not necessary) to start with calculating the exact exchange contributions.
This is simply done by calling:

.. literalinclude:: Si_gw.py
    :lines: 14

In the output file, we find the results for non-selfconsistent Hartree-Fock,
sorted by spin, k-points (rows) and bands (columns).

.. note::

    By default, the results are stored in a pickle file called ``EXX.pckl``.
    The name can be changed by using ``gw.get_exact_exchange(file='myown_EXX.pckl')``.

Check for convergence with respect to the plane wave cutoff and number of k-points
by changing the respective values in the groundstate calculation and restarting.

.. image:: Si_EXX.png
       :height: 400 px

calculating the self-energy
---------------------------

Now, we are ready to calculate the GW quasiparticle spectrum by calling:

.. literalinclude:: Si_gw.py
    :lines: 16

While the calculation is running, timing information is printed in the output file.

In the end, the results for the quasiparticle spectrum are printed,
again sorted by spin, k-points (rows) and bands (columns).

.. note::

    By default, the results are stored in a pickle file called ``GW.pckl``.
    The name can be changed by using ``gw.get_QP_spectrum(file='myown_GW.pckl', exxfile='myown_EXX.pckl')``.

Repeat the calculation varying number of bands ``nbands`` and the planewave cutoff ``ecut``
and check how the results converge.

These two parameters are not independent of each other. For a proper convergence, the energy of the highest band
should be around the plane wave cutoff. This can be done automatically by setting ``ecut='npw'``,
so that the number of bands is set equally to the number of plane waves correspoding to the cutoff energy.

.. image:: Si_GW.png
       :height: 400 px

frequency dependence
--------------------

Next, we should check the quality of the Plasmon Pole Approximation and use the fully frequency-dependent dielectric matrix.

Remove the line ``ppa = True`` and insert ``w = np.array([50., 150., 0.05])``.
This creates a frequency grid which is linear from 0 to 50 eV with a spacing of 0.05 eV and increasing steps up to 150 eV.
This will correspond to ~1064 frequency points. The calculation takes about 1 hour.

The results should be very close to what we obtained within the Plasmon Pole Approximation, verifying its validity for this system.

At last, see how the results depend on the chosen frequency grid. It is important to have a very fine grid in the lower frequency range,
where the electronic structure of the system is more complicated.

.. image:: Si_w.png
       :height: 400 px

Good convergence is reached for :math:`\omega_\text{lin} \approx \omega_\text{max}/3` and :math:`\Delta\omega` = 0.05 eV.

.. note::

    If more memory is needed, use ``wpar=int`` to parallelize over frequency points. ``int`` needs to be an integer divisor of the available cores.
