.. _unfolding_tutorial:

=======================================================================
Electronic Band Structure Unfolding for Supercell Calculations:Tutorial
=======================================================================

Brief theory overview
=====================
Due to the many electrons in the unit cell, the electronic band structure of a supercell (SC) calculation is in general quite messy.
However, supercell calculations are usually performed in order to allow for minor modification of the crystal structure, i.e. defects,
distortions etc. In this cases it could be useful to investigate, up to what extent, the electronic structure of the non-defective is preserved.
To this scope, unfolding the band structure of the SC to the one of the primitive cell (PC) becomes handy.

Consider the case where the basis vectors of the SC and PC satisfy:

 .. math::
    \underline{\vec{A}} = \underline{M}\cdot\underline{\vec{a}},

where :math:`\underline{\vec{A}}` and :math:`\underline{\vec{a}}` are matrices with the cell basis vectors as rows and :math:`\underline{M}` the trasformation matrix.
As a general convention, capital and lower caseletters indicate quantities in the SC and PC respectively.
A similar relation holds in reciprocal space:
 
 .. math::
    \underline{\vec{B}} = \underline{M}^{-1}\cdot\underline{\vec{b}},

with obvious notation.
Now given a :math:`\vec{k}` in the PBZ there is only a :math:`\vec{K}` in the SBZ to which it folds into, 
and the two vectors are related by a reciprocal lattice vector :math:`\vec{G}` in the SBZ:

 .. math::
    \vec{k} = \vec{K}+\vec{G},

whereas the opposite is not true since a :math:`\vec{K}` unfolds into a subset of PBZ vectors :math:`\left{\vec{k}_i\right}`.

We are in general interested in finding the spectral function of the PC calculation starting from eigenvalues and eigenfunctions of the SC one.
Such a spectral function can be calculated as follow:

  .. math::
    A(\vec{k},\epsilon) = \sum_m P_{\vec{K}m}(\vec{k}) \delta(\epsilon_{\vec{K}m}-\epsilon).
 
Remember that :math:`\vec{k}` and :math:`\vec{K}` are related to each other and this is why we made :math:`\vec{K}` appear on the RHS. 
In the previous equation, :math:`P_{\vec{K}m}(\vec{k})` are the weights defined by:

  .. math::
    P_{\vec{K}m}(\vec{k}) = \sum_n |\langle\vec{K}m|\vec{k}n\rangle|^2

which give information about how much of the character :math:`|\vec{k}n\rangle` is preserved in :math:`|\vec{K}n\rangle`. 
From the expression above, it seems that calculating the weights requires the knowledge of the PC eigenstates.
However, V.Popescu and A.Zunger show in  [#Review]_ that weights can be found using only SC quantities according to:

  .. math::
    P_{\vec{K}m}(\vec{k}) = \sum_{sub\{\vec{G}\}} |C_{\vec{K}m}(\vec{G}+\vec{k}-\vec{K})|^2

where :math:`C_{\vec{K}m}` are the Fourier coefficients of the eigenstate :math:`|\vec{K}m\rangle` and :math:`sub\{\vec{G}\}`
a subset of reciprocal space vectors of the SC, specifically the ones that match the reciprocal space vectors of the PC.
This is the method implemented in GPAW and it works in 'realspace', 'lcao' and 'pw' with or without spin-orbit.

.. [#unfolding_theory] V. Popescu and A. Zunger
                      Extracting E versus :math:`\vec{k}` effective band structure from supercell calculations on alloys and impurities
		      *Phys. Rev. B* **85**, 085201 (2012)


Band Structure Unfolding for MoS:math:`_2 3\times3` supercell with single sulfur vacancy
========================================================================================


Groundstate calculation
-----------------------

First, we need a regular groundstate calculation for a `3\times3`
supercell with a sulfur vacancy.  For supercell calculation it is
convenient to use 'lcao' with a 'dzp' basis.

.. literalinclude:: gs_3x3_defect.py

It takes a few minutes if run in parallel. 
The last line in the script creates a .gpw file which contains all the informations of the system, including the wavefunctions.


Defining the band path in the PBZ and finding the corresponding  :math:`\vec{K}` in the SBZ
-------------------------------------------------------------------------------------------

Next, we set up the path  in the PBZ along which we want to calculate the spectral function, we define the transformation matrix M,
and find the corresponding `\vec{K}` in the SBZ.

.. literalinclude:: unfold_3x3_defect.py
    :lines: 1-29

Non self-consistent calculation at the :math:`\vec{K}` points
-------------------------------------------------------------
Once we have the relevant :math:`\vec{K}` we have to calculate eigenvalues and and eigenvectors; we can do that self-consistently.

.. literalinclude:: unfold_3x3_defect.py
    :lines: 32-41

Unfolding the bands and calculating Spectral Function
-----------------------------------------------------

We are now ready to proceed with the unfolding. First we set up the 'unfold' class.

.. literalinclude:: unfold_3x3_defect.py
    :lines: 44-48

and then we call the function for calculating the spectral function.

.. literalinclude:: unfold_3x3_defect.py
    :lines: 50

This can be run in parallel over kpoints, and you may want to do that since the supercell is relatively big.

.. note::
    The function produces two outputs, 'weights_3x3_defects.pckl' containing the eigenvalues :math:`\epsilon_{\vec{K}m}` and the weights :math:`P_{\vec{K}m}(\vec{k})`
    and 'sf_3x3_defects.pckl' which contains the spectral function and the corresponding energy array.


Plotting Spectral Function
---------------------------
Finally you can plot the spectral function using an utility function included in bands_unfolding.py

.. literalinclude:: unfolding_3x3_defect.py
    :lines: 52-56

which will produce the following image:

.. image:: sf.png
    :height: 400 px

where you can clearly see the defect state in the gap!
