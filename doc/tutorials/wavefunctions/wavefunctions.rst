Obtaining all-electron wave functions
=====================================

To get from the pseudo (PS) wave function to the all-electron (AE )wave
function, a PAW correction needs to be added.  The correction will add the
cusp and all the wiggles neccesary for the wave function to be orthogonal to
all the frozen core states.  This tutorial show how to use the
:class:`~gpaw.utilities.wavefunctions.WaveFunctionInterpolator` class for
interpolating the PS wave function to a fine grid where the PAW
corrections can be represented:

.. autoclass:: gpaw.utilities.wavefunctions.WaveFunctionInterpolator

Here is the code for plotting some AE wave functions for a HLi dimer using a
PAW dataset for Li with a frozen 1s orbital:
    
.. literalinclude:: hli.py

.. figure:: HLi.png
