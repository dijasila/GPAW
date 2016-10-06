==========================
Dipole corrections in GPAW
==========================

As an example system, a 2 layer `2\times2` slab of fcc (100) Al
is constructed with a single Na adsorbed on one side of the surface.

.. literalinclude:: dipole.py
    :lines: 1-14

.. image:: slab.png

The :func:`ase.build.fcc100` function will create a slab
with periodic boundary conditions in the xy-plane only and GPAW will
therefore use zeros boundary conditions for the the wave functions and
the electrostatic potential in the z-direction as shown here:

.. image:: zero.png

The blue line is the xy-averaged potential and the green line is the
fermi-level.

.. note::

    You need a bit of magic to get the electrostatic potential from a
    gpw file:

    >>> from ase.units import Hartree
    >>> from gpaw import GPAW
    >>> calc = GPAW('zero.gpw', txt=None)
    >>> calc.restore_state()
    >>> v = calc.hamiltonian.vHt_g * Hartree
    >>> v.shape
    (56, 56, 167)

If we use periodic boundary conditions in all directions:

.. literalinclude:: dipole.py
    :lines: 16-19

the electrostatic potential will be periodic and average to zero:

.. image:: periodic.png

In order to estimate the work functions on the two sides of the slab,
we need to have flat potentials or zero electric field in the vacuum
region away from the slab.  This can be achieved by using a dipole
correction:

.. literalinclude:: dipole.py
    :lines: 21-25

.. image:: corrected.png

.. warning::

    * Information about use of a dipole correction is currently not
      written to the gpw file.  See below how to restart such a
      calculation.

See the full Python script here: :download:`dipole.py`.  The script
used to create the figures in this tutorial is shown here:

.. literalinclude:: plot.py

.. autoclass:: gpaw.dipole_correction.DipoleCorrection
    :members:

Dipole Correction in plane wave mode
====================================
For the dipole correction in PW mode we use another example, namely 
a HCl molecule in a periodic cell.

In the following we calculate a single HCl structure using a plane 
wave basis set. 

.. literalinclude:: dipole_pw_nocorr.py 


Due to the dipole in the cell the electrostatic potential in the 
vacuum on both sides of the molecule is different and in order to
account for the periodicity the potentials approach each other 
linearly.

.. image:: dipole_pw_nocorr.png

In order to cancel the interaction between neighboring cells and to
be able to determine the two different work functions in our chosen 
directory ('z') we now apply a dipole correction:

.. literalinclude:: dipole_pw_corr.py 

Resulting in two different vacuum levels on both sides of the 
molecule.

.. image:: dipole_pw_corr.png

An evaluation of the method consisting of comparing the vacuum levels 
in the described cell with the electrostatic potential in a cell 
containing two mirrored HCl molecule can be calculated and seen here: 
:download:`dipole_pw.py`. The vacuum levels on the two sides of our 
single HCl molecule should correspond to the respective ones in the 
cell containing two HCl molecules. The resulting two electrostatic 
potentials look like this:

.. image:: dipole_pw_benchmark.png

The raise and decrease on the two sides of the small cells guarantee 
that the two vacuum levels still match at the boundary.

.. warning:: 
   * Up to now the electrostatic potential can not be retrieved by 
     restarting from a .gpw file, since it will be reset to be just
     0 everywhere.

