.. module:: gpaw.new.spinspiral
.. _spinspiral calculations:

========================
Spin spiral calculations
========================

.. warning::

   This tutorial is *work in progress*

In this tutorial we employ the Generalized Bloch's Theorem approach to
calculate the spin spiral ground-state [#Knöpfle]_. In this approach we can
choose any wave vector of the spin spiral `q`, and rotate the spin degrees
through the periodic boundary conditions accordingly. This rotation can be
included in Bloch's theorem by applying a combined translation and spin
rotation to the wave function at the boundaries. Then we get the generalized
Bloch's theorem,

.. math::

   \phi(\mathbf{k}, \mathbf{r}) =
   e^{i\mathbf{k} \cdot \mathbf{r}}
   [e^{-i\mathbf{q} \cdot \mathbf{r}/2} F(\mathbf{k}, \mathbf{r}),
    e^{i\mathbf{q} \cdot \mathbf{r}/2} G(\mathbf{k}, \mathbf{r})]^T

With two new spin Bloch functions `F(\mathbf{k}, \mathbf{r})` and
`G(\mathbf{k}, \mathbf{r})` replacing the regular Bloch function
`u(\mathbf{k}, \mathbf{r})`. There are some limitations associated with these
wave functions, because the spin structure should decouple from the lattice
such that the density matrix is invariant under the spin rotation.
In order for this to be the case, we can only apply spin orbit coupling
perturbatively, and not as part of the self consistent calculation.
Furthermore, with the density being invariant under the this spin rotation, so
will also the z-component of the magnetization. This can be understood by
looking at the  magnetization density `\tilde{\rho} = I_2\rho + \sigma\cdot\m`
under the spin spiral rotation, where one sees that the entire diagonal is
left invariant. Thus we are limited to spiral structures which have
magnetization vectors

.. math::

   \hat{e} = [\cos(\mathbf{q} \cdot \mathbf{r}),
              \sin(\mathbf{q} \cdot \mathbf{r}),
              0]^T

which are called flat spin spirals, because they always rotate in the
`xy`-plane. However, there is nothing special about the `xy`-plane, since
spin-orbit is neglected at this stage, the spin spiral is invariant under any
global rotation. The reward is that we can simulate any incommensurate spin
spiral of this type in the principle unit cell. Additional care does need to
be taken when taking structures with multiple magnetic atoms within the unit
cell. This is because we only modify the boundary condition of the
self-consistent calculation;
the magnetization within the unit cell handled as a
regular non-collinear magnetization density. For example, with two magnetic
atoms in the unit cell, such as Cr2I6, one could consider parallel, anti-
parallel or any canted alignment between the two Cr atoms on top of the spin
spiral structure. In practice, canted order or ferrimagnetic order can be
found self- consistently however finding anti-ferromagnetic order from a
ferromagnetic starting point seem unlikely. Thus in order to find most spin
spiral structures, one should in run calculations with both collinear
starting structures.


Ground state of FCC Fe
======================

At high temperatures, elementary iron has a phase transition to the iron
allotrope :math:`{\gamma}-Fe` which has a FCC lattice. The spin structure of
:math:`{\gamma}-Fe` was measured by stabilizing the phase at lower
temperatures using Co. [#Tsunoda]_ They found a spin spiral ground state with
wave vector `q_{exp}=\frac{1}{5}XW = \frac{2\pi}{a}(1, 0, 1/10)` at an atomic
volume of `{\Omega}=11.44 Å^3`.

DFT simulations of :math:`{\gamma}-Fe` have found this system to be extremely
sensitive to the lattice parameter. In fact we do not find the experimental
spin spiral at the experimental volume. Instead we construct a FCC crystal
with a slightly smaller unit cell of `{\Omega}=10.72 Å^3`. The
following script :download:`fe_sgs.py` (Warning, requires HPC resources) will
construct the Fe FCC lattice and calculate the spin spiral ground-states with q
along the high symmetry axis in the reciprocal lattice.

.. literalinclude:: fe_sgs.py

As a result we find a spectrum with two local minimum, one of which match the
experimentally measured spin spiral. Since only one atom is present in the
unit cell, we do not need to worry about any magnetic structure inside the
unit cell.

.. figure:: e-spiral.png
.. figure:: m-spiral.png

   (see :download:`plot.py`)

Calculating the energy of the spin spiral ground state could be done using a
(2, 1, 10) supercell of the iron lattice in a standard non-collinear ground
state calculation. It would however be difficult to verify the local minimum
since wave vectors close to the minimum are incommensurate with the unit
cell, and so a huge supercell would be required.


.. [#Tsunoda] Y. Tsunoda 1989 J. Phys.: Condens. Matter 1 10427

.. [#Knöpfle] K. Knöpfle, L. M. Sandratskii, and J. Kübler
   Spin spiral ground state of γ-iron,
   Phys. Rev. B 62, 5564 – Published 1 September 2000
   :doi:`10.1103/PhysRevB.62.5564`
