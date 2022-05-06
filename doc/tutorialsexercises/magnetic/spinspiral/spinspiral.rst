.. module:: gpaw.new.spinspiral
.. _spinspiral tutorial:

=========================
 Spin spiral calculations
=========================

.. warning::

   This tutorial is *work in progress*

In this tutorial we employ the Generalized Bloch's Theorem approach to
calculate the spin spiral groundstate. In this approach we can choose any wave
vector of the spin spiral `q`, and rotate the spin degrees through the
periodic boundary conditions accordingly. This rotation can be included in
Blochs theorem by applying a combined translation and spin rotation to the
wavefunction at the boundaries. Then we get the generalized Bloch's theorem,

.. math::

   \phi(\mathbf{k}, \mathbf{r}) =
   e^{i\mathbf{k} \cdot \mathbf{r}}
   [e^{-i\mathbf{q} \cdot \mathbf{r}/2} F(\mathbf{k}, \mathbf{r}),
    e^{i\mathbf{q} \cdot \mathbf{r}/2} G(\mathbf{k}, \mathbf{r})]^T

With two new spin Bloch functions `F(\mathbf{k}, \mathbf{r})` and
`G(\mathbf{k}, \mathbf{r})` replacing the regular Bloch function
`u(\mathbf{k}, \mathbf{r})`. There are some limitations associated with these
wavefunctions, because the spin structure should decouple from the lattice
such that the density matrix is invariant under the spin rotation.
In order for this to be the case, we can only apply spin orbit coupling
perturbatively, and not as part of the self consistent calculation.
Furthermore, with the density being invariant under the this spin rotation, so
will also the z-compenent of the magnetization. This can be understood by
looking at the  magnetization density `\tilde{\rho} = I_2\rho + \sigma\cdot\m`
under the spin spiral rotation, where one sees that the entire diagonal is
left invariant. Thus we are limited to spiral structures which have
magnetization vectors

.. math::

   \hat{e} = [cos(\mathbf{q} \cdot \mathbf{r}),
              sin(\mathbf{q} \cdot \mathbf{r}),
              0]^T

which are called flat spin spirals, because they always rotate in the
xy-plane. However, there is nothing special about the xy-plane, since spin-orbit
is neglected at this stage, the spin spiral is invariant under any global
rotation. The reward is that we can simulate any incommensurate spin spiral of
this type in the principle unit cell. Additional care does need to be taken
when taking structures with multiple magnetic atoms within the unit cell. This
is because we only modify the boundary condition of the self-consistent
calculation; the magnetization within the unit cell handled as a regular non-
collinear magnetization density. For example, with two magnetic atoms in the
unit cell, such as Cr2I6, one could consider parallel, antiparallel or any
canted alignment between the two Cr atoms on top of the spin spiral structure.
In practice, canted order or ferrimagnetic order can be found self-
consistently however finding antiferromagnetic order from a ferromagnetic
starting point seem unlikely. Thus in order to find most spin spiral
structures, one should in run calculations with both collinear starting
structures.


Ground state of fcc Fe
======================

At high temperatures, elementary iron has a phase transition to the iron
allotrope :math:`{\gamma}-Fe` which has a FCC lattice. The spin structure of
:math:`{\gamma}-Fe` was measured by stabilizing the phase at lower
temperatures using Co. [#Tsunoda]_ They found a spin spiral ground state with
wave vector `q_{exp}=\frac{1}{5}XW = \frac{2\pi}{a}(1, 0, 1/10)` at an atomic
volume of `{\Omega}=11.44\angstrom^3`.

DFT simulations of :math:`{\gamma}-Fe` have found this system to be extremely
sensitive to the lattice parameter. In fact we do not find the experimental
spin spiral at the experimental volume. Instead we construct a fcc crystal
with a slightly smaller unit cell of `{\Omega}=10.72\angstrom^3`. The
following script :download:`Fe_sgs.py` (Warning, requires HPC resources) will
construct the Fe fcc lattice and calculate the spin spiral groundstates with q
along the high symmetry axis in the reciprocal lattice. As a result we find a
spectrum with two local minimum, one of which match the experimentally
measured spin spiral. Since only one atom is present in the unit cell, we do
not need to worry about any magnetic structure inside the unit cell.

Calculating the energy of the spin spiral ground state could be done using a
(2, 1, 10) supercell of the iron lattice in a standard noncollinear ground
state calculation. It would however be difficult to verify the local minimum
since wave vectors close to the minimum are very incommensurate with the unit
cell, and so huge supercell would be required.


.. [#Tsunoda] Y. Tsunoda 1989 J. Phys.: Condens. Matter 1 10427
