.. _mft:

=================================================
Magnon dispersion from the magnetic force theorem
=================================================

A magnon is a collective spin wave excitation carrying a single unit of spin
angular momentum. The magnon quasi-particles can be viewed as the bosons that
quantize the fluctuations in the magnetization of a crystal the same way that
phonons quantize lattice vibrations. Here we show how to compute the magnon
energy dispersion in GPAW, using the magnetic force theorem in a linear
response formulation.

Background theory
=================

Classical Heisenberg model
--------------------------

In the classical isotropic Heisenberg model, the magnetization of a crystal is
partitioned into distinct magnetic sites with individual magnetic moments `M_a`
and orientation of the moments `\mathbf{u}_{ia}` (unit vector). Here, `i` is
the unit cell index and `a` indexes the magnetic sublattices of the crystal.
The system energy is then parametrized in terms of the Heisenberg exchange
parameters `J_{ij}^{ab}` as follows:

.. math::

   E_{\mathrm{H}} = - \frac{1}{2} \sum_{i,j} \sum_{a,b} J_{ij}^{ab}
   \mathbf{u}_{ia} \cdot \mathbf{u}_{jb}.

From the exchange parameters, one can compute the magnon dispersion of the
system using linear spin wave theory. In particular, the magnon energies
`\hbar\omega_n(\mathbf{q})`, where `n` is the mode index, are given as a
function of the wave vector `\mathbf{q}` by the eigenvalues to the dynamic spin
wave matrix:

.. math::

   H^{ab}(\mathbf{q}) = \frac{g\mu_{\mathrm{B}}}{\sqrt{M_a M_b}}
   \left[\sum_c \bar{J}^{ac}(\mathbf{0}) \delta_{ab}
   - \bar{J}^{ab}(\mathbf{q})\right].

Here `\bar{J}^{ab}(\mathbf{q})` denotes the periodic part of the lattice
Fourier transform of the exchange parameters,

.. math::

   \bar{J}^{ab}(\mathbf{q}) = \sum_i J_{0i}^{ab}
   e^{i\mathbf{q}\cdot\mathbf{R}_i},

where `\mathbf{R}_i` refers to the Bravais lattice point corresponding to the
`i`'th unit cell.

Magnetic force theorem (MFT)
----------------------------
   
In reference [#Durhuus]_, it is shown how the Kohn-Sham Hamiltonian in the
local spin-density approximation can be mapped onto a classical isotropic
Heisenberg model based on the magnetic force theorem. To do so, it is assumed
that the ground state magnetization
`\mathbf{m}(\mathbf{r})=m(\mathbf{r}) \mathbf{u}(\mathbf{r})` can be treated
in the rigid spin approximation,

.. math::

   \mathbf{u}(\mathbf{r}) \simeq \sum_i \sum_a
   \Theta(\mathbf{r}\in\Omega_{ia}) \mathbf{u}_{ia},

where `\Theta(\mathbf{r}\in\Omega_{ia})` is a unit step function, which is
nonzero for positions `\mathbf{r}` inside the site volume `\Omega_{ia}`.
In this way, it is the geometry of the sublattice site volumes, that defines
what Heisenberg model the DFT problem is mapped onto, and it is not clear
*a priori* that there should exist a unique choice for these geometries.

Once the site geometries have been appropriately chosen, the exchange
constants can be calculated in a linear response formulation of the magnetic
force theorem,

.. math::
   
   \bar{J}^{ab}(\mathbf{q}) = - \frac{2}{\Omega_{\mathrm{cell}}}
   B^{\mathrm{xc}\dagger} K^{a\dagger}(\mathbf{q})
   \chi_{\mathrm{KS}}^{'+-}(\mathbf{q}) K^{b}(\mathbf{q}) B^{\mathrm{xc}},

where all quantities are given in a plane wave basis and matrix/vector
multiplication in reciprocal lattice vectors `\mathbf{G}` is implied. In
this MFT formula for the exchange constants, `\Omega_{\mathrm{cell}}`
denotes the unit cell volume, `B^{\mathrm{xc}}(\mathbf{r})
= \delta E_{\mathrm{xc}}^{\mathrm{LSDA}} / \delta m(\mathbf{r})`,
`\chi_{\mathrm{KS}}^{'+-}(\mathbf{q})` is the reactive part of the static
transverse magnetic susceptibility of the Kohn-Sham system, and so-called
sublattice site kernels,

.. math::

   K_{\mathbf{GG}'}^{a}(\mathbf{q}) = \frac{1}{\Omega_{\mathrm{cell}}}
   \int \mathrm{d}\mathbf{r}\:
   e^{-i(\mathbf{G} - \mathbf{G}' + \mathbf{q})\cdot\mathbf{r}}
   \Theta(\mathbf{r}\in\Omega_{a}),

have been introduced to encode the sublattice site geometries defined
though `\Omega_{a}=\Omega_{0a}`. For additional details, please refer to
[#Durhuus]_.


GPAW implementation
===================

In GPAW, the computation of MFT Heisenberg exchange constants is implemented
through the ``IsotropicExchangeCalculator``. The calculator is constructed
from a ``ChiKS`` instance, which is a separate calculator for computing the
Kohn-Sham transverse magnetic plane wave susceptibility:

.. math::

   \chi_{\mathrm{KS},\mathbf{GG}'}^{+-}(\mathbf{q}, \omega + i \eta)
   = \frac{1}{\Omega} \sum_{\mathbf{k}} \sum_{n,m}
   \frac{f_{n\mathbf{k}\uparrow} - f_{m\mathbf{k}+\mathbf{q}\downarrow}}
   {\hbar\omega - (\epsilon_{m\mathbf{k}+\mathbf{q}\downarrow}
   - \epsilon_{n\mathbf{k}\uparrow}) + i\hbar\eta}
     n_{n\mathbf{k}\uparrow,m\mathbf{k}+\mathbf{q}\downarrow}(\mathbf{G} +
     \mathbf{q}) n_{m\mathbf{k}+\mathbf{q}\downarrow,n\mathbf{k}\uparrow}(
     -\mathbf{G}' - \mathbf{q}).

Here, `\Omega` is the crystal volume, `\epsilon_{n\mathbf{k}s}` and
`f_{n\mathbf{k}s}` the Kohn-Sham eigenvalues and occupation numbers and the
plane wave pair densities,

.. math::

   n_{n\mathbf{k}s,m\mathbf{k}'s'}(\mathbf{G} + \mathbf{q}) =
   \int_{\Omega_{\mathrm{cell}}} \mathrm{d}\mathbf{r}\:
   e^{-i(\mathbf{G}+\mathbf{q})\cdot\mathbf{r}}
   \psi_{n\mathbf{k}s}^*(\mathbf{r}) \psi_{m\mathbf{k}'s'}(\mathbf{r})

are computed directly from the Kohn-Sham orbitals
`\psi_{n\mathbf{k}s}(\mathbf{r})`. For more details on the transeverse
magnetic susceptibility and the details on its GPAW implementation, please
refer to [#Skovhus]_. The ``ChiKS`` calculator evaluates the sum over
`\mathbf{k}`-points by point integration on the Monkhorst-Pack grid
specified by an input ground state DFT calculation. Because of this, it only
accepts wave vectors `\mathbf{q}` that are commensurate with the underlying
`\mathbf{k}`-point grid. Furthermore, it takes input arguments ``ecut``,
``nbands`` and ``eta`` in the constructor, specifying the plane wave energy
cutoff, number of bands in the band summation and frequency broadening
respectively.

The ``IsotropicExchangeCalculator`` uses the ``ChiKS`` instance from which
it is initialized to compute the reactive part of the susceptibility,

.. math::

   \chi_{\mathrm{KS},\mathbf{GG}'}^{'+-}(\mathbf{q}, \omega + i \eta)
   = \frac{1}{2} \left[
   \chi_{\mathrm{KS},\mathbf{GG}'}^{+-}(\mathbf{q}, \omega + i \eta)
   +
   \chi_{\mathrm{KS},-\mathbf{G}'-\mathbf{G}}^{-+}(-\mathbf{q},
   -\omega + i \eta) \right],

in the static limit `\omega=0` for a given wave vector `\mathbf{q}`.
With this in hand, it uses a supplied ``SiteKernels`` instance defining
the sublattice site geometries to compute the exchange constants
`\bar{J}^{ab}(\mathbf{q})`. At present, spherical, cylindrical and
parallelepipedic site kernel geometries are supported through the
``SphericalSiteKernels``, ``CylindricalSiteKernels`` and
``ParallelepipedicSiteKernels`` classes.

When using the GPAW code for computing MFT Heisenberg exchange constants,
please reference both of the works [#Durhuus]_ and [#Skovhus]_.


Example 1 (Introductory): bcc-Fe
================================

In this first example, we will compute the magnon dispersion of iron, which
is an itinerant ferromagnet with a single magnetic atom in the unit cell.

First, you should download the ground state calculation script
:download:`Fe_gs.py`
and run it using a cluster available to you. Resource estimate: 10
minutes on a 40 core node. The script will perform a LSDA ground state
calculation and store all its data to a file, ``Fe_all.gpw``.

Secondly, download and run the
:download:`Fe_mft.py`
script to perform the MFT calculation of the Heisenberg exchange
parameters. Resource estimate: 30 minutes on a 40 core node. The script
computes the exchange constants on the high-symmetry path G-N-P-G-H
using two different site geometries:

1) Spherical site volumes centered on the Fe atoms with varying radii.
2) Parallelepipedic site volumes filling out the entire unit cell.

After the calculation, the `\mathbf{q}`-point path, spherical radii
and exchange constants are stored in separate ``.npz`` files.

Now it is time to visualize the data. GPAW distributes functionality to
compute the magnon dispersion for a single site ferromagnet from its
isotropic exchange constants `\bar{J}(\mathbf{q})`, namely through the
method ``calculate_single_site_magnon_energies``. In the script
:download:`Fe_plot_magnons_vs_rc.py`,
the magnon energy of iron in the high-symmetry points N, P and H is
plotted as a function of the spherical site radii, resulting in the
following figure:

.. image:: Fe_magnons_vs_rc.png
	   :align: center

Although there does not exist a unique definition of the correct magnetic
site volumes, there clearly seems to be a range of spherical cutoff radii
`r_{\mathrm{c}}\in[1.0, 1.5]` in which the MFT magnon energy for a given
wave vector `\mathbf{q}` is well defined! It is not clear *a priori* that
there always exists such a range, why it should always be double-checked,
when performing MFT calculations.

Finally, we use the script
:download:`Fe_plot_magnon_dispersion.py`,
to plot the magnon dispersion along the entire band path for both of our
chosen site geometries:

.. image:: Fe_magnon_dispersion.png
	   :align: center

Even though we are showing the entire range of magnon energies for
`r_{\mathrm{c}}\in[1.0, 1.5]`, the spread is not visible on the frequency
scale of the actual magnon dispersion, why we can conclude that the MFT
magnon dispersion is well defined for the entire Brillouin Zone! This is
confirmed by the calculations using the parallelepipedic site volumes, which
yields identical results.


Example 2 (Advanced): hcp-Co
============================

In the second example we will consider hcp-Co, which is also an itinerant
ferromagnet, but this time with two magnetic atoms in the unit cell. This
means that we will have two magnetic sublattices and two magnon modes, the
usual acoustic Goldtone mode and an optical mode.

Again, we start of by calculating the LSDA ground state using the script
:download:`Co_gs.py`
(resource estimate: 20 minutes on a 40 core node). However, this time we do
not save the Kohn-Sham orbitals as they can take up a significant amount of
disc space (hundreds of GB) for large systems. Instead, we will recalculate
the orbitals as the first thing in the MFT calculation script
:download:`Co_mft.py`.
Typically, this will not take much extra time. In fact, it is (depending on
your hard disk/file system) sometimes faster, as file io can be a real
bottle-neck when working with hundreds of GBs of data.

In the remainder of this tutorial, the magnon spectrum of Co in the hcp
crystal structure is computed as an example. The example scripts (in the
order they should be run) are,
:download:`magnon_dispersion_plot.py`.
We map to a model with 2 magnetic sites, centered on the 2 atoms of the hcp
unit cell, which results in 2 magnon bands.

First step is to converge the ground state. This is done in
:download:`converge_gs.py`. Here certain parameters must be carefully chosen
to be consistent with subsequent response calculations.
As mentioned above only wavevectors in the Monkhorst Pack grid can be
simulated, so ``k`` determines which points and how many are computable.
For instance, the high-symmetry point `L = (1/2, 0, 1/2)` requires ``k`` to
be a multiple of 2. Incidentally all high symmetry points of the hcp lattice
are included if ``k`` is a multiple of 6. The plane-wave cutoff for the
ground state, ``pw``, should be larger than the response cutoff ``ecut``.
Also ``nbands_gs`` `>=` ``nbands_response`` `> N_{occupied}` is required to
have some converged unoccupied bands, where `N_{occupied} = 12` for Co
(hcp).

With ground state converged, we can look at how integration regions affect
magnon energies. In :download:`high_sym_pts.py` the magnon energy at all
high-symmetry points is computed with an integration sphere centered at each
atom. See the generated :download:`spts.json` file for coordinates of the
high-symmetry points. This is done for different integration sphere radii,
`r_c`. :download:`magnon_energy_plot.py` plots the result, which should
look like

.. image:: Co_magnons_vs_rc.png
	   :align: center

Here solid lines indicate the low energy magnon band and dashed lines the
high energy band. Note that the bands at A, K, L and H are degenerate.
The magnon energies go to 0 as the integration spheres shrink
to nothing and diverge as the spheres overlap substantially (nearest
neighbour distance is 2.5 Å). In between these extremes is a wide plateau
where the results are insensitive to `r_c`, which suggests that all the
relevant electron spins are included. `r_c` should be chosen in this plateau
. The same conclusions hold for cylindrical integration regions.

Now that we have appropriate parameters for the Heisenberg lattice, let's
compute the magnon spectrum. One could sample the whole Brillouin zone and
interpolate to get the full spectrum, but this is computationally expensive.
Instead we look at the magnon dispersion on straight lines running between
the high symmetry points. The calculation is in :download:`high_sym_path.py`
and the plotting in :download:`magnon_dispersion_plot.py`.

.. image:: Co_magnon_dispersion.png
	   :align: center

We note that the 2 bands are degenerate for the entire path K->H->A->L.
The endpoints were also degenerate in the previous calculation.
Also, there are parabolic dispersions around the A and Gamma points. The
fact that all magnon energies are positive indicates that Co(hcp) is stable
in the ferromagnetic state, at least against spin-rotations.

References
==========

.. [#Durhuus] F. L. Durhuus, T. Skovhus and T. Olsen,
           *Phys. Rev. B* **??**, ????? (2022)

.. [#Skovhus] T. Skovhus and T. Olsen,
           *Phys. Rev. B* **103**, 245110 (2021)
