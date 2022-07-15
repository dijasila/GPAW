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
transverse magnetic Kohn-Sham susceptibility and so-called sublattice
site kernels,

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

`B^{xc}` and `\chi^{-+}_{KS}` can be computed ab-initio with a
converged ground state as input. This is implemented in GPAW in a
plane-wave basis. Once ground state and `B^{xc}` have been computed once for
a given Monkhorst-Pack grid, `\chi^{-+}_{KS}(\mathbf{q})` and `K^{\mu}
(\mathbf{q})` can be computed `\mathbf{q}`-point by `\mathbf{q}`-point
(flowchart in [#Durhuus]_).
However, `\chi^{-+}_{KS}(\mathbf{q})` is only computable at
wavevectors (`\mathbf{q}`-points) included in the Monkhorst-Pack grid.

In addition to the standard ground state convergence parameters, there is an
energy cutoff, ``ecut``, for the number of `G`-vector components, plus the
shape and positions of the integration regions specifying magnetic sites.
Also, for the response calculation it is necessary to
converge a number of unoccupied bands.

The interface to all these calculations is the class

.. autoclass:: gpaw.response.mft

Example 1 (Introductory): bcc-Fe
================================

Bla Bla Bla

Example 2 (Advanced): hcp-Co
============================

In the remainder of this tutorial, the magnon spectrum of Co in the hcp
crystal structure is computed as an example. The example scripts (in the
order they should be run) are
:download:`converge_gs.py`,
:download:`high_sym_pts.py`,
:download:`magnon_energy_plot.py`,
:download:`high_sym_path.py`,
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

.. image:: Fe_magnons_vs_rc.png
           :height: 500 px

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

.. image:: Fe_magnon_dispersion.png
           :height: 500 px

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
