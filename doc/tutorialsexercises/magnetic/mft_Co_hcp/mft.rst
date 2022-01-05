.. _mft:

=========================================
Magnon energies by magnetic force theorem
=========================================

A magnon is a quantum of a spin-wave in a magnetic material, the same way
phonons are quanta of lattice vibrations. Here we show how to compute
magnon energies in GPAW by way of the magnetic force theorem.

=================
Background theory
=================

Using a classical, isotropic Heisenberg model, the magnon energy for a given
wavevector, `\mathbf{q}`, can be computed from the Heisenberg exchange
parameters `J(\mathbf{q})`. In order to map DFT calculations onto a Heisenberg
model, it is necessary to define a lattice of discrete magnetic moments. We
define the `\mu`'th moment in the magnetic unit cell as

.. math::

    \mathbf{M}_{\mu} = \int_{\Omega_{\mu}} \mathbf{m}(\mathbf{r})
    \mathrm{d}\mathbf{r}

where `\mathbf{m}(\mathbf{r})` is magnetisation density and
`\Omega_{\mu}` is a user-defined integration volume. Then, using linear
response theory and the magnetic force theorem, one can show that the
inter-site exchange is given by

.. math::

    J^{\mu\nu}(\mathbf{q}) = \sum_{\mathbf{G}_1\mathbf{G}_2\mathbf{G}_3
    \mathbf{G}_4}B^{xc}_{\mathbf{G}_1}
    K^{\nu}_{\mathbf{G}_1\mathbf{G}_2}(\mathbf{q})
    \chi^{-+}_{KS,\mathbf{G}_2\mathbf{G}_3}
    (\mathbf{q})K^{\mu*}_{\mathbf{G}_3\mathbf{G}_4}(\mathbf{q})
    B^{xc*}_{\mathbf{G}_4}

Here `B^{xc}` is the exchange-correlation B-field, `\chi^{-+}_{KS}` is the
reactive part of the static, transverse magnetic susceptibility for the
non-interacting Kohn-Sham system and `G_i` are reciprocal space vectors.
`K^{\mu}`, known as the site-kernel for site `\mu`, carries all information
about how the magnetic lattice is defined.
See [#Durhuus]_ for background and derivations. See [#Skovhus]_ for
how to compute `\chi^{-+}_{KS}` and other magnetic response functions
ab-initio.

Note that the present approach neglects Stoner excitations (adiabatic
approximation) and magnon-magnon scattering, so the output is the
adiabatic spectrum of non-interacting magnons. Also, the algorithm really
outputs `J^{\mu\nu}(\mathbf{q})` plus the Brillouin zone average of `J^{\mu\mu}
(\mathbf{q})`.

==============
GPAW algorithm
==============

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

==========================
Example : Co(hcp) spectrum
==========================

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

.. image:: high_sym_pts_energy_vs_rc.png
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

.. image:: Magnon_dispersion_Co_hcp.png
           :height: 500 px

We note that the 2 bands are degenerate for the entire path K->H->A->L.
The endpoints were also degenerate in the previous calculation.
Also, there are parabolic dispersions around the A and Gamma points. The
fact that all magnon energies are positive indicates that Co(hcp) is stable
in the ferromagnetic state, at least against spin-rotations.

References
==========

.. [#Durhuus] F. L. Durhuus, T. Skovhus, T. Olsen,
           *Phys. Rev. B* **??**, ????? (2022)

.. [#Skovhus] T. Skovhus and T. Olsen,
           *Phys. Rev. B* **103**, 245110 (2021)