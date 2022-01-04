.. _mft:

===================
Magnon energies by magnetic force theorem
===================

A magnon is a quantum of a spin-wave in a magnetic material, the same way
phonons are quanta of lattice vibrations. Here we show how to compute
magnon energies in GPAW by way of the magnetic force theorem.

===================
Background theory
===================

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

===================
GPAW algorithm
===================

`B^{xc}` and `\chi^{-+}_{KS}` can be computed ab-initio with a
converged ground state as input. This is implemented in GPAW in a
plane-wave basis. Once ground state and `B^{xc}` have been computed once for
a given Monkhorst-Pack grid, `\chi^{-+}_{KS}(\mathbf{q})` and `K^{\mu}
(\mathbf{q})` can be computed `\mathbf{q}`-point by `\mathbf{q}`-point,
see flowchart.
However, `\chi^{-+}_{KS}(\mathbf{q})` is only computable at
wavevectors (`\mathbf{q}`-points) included in the Monkhorst-Pack grid.

.. image:: Flowchart_for_algorithm.pdf
    :width: 800 px

In addition to the standard ground state convergence parameters, there is an
energy cutoff, ``ecut``, for the number of `G`-vector components, plus the
shape and positions of the integration regions specifying magnetic sites.
Also, for the response calculation it is necessary to
converge a number of unoccupied bands.

The interface to all these calculations is the class

.. autoclass:: gpaw.response.mft