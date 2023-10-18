.. _sites:

=============================================
Local properties of individual magnetic sites
=============================================

It is almost always very useful to analyze magnetic systems in terms of the
individual magnetic sites of the crystal. In this tutorial, we illustrate how
to calculate individual site properties for the magnetic atoms in GPAW.

Since it is not well-defined *a priori* where one site ends and another begins,
GPAW supplies functionality to calculate the site properties as a function of
spherical radii `r_\mathrm{c}`. In this picture, the site properties are defined
in terms of integrals with unit step functions
`\Theta(\mathbf{r}\in\Omega_{a})`, which are nonzero only inside a sphere of
radius `r_\mathrm{c}` around the given magnetic atom `a`.

Local functionals of the spin-density
=====================================

For any functional of the spin-density `f[n^\uparrow,n^\downarrow](\mathbf{r})`,
one may define a corresponding site quantity,

.. math::
   f_a = \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a})
   f[n^\uparrow,n^\downarrow](\mathbf{r}).

Currently, GPAW supplies functionality to compute such site quantities
defined based on *local* functionals of the spin-density,
`f[n^\uparrow,n^\downarrow](\mathbf{r}) = f(n^\uparrow(\mathbf{r}),n^\downarrow(\mathbf{r}))`.

In particular, the site magnetization,

.. math::
   m_a = \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a})
   \left(n_\uparrow(\mathbf{r}) - n_\downarrow(\mathbf{r})\right),

can be calculated via the function ``calculate_site_magnetization``, whereas
the function ``calculate_site_spin_splitting`` computes the LSDA site spin
splitting,

.. math::
   \Delta_a^\mathrm{xc} = -2 \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a})
   B^\mathrm{xc}(\mathbf{r}) m(\mathbf{r}).
