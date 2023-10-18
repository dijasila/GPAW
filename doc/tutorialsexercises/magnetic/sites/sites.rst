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
