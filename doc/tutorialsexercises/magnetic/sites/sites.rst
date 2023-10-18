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

For any functional of the (spin-)density `f[n, \mathbf{m}](\mathbf{r})`,
one may define a corresponding site quantity,

.. math::
   f_a = \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a})
   f[n,\mathbf{m}](\mathbf{r}).

GPAW supplies functionality to compute such site quantities defined based on
*local* functionals of the spin-density for collinear systems,
`f[n,\mathbf{m}](\mathbf{r}) = f(n(\mathbf{r}),n^z(\mathbf{r}))`.
The implementation (using the PAW method) is documented in [#Skovhus]_.

In particular, the site magnetization,

.. math::
   m_a = \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a}) n^z(\mathbf{r}),

can be calculated via the function ``calculate_site_magnetization``, whereas
the function ``calculate_site_spin_splitting`` computes the LSDA site spin
splitting,

.. math::
   \Delta_a^\mathrm{xc} = -2 \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a})
   B^\mathrm{xc}(\mathbf{r}) m(\mathbf{r}).

Example: Iron
-------------

In the script
:download:`Fe_site_properties.py`,
the site magnetization and spin splitting are calculated from the ground state
of bcc iron. The script should take less than 10 minutes on a 40 core node.
After running the calculation script, you can download and excecute
:download:`Fe_plot_site_properties.py`
to plot the site magnetization and spin splitting as a function of the
spherical site radius `r_\mathrm{c}`.

.. image:: Fe_site_properties.png
	   :align: center

Although there does not exist an *a priori* magnetic site radius `r_\mathrm{c}`,
we clearly see that there is a region, where the site spin splitting is constant
as a function of the radius, hence making `\Delta_a^\mathrm{xc}` a well-defined
property of the system in its own right.
However, the same cannot be said for the site magnetization, which continues to
varry as a function of the cutoff radius. This is due to the fact that the
interstitial region between the Fe atoms is slightly spin-polarized
anti-parallely to the local magnetic moments, resulting in a radius
`r_\mathrm{c}^\mathrm{max}` which maximizes the site magnetization. If one wants
to employ a rigid spin approximation for the magnetic site, i.e. to assume that
the direction of magnetization is constant within the site volume, it would be a
natural choice to use `r_\mathrm{c}^\mathrm{max}` to define the sites.


Site-based sum rules
====================

In addition to site quantities, one may also introduce site matrix elements,
that is, expectation values of functionals
`f(\mathbf{r})=f[n, \mathbf{m}](\mathbf{r})`
evaluated on specific spherical sites,

.. math::
   f^a_{n\mathbf{k}s,m\mathbf{k}+\mathbf{q}s'} = \langle \psi_{n\mathbf{k}s}|
   \Theta(\mathbf{r}\in\Omega_{a}) f(\mathbf{r})
   |\psi_{m\mathbf{k}+\mathbf{q}s'} \rangle.
   = \int d\mathbf{r}\: \Theta(\mathbf{r}\in\Omega_{a}) f(\mathbf{r})\,
   \psi_{n\mathbf{k}s}^*(\mathbf{r})
   \psi_{m\mathbf{k}+\mathbf{q}s'}(\mathbf{r}).

Similar to the site quantities, GPAW includes functionality to calculate site
matrix elements for arbitrary *local* functionals of the (spin-)density
`f(\mathbf{r}) = f(n(\mathbf{r}),n^z(\mathbf{r}))`, as documented in
[#Skovhus]_.

In particular, bla bla bla.


References
==========

.. [#Skovhus] T. Skovhus and T. Olsen,
           *publication in preparation*, (2024)
