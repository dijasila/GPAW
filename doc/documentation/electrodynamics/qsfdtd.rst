.. _qsfdtd:

================================================
Quasistatic Finite-Difference Time-Domain method
================================================
The optical properties of all materials depend on how they
respond (absorb and scatter) to external electromagnetic fields.
In classical electrodynamics, this response is described by
the Maxwell equations. One widely used method for solving them
numerically is the finite-difference time-domain (FDTD)
approach. \ [#Taflove]_.
It is based on propagating the electric and magnetic fields
in time under the influence of an external perturbation (light)
in such a way that the observables are expressed in real space
grid points. The optical constants are obtained by analyzing
the resulting far-field pattern. In the microscopic limit of
classical electrodynamics the quasistatic approximation is
valid and an alternative set of time-dependent equations for
the polarization charge, polarization current, and the
electric field can be derived.\ [#coomar]_

The quasistatic formulation of FDTD is implemented in GPAW.
It can be used to model the optical properties of metallic
nanostructures (i) purely classically, or (ii) in combination with
:ref:`timepropagation`, which yields :ref:`hybridscheme`.

.. TODO: a schematic picture of classical case and hybrid case

-------------------------
Quasistatic approximation
-------------------------
The quasistatic approximation of classical electrodynamics
means that the retardation effects due to the finite speed
of light are neglected. It is valid at very small length
scales, below ~50 nm.


------------
Permittivity
------------

In the current implementation, the permittivity of the classical material is parametrized as

.. math::

    \epsilon(\mathbf{r}, \omega) = \epsilon_{\infty} + \sum_j \frac{\epsilon_0 \beta_j(\mathbf{r})}{\bar{\omega}_j^2(\mathbf{r})-\mbox{i}\omega\alpha_j(\mathbf{r})-\omega^2},

where :math:`\alpha_j, \beta_j, \bar{\omega}_j` are
fitted to reproduce the experimental permittivity.
For gold and silver they can be found in Ref. \ [#Coomar]_.
Permittivity defines how classical charge density polarizes
when it is subject to external electric fields.
The time-evolution for the charges in GPAW is performed with
the leap-frog algorithm, following Ref. \ [#Gao]_.

----------------
Optical response
----------------
The QSFDTD method can be used to calculate the optical photoabsorption
spectrum just like in :ref:`timepropagation`:
The classical charge density is first perturbed with an instantaneous
electric field, and then the time dependence of the induced dipole moment
is recorderd. Its Fourier transformation gives the photoabsorption spectrum.

-------------------------------------------
Example: photoabsorption of gold nanosphere
-------------------------------------------
This example calculates the photoabsorption spectrum of a nanosphere
that has a diameter of 10 nm, and compares the result with analytical
Mie scattering limit. The QSFDTD is implemented in a specific Poisson
solver. The real-space grid is defined when initializing the
FDTDPoissonSolver, and the actual time propagation is embedded in the
:ref:`timepropagation` routine.

.. literalinclude:: gold_nanosphere_calculate.py

Note that one empty KS-orbital was included in the calculation.
Also note that the permittivity was initialized as PermittivityPlus,
where Plus indicates that a renormalizing Lorentzian term is included:
this extra term brings the static limit to vacuum value, i.e.,
:math:`\epsilon(\omega=0)=\epsilon_0`, see Ref. \ [#Sakko]_ for
detailed explanation.
The above script generates the photoabsorption spectrum and compares
it with analytical formula of the Mie theory:

.. math::
    S(\omega) = \frac{3V\omega}{2\pi^2}\mbox{Im}\left[\frac{\epsilon(\omega)-1}{\epsilon(\omega)+2}\right],

where *V* is the nanosphere volume:

|qsfdtd_vs_mie|

.. |qsfdtd_vs_mie| image:: qsfdtd_vs_mie.png

The localized surface plasmon resonance (LSPR) at 2.5 eV is
nicely reproduced. The shoulder at 1.9 eV and the stronger
overall intensity are examples of the inaccuracies of the
used discretization scheme: the shoulder
originates from spurious surface scattering, and the intensity from
the larger volume of the nanosphere defined in the grid.

-----------
Limitations
-----------

-The scattering from the spurious surfaces of materials, which
are present because of the representation of the polarizable
material in uniformly spaced grid points, can cause unphysical
broadening of the spectrum.

-Nonlinear response (hyperpolarizability) of the classical
material is not supported, so do not use too large external
fields. In addition to nonlinear media, also other special
cases (nonlocal permittivity, natural birefringence, dichroism,
etc.) are not enabled by the present implementation.

-The frequency-dependent permittivity of the classical material must be
represented as a linear combination of Lorentzian oscillators. Other
forms, such as Drude terms, should be implemented in the future. Also,
the high-frequency limit must be vacuum permittivity. Future
implementations should get rid of also this limitation.

-----------------
Technical remarks
-----------------

-Double grid technique
-Parallelizatility: domain parallelization only
-Multipole corrections to Poissonsolver

----
TODO
----

-dielectrics (:math:`epsilon_{\infty}\neq\epsilon_0`)
-subcell averaging
-full FDTD (retardation effects) or interface to an external FDTD software


----------
References
----------

.. [#Taflove] A. Taflove and S. Hagness,
            Computational Electrodynamics: The Finite-Difference Time-Domain Method (3rd ed.),
            Artech House, Norwood, MA (2005).

.. [#Coomar] A. Coomar, C. Arntsen, K. A. Lopata, S. Pistinner and D. Neuhauser,
            Near-field: a finite-difference time-dependent method for simulation of electrodynamics on small scales,
            *J. Chem. Phys.* **135**, 084121 (2011)

.. [#Gao] Y. Gao and D. Neuhauser,
            Dynamical quantum-electrodynamics embedding: Combining time-dependent density functional theory and the near-field method
            *J. Chem. Phys.* **137**, 074113 (2012)

.. [#Sakko] A. Sakko, T. P. Rossi and R. M. Nieminen,
            Dynamical coupling of plasmons and molecular excitations by hybrid quantum/classical calculations: time-domain approach
            *J. Phys.: Condens. Matter **XX**, XXXXXX (2014)
