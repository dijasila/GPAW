=================================
Range separated functionals (RSF)
=================================

Introduction
============

Range separated functionals (RSF) are a subgroup of hybrid 
functionals. While conventional hybrid functionals like PBE0
or B3LYP use fixed fractions of Hartree-Fock (HFT, E\ :sub:`XX`\ )
and DFT (E\ :sub:`X`\ )
exchange for exchange, f.e. 1/4 E\ :sub:`XX`\  and 3/4 E\ :sub:`X`\  in
the case of PBE0,
RSFs mix the two contributions by the distance between to points in
space by a soft function `\omega_\mathrm{RSF}`.

To achive this, the coulomb interaction kernel,
`\frac{1}{|r_1 - r_2|} = \frac{1}{r_{12}}`
which appears in the exchange integral from HFT is split into two parts:

`\frac{1}{r_{12}} = \underbrace{\frac{1 - [\alpha + \beta ( 1 - \omega_\mathrm{RSF} (\gamma, r_{12}))]}{r_{12}}}_{\text{SR, DFT}} + \underbrace{\frac{\alpha + \beta ( 1 - \omega_\mathrm{RSF} (\gamma, r_{12}))}{r_{12}}}_{\text{LR, HFT}}`, the short-range (SR) part is handled by the exchange from a (semi-)local LDA of GGA functional such as PBE, while the long-range part (LR) is handled by the exchange from HFT. `\alpha¸ and `\beta` are mixing parameters. `\alpha \ne 0` and `\beta = 0` resembles the conventional hybrids, RSFs with `\alpha = 0` and `\beta \ne 0` are denoted by ``LC`` and the name of the semi-local functional, f.e. LC-PBE, `\alpha \ne 0` and `\beta \ne 0` denotes RSFs handled by the coulomb attenuation method (CAM) scheme and were prefixed by ``CAM``, f.e. CAM-B3LYP.

For the separating function `\omega_\mathrm{RSF}`, two functions are in common use: either the complementary error function, `\omega_\mathrm{RSF} = \mathrm{erfc}(\gamma r_{12})` of the Slater-function, `\omega_\mathrm{RSF} = e^{(-\gamma r_{12})}`. While the use of the complementary error function is computationally fortunate for code utilizing Gaussian type basis sets, the Slater-function give superior results in the calculation of Rydberg-state and charge transfer excitations. To distinguish between these both functions, functionals using the Slater-function prefix the RSF marker by the letter "Y", f.e. LCY-PBE or CAMY-B3LYP, while functionals using the complementary error function stick on the scheme mentioned above.

Besides `r_{12}` both separation functions use a second parameter, the screening factor `\gamma`. The optional value for `\gamma` is under discussion, a density dependence is stated. For most RSF standard values for `\gamma` are defined, althought it is possible to tune `\gamma` to optimal values for calculations investigating ionization potentials, charge transfer excitations and the binding curves of bi-radical cations.

The implementation of RSFs in gpaw is based on the finite difference exact exchange code (hybrid.py) and therefore inherits its positive and negative sides, in summary:

 * self-consitent calculations using RSFs
 * calculations can only be done for the `\Gamma` point
 * only non-periodic boundary conditions can be used
 * only RMMDIIS can be used as eigensolver

 As one of the major benefits of the RSF is to retain the `\frac{1}{r}` asymptote of the potential, one has to use large boxes is neutral systems where considered. Large boxes start at `6\AA` vacuum around each atom.

Simple usage
============

In general calculations using RSF can simply be done choosing the appropriate
functional as in the following snippet:

.. literalinclude:: rsf_simple.py

Three main points can be seen allready in this small snippet. Even if choosing the RSF is quite simple by choosing ``xc=LCY_PBE``, one has to choose RMMDIIS as eigensolver, ``eigensolver=RMMDIIS()`` and has to decrease the convergence criteria a little.

Improving results
=================

However, there are a few drawbacks, at first in an SCF calclation the contributions from the core electrons are also needed, which have to be calculated during the generation of the PAW datasets. Second: for the calculation of the exchange on the Cartesian grid, the (screened) Poisson equation is solved numerically. For a charged system, as f.e. the exchange of a state with istself, on has to neutralize the charge by subtracting a Gaussian representing the "over-charge", solve the Poisson-Equation for the neutral system and add the solution for the Gaussian to the solution for the neutral system. However, if the charge to remove is "off-center", the center of the neutralizing charge should match the center of the "over-charge" preventing an artificial dipole. The next listing shows these two steps:


.. literalinclude:: rsf_setup_poisson.py

The generation of setups can also be done by ``gpaw-setup -f PBE -x --gamma=0.75 C O``

Tuning `\gamma`
===============

As stated in the introduction, the optimal value for `\gamma` is under discussion. One way to find the optimal value for `\gamma` for ionization potentials is to tune `\gamma` in a way, that the negative eigenvalue of the HOMO matches the calculated IP. To use different values of `\gamma`, one has to instantiate the RSF directly by ``HybridXC`` and give the value of `\gamma` to the variable ``omega`` (the latter was choosen to prevent poluting the code with variables (value for `\gamma` was choosen from paper):

.. literalinclude:: rsf_gamma.py
