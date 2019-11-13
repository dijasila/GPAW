.. _defect:

===========================
Defect scattering potential
===========================

This module allows to model the impact of defects on the electron dynamics
(e.g., quasiparticle and carrier scattering which affect, respectively, the
spectral function and transport) in disordered materials. 

Defect scattering potential

.. math::

    \hat{V}_i = V_\mathrm{def}(\hat{\mathbf{r}})
                - V_\mathrm{pris}(\hat{\mathbf{r}})

The full Bloch Hamiltonian in a basis of scalar relativistic states becomes

.. math::

    V_{\matbf{k}\mathbf{k'}}^{nn'}=
    \varepsilon_{nk\sigma}\delta_{nn'\sigma\sigma'}+
    \langle\psi_{nk\sigma}|H^{SO}(k)|\psi_{n'k\sigma'}\rangle=
    \varepsilon_{nk\sigma}\delta_{nn'\sigma\sigma'}+

where the spinors are chosen along the `z` axis as default. Thus, if calc is

    from gpaw.spinorbit import get_spinorbit_eigenvalues
    e_mk = get_spinorbit_eigenvalues(calc)

Here e_mk is an array of dimension (2 * Nb, Nk), where Nb is the number of

The script ``electrostatics.py`` takes the gpw files of the defective and
pristine calculation as input, as well as the gaussian parameters and
dielectric constant, and calculates the different terms in the correction
scheme. For this case, the calculated value of `E_{\mathrm{l}}` is -1.28 eV.

.. literalinclude:: electrostatics.py

.. math::

    s_{mk}\equiv\langle mk|\sigma_z|mk\rangle

and is useful for analyzing the degree of spin-orbit induced hybridization
between spin up and spin down states. Examples of this will be given below.
The implementation is documented in Ref. [#Olsen]_


Band structure of bulk Pt
=========================

every second spin-orbit band, since time-reversal symmetry along with
inversion symmetry dictates that all bands are two-fold degenerate (you can
check this for the present case). The plot is shown below.

.. image:: Pt_bands.png
           :height: 500 px

An important property of the spin-orbit interaction is the fact that it can
lift degeneracies between states that are protected by symmetry when spin-
orbit coupling is absent. This is well-known for the hydrogen atom where the

where `\xi_m` are the parity eigenvalues of Kramers pairs of occupied bands at
the parity invariant points `\Lambda_a`.

but in the present case only 4 are inequivalent. These are calcaluted with
the script :download:`high_sym.py` and the parity eigenvalues are
obtained with :download:`parity.py`. Note that the product of parity
eigenvalues at `\Gamma` changes from -1 to 1 when spin-orbit coupling is added
and the `\nu` thus changes from 0 to 1.


.. _magnetic anisotropy:

Magnetic anisotropy of hcp Co
=============================

As a final application of the spinorbit module we will calculate the magnetic
anisotropy of hcp Co. The idea is that the direction of spin polarization


.. [#Olsen] T. Olsen,
           *Phys. Rev. B* **94**, 235106 (2016)
.. [#Kane] M. Z. Hasan and C. L. Kane,
           *Rev. Mod. Phys.* **82**, 3045 (2010)
