.. _nlo_tutorial:

================================================
Nonlinear optical response of an extended system
================================================

A short introduction
=====================

The nonlinear optical (NLO) response of an extended system can be obtained 
from its ground state electronic structure.
Due to large computional cost, the local field effect effect is neglected.
Hence, the NLO response is only frequency-dependent. 
The NLO response can be descibed by nonlinear susceptibility tensors. 
There are numerious NLO processes, depending on the number of photons involved in the process.

The second harmonic generation (SHG) is a NLO process, in which two photons at the frequency of 
`\omega` generate a photon at the frequency of `2\omega`. The SHG response is characterized by
the second-order (quadratic) susceptibility tensor, defined via 

.. math:: P_{\gamma}(t) = \sum_{\alpha\beta} \chi_{\gamma\alpha\beta}^{(2)}(\omega,\omega) E_{\alpha}E_{\beta} e^{-2i\omega t}+\textrm{c.c.}

where `\alpha,\beta=\{x,y,z}` denotes the Cartezain coordinates, 
and  `E_{\alpha}` and `E_{\alpha}` are the polarization and electric 
fields, respectivly.


Example 1: SHG spectrum of semiconductor: Monolayer MoS2
============================================================

To compute the SHG spectrum of given structure, 3 steps are performed:

1. Ground state (GS) calculation

  .. literalinclude:: silicon_ABS.py
      :lines: 1-29

  In this script a normal ground state calculation is performed with fine
  kpoint grid. 
  .. note::

    For semiconductors, it is better to use either small Fermi-smearing in the
    ground state calculation::

      from gpaw import FermiDirac
      calc = GPAW(...
                  occupations=FermiDirac(0.001),
                  ...)

    or larger ftol, which determines the threshold for transition
    in the dielectric function calculation (`f_i - f_j > ftol`), not
    shown in the example script)::

   
2. Get the required matrix elements from the GS
  Here, the matrix elements of momentum are computed. Then, all required quantities
  such as energies, occupations, and momentum matrix elements are saved in a file ('mml.npz').
  The GS file cane be removed after this step.

  .. literalinclude:: silicon_ABS.py
      :lines: 31-36

  Note that, `ni` and `nf' are the first and last bands used for calculations.

3. Compute the SHG spectrum

  In this step, the SHG spectrum is calculated using the saved data.

Result
------

The figure shown here is generated from script :
:download:`silicon_ABS.py` and :download:`plot_ABS.py`.
It takes 30 minutes with 16 cpus on Intel Xeon X5570 2.93GHz.

.. image:: silicon_ABS.png
    :height: 300 px
    :align: center

The arrows are data extracted from \ [#Kresse]_.

The calculated macroscopic dielectric constant can be seen in the table below
and compare good with the values from [#Kresse]_. The experimental value is
11.90. The larger theoretical value results from the fact that the ground
state LDA (even GGA) calculation underestimates the bandgap.

.. csv-table::
   :file: mac_eps.csv



Technical details:
==================

There are few points about the implementation that we emphasize:

* The code is parallelized over kpoints and occupied bands. The
  parallelization over occupied bands makes it straight-forward to utilize
  efficient BLAS libraries to sum un-occupied bands.

* The code employs the Hilbert transform in which the spectral function
  for the density-density response function is calculated before calculating
  the full density response. This speeds up the code significantly for
  calculations with a lot of frequencies.

* The non-linear frequency grid employed in the calculations is motivated
  by the fact that when using the Hilbert transform the real part of the
  dielectric function converges slowly with the upper bound of the frequency
  grid. Refer to :ref:`df_theory` for the details on the Hilbert transform.

Drude phenomenological scattering rate
======================================
A phenomenological scattering rate can be introduced using the ``rate``
parameter in the optical limit. By default this is zero but can be set by
using::

    df = DielectricFunction(...
                            rate=0.1,
                            ...)

to set a scattering rate of 0.1 eV. If rate='eta' then the code with use the
specified ``eta`` parameter. Note that the implementation of the rate parameter
differs from some literature by a factor of 2 for consistency with the linear
response formalism. In practice the Drude rate is implemented as

.. math::

    \frac{\omega_\mathrm{p}^2}{(\omega + i\gamma)^2}

where `\gamma` is the rate parameter.

Useful tips
===========

Use dry_run option to get an overview of a calculation (especially useful for
heavy calculations!)::

    $ python3 filename.py --gpaw=df-dry-run=8

.. Note ::

    But be careful ! LCAO mode calculation results in unreliable unoccupied
    states above vacuum energy.

It's important to converge the results with respect to::

    nbands,
    nkpt (number of kpoints in gs calc.),
    eta,
    ecut,
    ftol,
    omegamax (the maximum energy, be careful if hilbert transform is used)
    domega0 (the energy spacing, if there is)
    vacuum (if there is)


Parameters
==========

=================  =================  ===================  ================================
keyword            type               default value        description
=================  =================  ===================  ================================
``calc``           ``str``            None                 gpw filename
                                                           (with 'all' option when writing
                                                           the gpw file)
``name``           ``str``            None                 If specified the chi matrix is
                                                           saved to ``chi+qx+qy+qz.pckl``
                                                           where ``qx, qy, qz`` is the
                                                           wave-vector components in
                                                           reduced coordinates.
``frequencies``    ``numpy.ndarray``  None                 Energies for spectrum. If
                                                           left unspecified the frequency
                                                           grid will be non-linear.
                                                           Ex: numpy.linspace(0,20,201)
``domega0``        ``float``          0.1                  `\Delta\omega_0` for
                                                           non-linear frequency grid.
``omega2``         ``float``          10.0 (eV)            `\omega_2` for
                                                           non-linear frequencygrid.
``omegamax``       ``float``          Maximum energy       Maximum frequency.
                                      eigenvalue
                                      difference.
``ecut``           ``float``          10 (eV)              Planewave energy cutoff.
                                                           Determines the size of
                                                           dielectric matrix.
``eta``            ``float``          0.2 (eV)             Broadening parameter.
``ftol``           ``float``          1e-6                 The threshold for transition:
                                                           `f_{ik} - f_{jk} > ftol`
``txt``            ``str``            stdout               Output filename.
``hilbert``        ``bool``           True                 Switch for hilbert transform.
``nbands``         ``int``            nbands from gs calc  Number of bands from gs calc
                                                           to include.
``rate``           ``float`` or
                   ``str``            0.0 (eV)             Phenomenological Drude
                                                           scattering rate. If rate="eta" then
                                                           use "eta". Note that this may differ
                                                           by a factor of 2 for some definitions
                                                           of the Drude scattering rate. See the
                                                           section on Drude scattering rate.
=================  =================  ===================  ================================


Details of the DF object
========================


.. autoclass:: gpaw.response.df.DielectricFunction
   :members: get_dielectric_function, get_macroscopic_dielectric_constant,
             get_polarizability, get_eels_spectrum


.. [#Kresse] M. Gajdoš, K. Hummer, G. Kresse, J. Furthmüller and F. Bechstedt,
              Linear optical properties in the projected-augmented
              wave methodology,
              *Phys. Rev. B* **73**, 045112 (2006).


.. [#Rubio] A. G. Marinopoulos, L. Reining, A. Rubio and V. Olevano,
             Ab initio study of the optical absorption and
             wave-vector-dependent dielectric response of graphite,
             *Phys. Rev. B* **69**, 245419 (2004).


.. [#MacDonald] 1. MacDonald, A. H., Vosko, S. H. & Coleridge, P. T.,
               Extensions of the tetrahedron method for evaluating spectral
               properties of solids. *J. Phys. C Solid State Phys*. **12**,
               2991–3002 (1979).
