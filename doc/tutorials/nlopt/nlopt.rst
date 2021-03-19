.. module:: gpaw.nlopt

.. _nlo_tutorial:

================================================
Nonlinear optical response of an extended system
================================================

A short introduction
=====================

The nonlinear optical (NLO) response of an extended system can be obtained
from its ground state electronic structure. Due to large computional cost,
the local field effect effect is neglected. Hence, the NLO response is only
frequency-dependent. The NLO response can be descibed by nonlinear
susceptibility tensors. There are numerious NLO processes, depending on the
number of photons involved in the process.

The second harmonic generation (SHG) is a NLO process, in which two photons
at the frequency of `\omega` generate a photon at the frequency of `2\omega`.
The SHG response is characterized by the second-order (quadratic)
susceptibility tensor, defined via

.. math::

    P_{\gamma}(t) =
    \sum_{\alpha\beta} \chi_{\gamma\alpha\beta}^{(2)}(\omega,\omega)
    E_{\alpha}E_{\beta} e^{-2i\omega t}+\textrm{c.c.}

where `\alpha,\beta={x,y,z}` denotes the Cartezain coordinates, and
`E_{\alpha}` and `E_{\alpha}` are the polarization and electric fields,
respectivly. For bulk systems, `\chi_{\gamma\alpha\beta}^{(2)}` is expressed
in the units of m/V. The details of computing
`\chi_{\gamma\alpha\beta}^{(2)}` is documented in Ref. [#Taghizadeh]_.


Example 1: SHG spectrum of semiconductor: Monolayer MoS2
========================================================

To compute the SHG spectrum of given structure, 3 steps are performed:

1. Ground state (GS) calculation

  .. literalinclude:: shg_MoS2.py
    :end-before: momentum

  In this script a normal ground state calculation is performed with coarse
  kpoint grid. Note that LCAO basis is used here, but the PW basis set can
  also be used. For a smoother spectrum, a finer mesh should be employed.


2. Get the required matrix elements from the GS

   Here, the matrix elements of momentum are computed. Then, all required
   quantities such as energies, occupations, and momentum matrix elements are
   saved in a file ('mml.npz'). The GS file cane be removed after this step.

   .. literalinclude:: shg_MoS2.py
     :start-after: write
     :end-before: Shift

   Note that, two optional paramters are available in the
   :func:`gpaw.nlopt.matrixel.make_nlodata` function:
   ``ni`` and ``nf`` as the first and last bands used for calculations of SHG.

3. Compute the SHG spectrum

   In this step, the SHG spectrum is calculated using the saved data.
   There are two well-known gauges that can be used: length gauge or velocity
   gauge. Formally, they are equivalnent but they may generate different
   results.   Here, SHG susceptibility is computed in both gauges and saved.
   The SHG susceptibility is a rank-3 symmteric tensor with at most 18
   independent components.   In addition, the point group symmtery reduce the
   number of independent tensor elements.   Monolayer MoS2 has only one
   independent tensor element: yyy. A broadening is necessary to obtain smooth
   graphs, and here 50 meV has been used.

   .. literalinclude:: shg_MoS2.py
     :start-after: make_nlodata(gs_name=gs_name, out_name=mml_name)


Result
------

Now the calculated SHG spectra are plotted at the end. Both real and
imaginary parts of the computed SHG susceptibilities, obtained from two
gauges are shown. The gauge invariance is confirmed from the calculation.
Note that the bulk susceptibility (with SI units of m/V) is ill-defined for
2D materials, since the volume cannot be defined without ambiguity in 2D
systems. Instead, the sheet susceptibility, expressed in unit of m\ `^2`/V, is
an unambiguous quantity for 2D materials. Hence, the bulk susceptibility is
transformed to the unambiguous sheet susceptibility by multiplying with the
width of the unit cell in the `z`-direction.

.. literalinclude:: shg_plot.py

The figure shown here is generated from scripts:
:download:`shg_MoS2.py` and :download:`shg_plot.py`.
It takes 30 minutes with 16 cpus on Intel Xeon X5570 2.93GHz.

.. image:: shg.png
    :height: 300 px
    :align: center


Technical details
=================

There are few points about the implementation that we emphasize:

* The code is parallelized over kpoints.

* The code employs only the time reversal symmtery to improve the speed.

* Refer to [#Taghizadeh]_ for the details on the NLO theory.


Useful tips
===========

It's important to converge the results with respect to:

    * ``nbands``
    * ``nkpt`` (number of kpoints in gs calc.)
    * ``eta``
    * ``ecut``
    * ``ftol``
    * vacuum (if there is)


API
===

.. autofunction:: gpaw.nlopt.matrixel.make_nlodata
.. autofunction:: gpaw.nlopt.shg.get_shg
.. autofunction:: gpaw.nlopt.linear.get_chi_tensor


.. [#Taghizadeh] A. Taghizadeh, K. S. Thygesen and T. G. Pedersen
                 *Arxiv*, 2010.11596 (2020)
