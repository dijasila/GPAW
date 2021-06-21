.. _raman:

========================
Raman spectroscopy
========================

GPAW offers two ways of calculating Raman intensities. One can use the `ASE Raman
<https://wiki.fysik.dtu.dk/ase/ase/vibrations/raman.html>`_ utility together
with the GPAW LRTDDFT module as shown in the Resonant Raman tutorial (enter link).

GPAW also implements Raman spectroscopy for zone-center phonons of extended systems
using the electron-phonon coupling (see :ref:`elph`) within 3rd order perturbation 
theory :dfn:`Taghizadeh et a.` [#Taghizadeh2020]_ , which is discussed here. This method is currently only
implementated for the LCAO mode.

The Stokes Raman intensity can be written as

.. math::

    I(\omega) = I_0 \sum_\nu \frac{n_\nu+1}{\omega_\nu} \vert 
    \sum_{\alpha, \beta} u_{in}^\alpha R_{\alpha \beta}^\nu u_{out}^\beta
    \vert^2 \delta(\omega-\omega_\nu)

where `\nu` denotes phonon modes and `\alpha`, `\beta` denote polarisations
of the incoming and outgoing laser light.
The Raman tensor `R_{\alpha \beta}^\nu` has six terms and is given by
Ref. [#Taghizadeh2020]_ Eq. (10)

.. math::

    R_{\alpha \beta}^\nu \equiv \sum_{ijmn \mathbf{k}} \left[
    \frac{p_{ij}^\alpha (g_{jm}^\nu \delta_{in} - g_{ni}^\nu \delta_{jm})p_{mn}^\beta}{(\hbar \omega_{in}-\varepsilon_{ji})(\hbar \omega_{out}-\varepsilon_{mn})} +
    \frac{p_{ij}^\alpha (p_{jm}^\beta \delta_{in} - p_{ni}^\beta \delta_{jm})g_{mn}^\nu}{(\hbar \omega_{in}-\varepsilon_{ji})(\hbar \omega_{\nu}-\varepsilon_{mn})} + \\
    \frac{p_{ij}^\beta (g_{jm}^\nu \delta_{in} - g_{ni}^\nu \delta_{jm})p_{mn}^\alpha}{(-\hbar \omega_{out}-\varepsilon_{ji})(-\hbar \omega_{in}-\varepsilon_{mn})} +
    \frac{p_{ij}^\beta (p_{jm}^\alpha \delta_{in} - p_{ni}^\alpha \delta_{jm})g_{mn}^\nu}{(-\hbar \omega_{out}-\varepsilon_{ji})(\hbar \omega_{\nu}-\varepsilon_{mn})} + \\
    \frac{g_{ij}^\nu (p_{jm}^\alpha \delta_{in} - p_{ni}^\alpha \delta_{jm})p_{mn}^\beta}{(-\hbar \omega_{\nu}-\varepsilon_{ji})(\hbar \omega_{out}-\varepsilon_{mn})} +
    \frac{g_{ij}^\nu (p_{jm}^\beta \delta_{in} - p_{ni}^\beta \delta_{jm})p_{mn}^\alpha}{(-\hbar \omega_{\nu}-\varepsilon_{ji})(-\hbar \omega_{in}-\varepsilon_{mn})}
    \right] f_i(1-f_j)f_n(1-f_m)

The first term is considered to be the resonant term of the expression, the other
terms represent different time orderings of the interaction in the Feynman diagrams.

To compute the Raman intensity we need these ingredients: The momentum matrix
elements `p_{ij}^\alpha=\langle i \mathbf{k} | \hat p^\alpha| j \mathbf{k} \rangle`,
the electron-phonon matrix `g_{ij}^\nu = \langle i \mathbf{k} \vert \partial_{\nu{q=0}} V^{KS} \vert j \mathbf{k} \rangle`
in the optical limit `\mathbf{q}=0` and of course knowledge of the electronic
states and phonon modes throughout the Brillouin zone.
For these calculations we can employ in the :meth:`~gpaw.raman.dipoletransition.get_momentum_transitions`
method, the GPAW electron-phonon module :ref:`elph` and the ASE phonon module, respectively.

Some more details are elaborated in the related tutorial (link).

Example
=======

In this example we compute the Raman spectrum of diamond. For convenience
we split the calculations into three independent parts before the calculation
of the Raman tensor.

The momentum matrix elements are easiest to compute, as they only require a
converged DFT calculation (:git:`~doc/documentation/raman/momentum_matrix.py`):

.. literalinclude:: momentum_matrix.py

In the above script we converge all states to ensure valid matrix elements
throughout. The momentum matrix elements do not depend too strongly on grid spacings
and convergence parameters, though convergence studies are encouraged. Parameters
should be chosen to be consistent with electron-phonon calculations to ensure the same
order of nearly degenerate states.
The :meth:`~gpaw.raman.dipoletransition.get_momentum_transitions` method is
currently not aware of symmetry. It is therefore required to switch off
point-group symmetry in GPAW, so that matrix elements for all k-points and not
just the irreducible ones are calculated. The momentum matrix elements transform
as `p \rightarrow -p^*` under time-reversal symmetry `\mathbf{k} \rightarrow -\mathbf{k}`.
Accordingly, it is possible to enable time-reversal symmetry.
By default the routine saves a file called ``mom_skvnm.npy`` containing the
momentum matrix. This can be deactivated using the ``savetofile`` switch. The
matrix is always the return value of :meth:`~gpaw.raman.dipoletransition.get_momentum_transitions`.

Energy changes for phonons and potential changes for electron-phonon couplings
are both computed using a finite displacement technique. Both quantities can be
obtained simultaenously. In principle the phonon modes can be obtained in any
fashion, which yields an ASE phonon object though.
In GPAW an input script could look like this
(:git:`~doc/documentation/raman/phonon_and_potential.py`):

.. literalinclude:: phonon_and_potential.py

Again, in the above example the k-point grid is not converged. As only
zone-centre phonons are of required for the Raman calculations, no extended
supercell needs to be used. Point group symmetry needs to be deactivated again, as
symmetry is lowered by the displacements, which is not permitted by GPAW. The
above convergence criteria should be good starting points, but convergence tests
are always encouraged.
















TODO:
- test if we need supercell for g
- test parallelisation of load_supercell_matrix


calculate_supercell_matrix no domain parallelisation
calculate_supercell_matrix needs clean converged calc,


.. The Raman code offers wrappers for the electron-phonon part of the calculations:
.. :meth:`~gpaw.raman.elph.run_elph`
.. :meth:`~gpaw.raman.elph.calculate_supercell_matrix`
.. :meth:`~gpaw.raman.elph.get_elph_matrix`
.. 
.. where the last function is currently not parallelised and needs to be run in a
.. separate serial run.
.. 
.. Using the wrappers an electron-phonon calculation could look like this
.. (:git:`~doc/documentation/raman/elph.py`):
.. 
.. .. literalinclude:: elph.py
.. 
.. The last steps, which usually are not computationally intensive, have not been
.. parallelised yet, and need to be executed in a serial run.
.. 
.. The previously calculated supercell matrix needs to be converted into the
.. electron-phonon matrix for the Bloch states, which can be like this
.. (:git:`~doc/documentation/raman/elph_matrix.py`):
.. 
.. .. literalinclude:: elph_matrix.py
.. 
.. The electron-phonon matrix is saved as ``gsqklnn.npy`` and is our first
.. ingredient for the Raman calculation.
.. 
.. The optional ``load_gx_as_needed`` tag prevents from all supercell pickle files
.. being read at once. Instead they are loaded and processed one-by-one. This can
.. save lots of memory for larger systems with hundreds of atoms, where the
.. supercell matrix can be over 100GiB large.
.. 
..  The last
.. ingredient, the pickle file for the phonons should have been calculated in a
.. separate calculation as well. Now the Raman intensities can be computed
.. (:git:`~doc/documentation/raman/raman_intensities.py`):
.. 
.. .. literalinclude:: raman_intensities.py
.. 
.. The :meth:`calculate_raman` function computes the Raman tensor for each mode
.. for the given incident and outgoing directions. The results are saved as
.. ``Rlab_??.npy`` files. The optional ``resonant_only`` tag can be used to
.. deactivate the calculation of the last 5 terms in Ref. [#Taghizadeh2020]_ Eq.10.
.. This might be necessary for very large unit cells.
.. 
.. The :meth:`calculate_raman_intensity` function computes the Raman intensity on
.. a frequency grid using the ``Rlab`` files and and saves them into ``RI_??.npy``
.. files.
.. 
.. Lastly, we can plot the ``RI`` files with :meth:`plot_raman`. As the Raman
.. intensities are saved as ``npy`` files the users can of course use their own
.. routines instead for plotting.
.. 
.. .. image:: Raman_xx_532nm.png
..    :scale: 30
..    :align: center 

----------
References
----------

.. [#Taghizadeh2020] A. Taghizadeh, U. Leffers, T.G. Pedersen, K.S. Thygesen,
                   "A library of ab initio Raman spectra for automated
                   identification of 2D materials",
                   *Nature Communications* **11**, 3011 (2020).

----
Code
----

.. autofunction:: gpaw.raman.dipoletransition.get_momentum_transitions
.. autofunction:: gpaw.raman.elph.run_elph
.. autofunction:: gpaw.raman.elph.calculate_supercell_matrix

