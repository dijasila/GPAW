.. _grids:

=====
Grids
=====

Assume that we have an ``Atoms`` object contained in a cubic unit
cell of sidelength ``L``::

  L = 2.0
  atoms = Atoms(cell=(L, L, L), pbc=True)

and we use a calculator with a grid spacing of ``h=0.25`` Å or
``gpts=(8, 8, 8)``.  Since we have periodic boundary conditions, the
*x*-axis will look like this (the *y* and *z*-axes look the same)::

  0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 
 -+---------------+---------------+---------------+-
 -L               0               L              2*L

Wave functions are represented on 8x8x8 grids, where the grid points
are numbered from 0 to 7.

If we use zero boundary conditions (``pbc=False``), then the
*x*-axis will look like this::

                    0 1 2 3 4 5 6
                  +---------------+
                  0               L

Here the wave functions are exactly zero at *x*\ =0 Å and *x*\ =\ *L*,
and only the non-zero values are stored in 7x7x7 grids (grid points
numbered from 0 to 6).


Update this XXX how about padding?

An example:

>>> from ase import Atoms
>>> from gpaw import GPAW
>>> from ase.units import Bohr
>>> L = 2.0
>>> atoms = Atoms('H',
...               positions=[(1.0,1.0,1.0)],
...               cell=(L, L, L),
...               pbc=False)
>>> calc = GPAW(xc='LDA', gpts=(8, 8, 8), txt=None)
>>> atoms.set_calculator(calc)
>>> e = atoms.get_potential_energy()
>>> wf = calc.get_pseudo_wave_function(band=0)
>>> wf.shape
(8, 8, 8)
>>> calc.wfs.gd.h_cv[0,0] * Bohr
0.25
