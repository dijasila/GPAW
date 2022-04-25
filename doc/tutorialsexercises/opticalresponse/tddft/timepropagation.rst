.. module:: gpaw.tddft
.. _timepropagation:

======================
Time-propagation TDDFT
======================

Optical photoabsorption spectrum as well as nonlinear effects can be
studied using time propagation TDDFT. This approach
scales better than linear response, but the prefactor is so large that
for small and moderate systems linear response is significantly
faster.


------------
Ground state
------------

To obtain the ground state for TDDFT, one has to just do a standard ground state
with a larger simulation box. A proper distance from any atom to edge of the
simulation box is problem dependent, but a minimum reasonable value is around
6 Ångströms and recommended between 8-10 Ång. In TDDFT, one can use larger
grid spacing than for geometry optimization. For example, if you use h=0.25
for geometry optimization, try h=0.3 for TDDFT. This saves a lot of time.

A good way to start is to use too small box (vacuum=6.0), too large grid
spacing (h=0.35), and too large time step (dt=16.0). Then repeat the simulation
with better parameter values and compare. Probably lowest peaks are already
pretty good, and far beyond the ionization limit, in the continuum, the spectrum
is not going to converge anyway. The first run takes only fraction of
the time of the second run.

For a parallel-over-states TDDFT calculation, you must choose the number
of states so, that these can be distributed equally to processors. For
example, if you have 79 occupied states and you want to use 8 processes
in parallelization over states, add one unoccupied state to get 80 states
in total.


Ground state example::

  # Standard magic
  from ase import Atoms
  from gpaw import GPAW

  # Beryllium atom
  atoms = Atoms(symbols='Be',
                positions=[(0, 0, 0)],
                pbc=False)

  # Add 6.0 ang vacuum around the atom
  atoms.center(vacuum=6.0)

  # Create GPAW calculator
  calc = GPAW(nbands=1, h=0.3)
  # Attach calculator to atoms
  atoms.calc = calc

  # Calculate the ground state
  energy = atoms.get_potential_energy()

  # Save the ground state
  calc.write('be_gs.gpw', 'all')


Time-propagation TDDFT is also available in :ref:`LCAO mode <lcaotddft>`.

--------------------------------
Optical photoabsorption spectrum
--------------------------------

Optical photoabsorption spectrum can be obtained by applying a weak
delta pulse of dipole electric field, and then letting the system evolve
freely while recording the dipole moment. A time-step around 4.0-8.0
attoseconds is reasonable. The total simulation time should be few tens
of femtoseconds depending on the desired resolution.


Example::

  from gpaw.tddft import *

  time_step = 8.0                  # 1 attoseconds = 0.041341 autime
  iterations = 2500                # 2500 x 8 as => 20 fs
  kick_strength = [0.0,0.0,1e-3]   # Kick to z-direction

  # Read ground state
  td_calc = TDDFT('be_gs.gpw')

  # Save the time-dependent dipole moment to 'be_dm.dat'
  DipoleMomentWriter(td_calc, 'be_dm.dat')

  # Use 'be_td.gpw' as restart file
  RestartFileWriter(td_calc, 'be_td.gpw')

  # Kick with a delta pulse to z-direction
  td_calc.absorption_kick(kick_strength=kick_strength)

  # Propagate
  td_calc.propagate(time_step, iterations)

  # Save end result to 'be_td.gpw'
  td_calc.write('be_td.gpw', mode='all')

  # Calculate photoabsorption spectrum and write it to 'be_spectrum_z.dat'
  photoabsorption_spectrum('be_dm.dat', 'be_spectrum_z.dat')

When propagating after an absorption kick has been applied, it is a good
idea to periodically write the time-evolution state to a restart file.
This ensures that you can resume adding data to the dipole moment file
if you experience artificial oscillations in the spectrum because the total
simulation time was too short.

Example::

  from gpaw.tddft import *

  time_step = 8.0                  # 1 attoseconds = 0.041341 autime
  iterations = 2500                # 2500 x 8 as => 20 fs

  # Read restart file with result of previous propagation
  td_calc = TDDFT('be_td.gpw')

  # Append the time-dependent dipole moment to the already existing 'be_dm.dat'
  DipoleMomentWriter(td_calc, 'be_dm.dat')

  # Use 'be_td2.gpw' as restart file
  RestartFileWriter(td_calc, 'be_td2.gpw')

  # Propagate more
  td_calc.propagate(time_step, iterations)

  # Save end result to 'be_td2.gpw'
  td_calc.write('be_td2.gpw', mode='all')

  # Recalculate photoabsorption spectrum and write it to 'be_spectrum_z2.dat'
  photoabsorption_spectrum('be_dm.dat', 'be_spectrum_z2.dat')


Typically in experiments, the spherically averaged spectrum is measured.
To obtain this, one must repeat the time-propagation to each Cartesian
direction and average over the Fourier transformed dipole moments.


---------------------------------------------------
Fourier transformed density and electric near field
---------------------------------------------------

To analyze the resonances seen in the photoabsorption spectrum,
see :ref:`inducedfield`.


--------------------------------
Time propagation
--------------------------------

Since the total CPU time also depends on the number of iterations performed
by the linear solvers in each time-step, smaller time-steps around 2.0-4.0
attoseconds might prove to be faster with the ``ECN`` and ``SICN``
propagators because they have an embedded Euler step in each predictor step:

.. math::

  \tilde{\psi}_n(t+\Delta t) \approx (1 - i \hat{S}^{\;-1}_\mathrm{approx.}(t) \tilde{H}(t) \Delta t)\tilde{\psi}_n(t)

, where `\hat{S}^{\;-1}_\mathrm{approx.}` is an inexpensive operation
which approximates the inverse of the overlap operator `\hat{S}`. See
the :ref:`Developers Guide <overlaps>` for details.


Therefore, as a rule-of-thumb, choose a time-step small enough to minimize the
number of iterations performed by the linear solvers in each time-step, but
large enough to minimize the number of time-steps required to arrive at the
desired total simulation time.


----------------------
TDDFT reference manual
----------------------

.. autoclass:: gpaw.tddft.TDDFT
   :members:
   :exclude-members: read

.. autoclass:: gpaw.tddft.DipoleMomentWriter
   :members:

.. autoclass:: gpaw.tddft.MagneticMomentWriter
   :members:

.. autoclass:: gpaw.tddft.RestartFileWriter
   :members:

.. autofunction:: gpaw.tddft.photoabsorption_spectrum

.. autoclass:: gpaw.lcaotddft.densitymatrix.DensityMatrix
   :members:
