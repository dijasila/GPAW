.. module:: gpaw.tddft
.. _timepropagation:

======================
Ehrenfest dynamics (TDDFT/MD)
======================

Molecular dynamics (MD) simulations usually rely on the Born-Oppenheimer
approximation, where the electronic system is assumed to react so much
faster than the ionic system that it reaches its ground state at each timestep.
Thus, forces for the dynamics are calculated from the DFT groundstate density. 
While this approximation is sufficently valid in most situations, there are 
cases where the explicit dynamics of the electronic system can affect the
molecular dynamics, or the movement of the atoms can affect averaged spectral 
or other properties. These cases can be handled using so-called Ehrenfest 
dynamics, ie. time-dependent density functional theory molecular dynamics 
(TDDFT/MD). 

------------
Ground state
------------

Similar to static TDDFT calculations, one has to start with a standard ground
state simulation. In TDDFT, one can use larger grid spacing than for geometry 
optimization, so for example, if you use h=0.25 for geometry optimization, 
try h=0.3 for TDDFT to save a lot of time (the same spacing should be used for
the ground state and TDDFT calculators).

Ground state example::

   from ase import Atoms
   from gpaw import GPAW
   
   name = 'h2_diss'
   
   # Create H2 molecule in the center of a box
   d_bond = 0.754 # H2 equilibrium bond length
   atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, d_bond)])
   atoms.set_pbc(False)
   atoms.center(vacuum=4.0)
   
   # Set groundstate calculator and get and save wavefunctions
   calc = GPAW(h=0.3, nbands=1, basis='dzp', txt=name + '_gs.txt')
   atoms.set_calculator(calc)
   atoms.get_potential_energy()
   calc.write(name + '_gs.gpw', mode='all')

Ehrenfest TDDFT/MD is also available in :ref:`LCAO mode <lcaotddft>`.

--------------------------------
Simulating H2 dissociation
--------------------------------

We then simulate Ehrenfest dynamics of the H2 molecule in an intense laser field to observe
its dissociation.

Example::
   from gpaw.tddft import TDDFT
   from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
   from gpaw.tddft.laser import CWField
   from ase.units import Hartree, Bohr, AUT
   
   name = 'h2_diss'
   
   # Ehrenfest simulation parameters
   timestep = 20.0 # Timestep given in attoseconds
   ndiv = 10  # Write trajectory every 10 timesteps
   niter = 500 # Run for 500 timesteps 
   
   # TDDFT calculator with an external potential emulating an intense harmonic laser field
   # aligned along the H2 molecular axis.
   tdcalc = TDDFT(name + '_gs.gpw', txt=name + '_td.txt', propagator='EFSICN',
                  solver='BiCGStab', td_potential=CWField(1000 * Hartree, 1 * AUT, 10))

   # For Ehrenfest dynamics, we use this object for the Velocity Verlet dynamics.   
   ehrenfest = EhrenfestVelocityVerlet(tdcalc)

   # Trajectory to save the dynamics.
   traj = Trajectory(name + '_td.traj', 'w', tdcalc.get_atoms())
   
   # Propagates the dynamics for niter timesteps.
   for i in range(1, niter + 1):
      # Propagates for one timestep.
      ehrenfest.propagate(timestep)
   
      # Every ndiv timesteps, save an image in the trajectory file.
      if i % ndiv == 0:
          # Currently needed with Ehrenfest dynamics to save energy, forces and velocitites.
          epot = tdcalc.get_td_energy() * Hartree
          F_av = ehrenfest.F * Hartree / Bohr # Forces converted from atomic units
          v_av = ehrenfest.v * Bohr / AUT # Velocities converted from atomic units
          atoms.set_velocities(v_av) # Needed to save the velocities to the trajectory
   
          traj.write(atoms.copy(), energy=epot, forces=F_av)
   
   traj.close()

As can be verified from the trajectory, the distance between the H atoms at the end of
the dynamics is more than 3 A and their bond is broken by the intense laser field.
 
--------------------------------
TDDFT reference manual
--------------------------------

The :class:`~gpaw.tddft.TDDFT` class and keywords:

===================== =============== ============== =====================================
Keyword               Type            Default        Description
===================== =============== ============== =====================================
``ground_state_file`` ``string``                     Name of the ground state file
``td_potential``      ``TDPotential`` ``None``       Time-dependent external potential
``propagator``        ``string``      ``'SICN'``     Time-propagator (``'ECN'``/``'SICN'``/``'SITE'``/``'SIKE'``)
``solver``            ``string``      ``'CSCG'``     Linear equation solver (``'CSCG'``/``'BiCGStab'``)
``tolerance``         ``float``       ``1e-8``       Tolerance for linear solver
===================== =============== ============== =====================================

Keywords for :func:`~gpaw.tddft.TDDFT.propagate`:

====================== =========== =========== ================================================
Keyword                Type        Default     Description
====================== =========== =========== ================================================
``time_step``          ``float``               Time step in attoseconds (``1 autime = 24.188 as``)
``iterations``         ``integer``             Iterations
``dipole_moment_file`` ``string``  ``None``    Name of the dipole moment file
``restart_file``       ``string``  ``None``    Name of the restart file
``dump_interal``       ``integer`` ``500``     How often restart file is written
====================== =========== =========== ================================================

.. autofunction:: gpaw.tddft.ehrenfest

.. autoclass:: gpaw.tddft.TDDFT
   :members:

