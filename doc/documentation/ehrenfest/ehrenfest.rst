.. _ehrenfest:

=============================
Ehrenfest dynamics (TDDFT/MD)
=============================

For a brief introduction to the Ehrenfest dynamics theory and the details of
its implementation in GPAW, see :ref:`Ehrenfest theory <ehrenfest_theory>`.
The original implementation by Ari Ojanpera is described in
Ref. [#Ojanpera2012]_.


------------
Ground state
------------

Similar to static TDDFT calculations, one has to start with a standard ground
state simulation. In TDDFT, one can use larger grid spacing than for geometry
optimization, so for example, if you use ``h=0.25`` for geometry optimization,
try ``h=0.3`` for TDDFT to save a lot of time (the same spacing should be
used for the ground state and TDDFT calculators).

Ground state example:

.. literalinclude:: h2_gs.py

Ehrenfest TDDFT/MD is also available in :ref:`LCAO mode <lcaotddft>`.


--------------------------
Simulating H2 dissociation
--------------------------

We then simulate Ehrenfest dynamics of the H2 molecule in a very intense
laser field to observe its dissociation.

For Ehrenfest dynamics we must use the parameter ``propagator='EFSICN'`` for
the TDDFT calculator ``tdcalc`` to take into account the necessary
corrections in the propagation of the time-dependent Kohn-Sham equation. The
parameter ``solver='BiCGStab'`` to use the stabilized biconjugate gradient
method (``BiCGStab``) is generally recommended for Ehrenfest dynamics
calculations for any system more complex than simple dimers or dimers.

H2 dissociation example:

.. literalinclude:: h2_diss.py

As can be verified from the trajectory, the distance between the H atoms at
the end of the dynamics is more than 3 Å and thus their bond was broken by
the intense laser field.


-------------------------------
Electronic stopping in graphene
-------------------------------

A more complex use for Ehrenfest dynamics is to simulate the irradiation of
materials with either chared ions or neutral atoms (see Refs. \
[#Ojanpera2014]_ and \ [#Brand2019]_).

This example demonstrates how to carry out ion stopping calculations using
the Ehrenfest dynamics code. The following scripts create a system consisting
of a proton as a projectile and graphene as the target.


Creating the projectile + target system
---------------------------------------

The following script calculates the ground state of the projectile + target
system. An external potential is used at the hydrogen ion the converge the
calculation. One might also have to change the default convergence parameters
depending on the projectile used. Here, slightly less strict convergence
criteria are used. The impact point in this case is the center of a carbon
hexagon, but this can be modified by changing the x-y position of the H atom
(``projpos``).

Projectile + target example:

.. literalinclude:: graphene_h_gs.py

Finally, the following script can be used for performing an electronic stopping calculation for a hydrogen ion impacting
graphene with the initial velocity being 40 keV. The external potential is automatically set to zero when the TDDFT object is
initialized and hence does not affect the calculation. The calculation ends when the distance between the projectile and the bottom of
the supercell is less than 3 `\unicode{x212B}`. (Note: this is a fairly demanding calculation even with 32 cores.)

Electronic stopping example::

   from ase.units import _amu, _me, Bohr, AUT, Hartree
   from gpaw import GPAW
   from gpaw.tddft import TDDFT
   from ase.parallel import paropen
   from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
   from ase.io import Trajectory
   from gpaw.mpi import world
   import numpy as np

   name = 'graphene_h'
   Ekin = 40e3
   timestep = 1.0 * np.sqrt(10e3/Ekin)
   ekin_str = '_ek' + str(int(Ekin/1000)) + 'k'
   amu_to_aumass = _amu/_me
   strbody = name + ekin_str
   traj_file = strbody + '.traj'

   # The parallelization options should match the number of cores, here 32.
   p_bands = 2 # Number of bands to parallelise over
   dom_dc = (4,4,1) # Domain decomposition for parallelization
   parallel = {'band':p_bands, 'domain':dom_dc}

   tdcalc = TDDFT(name + '.gpw', propagator='EFSICN', solver='BiCGStab', txt=strbody + '_td.txt', parallel=parallel)

   proj_idx = 50
   v = np.zeros((proj_idx+1,3))
   delta_stop = 3.0 / Bohr
   Mproj = tdcalc.atoms.get_masses()[proj_idx]
   Ekin *= Mproj
   Ekin = Ekin / Hartree

   Mproj *= amu_to_aumass
   v[proj_idx,2] = -np.sqrt((2*Ekin)/Mproj) * Bohr / AUT
   tdcalc.atoms.set_velocities(v)

   evv = EhrenfestVelocityVerlet(tdcalc)
   traj = Trajectory(traj_file, 'w', tdcalc.get_atoms())

   trajdiv = 2 # Number of timesteps between trajectory images
   densdiv = 10 # Number of timesteps between saved electron densities
   niters = 200 # Total number of timesteps to propagate

   for i in range(niters):
      # Stopping condition when projectile z coordinate passes threshold
      if evv.x[proj_idx,2] < delta_stop:
         tdcalc.write(strbody + '_end.gpw', mode='all')
         break

      # Saving trajectory file every trajdiv timesteps
      if i % trajdiv == 0:
         F_av = evv.F * Hartree / Bohr # Forces converted from atomic units
         v_av = evv.v * Bohr / AUT # Velocities converted from atomic units
         epot = tdcalc.get_td_energy() * Hartree # Energy converted from atomic units
         atoms = tdcalc.get_atoms().copy()
         atoms.set_velocities(v_av)

         traj.write(atoms, energy=epot, forces=F_av)

      # Saving electron density every densdiv timesteps
      if (i != 0 and i % densdiv == 0):
         tdcalc.write(strbody + '_step'+str(i)+'.gpw')
         v[proj_idx,2] = -np.sqrt((2*Ekin)/Mproj) * Bohr / AUT
         tdcalc.atoms.set_velocities(v)

      evv.propagate(timestep)

   traj.close()

--------------------------------
TDDFT reference manual
--------------------------------

The TDDFT class and keywords:

===================== =============== ============== =====================================
Keyword               Type            Default        Description
===================== =============== ============== =====================================
``ground_state_file`` ``string``                     Name of the ground state file
``td_potential``      ``TDPotential`` ``None``       Time-dependent external potential
``propagator``        ``string``      ``'SICN'``     Time-propagator (``'ECN'``/``'SICN'``/``'SITE'``/``'SIKE'``)
``solver``            ``string``      ``'CSCG'``     Linear equation solver (``'CSCG'``/``'BiCGStab'``)
``tolerance``         ``float``       ``1e-8``       Tolerance for linear solver
===================== =============== ============== =====================================

Keywords for TDDFT.propagate:

====================== =========== =========== ================================================
Keyword                Type        Default     Description
====================== =========== =========== ================================================
``time_step``          ``float``               Time step in attoseconds (``1 autime = 24.188 as``)
``iterations``         ``integer``             Iterations
``dipole_moment_file`` ``string``  ``None``    Name of the dipole moment file
``restart_file``       ``string``  ``None``    Name of the restart file
``dump_interal``       ``integer`` ``500``     How often restart file is written
====================== =========== =========== ================================================


----------
References
----------

.. [#Ojanpera2012] A. Ojanpera, V. Havu, L. Lehtovaara, M. Puska,
                   "Nonadiabatic Ehrenfest molecular dynamics within the projector augmented-wave method",
                   *J. Chem. Phys.* **136**, 144103 (2012).

.. [#Ojanpera2014] A. Ojanpera, Arkady V. Krasheninnikov, M. Puska,
                   "Electronic stopping power from first-principles calculations with account for core
                   electron excitations and projectile ionization",
                   *Phys. Rev. B* **89**, 035120 (2014).

.. [#Brand2019] C. Brand et al., to be published.
