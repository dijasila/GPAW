.. _ehrenfest:

=============================
Ehrenfest dynamics (TDDFT/MD)
=============================

For a brief introduction to the Ehrenfest dynamics theory and the details of
its implementation in GPAW, see :ref:`Ehrenfest theory <ehrenfest_theory>`.
The original implementation by Ari Ojanpera is described in
Ref. [#Ojanpera2012]_.


.. seealso::

    * :ref:`timepropagation`
    * :class:`gpaw.tddft.TDDFT`
    * :meth:`gpaw.tddft.TDDFT.propagate`


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
<<<<<<< HEAD
--------------------------------

We then simulate Ehrenfest dynamics of the H2 molecule in a very intense laser field to 
observe its dissociation.

For Ehrenfest dynamics we must use the parameter ``propagator='EFSICN'`` for the TDDFT
calculator ``tdcalc`` to take into account the necessary corrections in the propagation
of the time-dependent Kohn-Sham equation. The parameter ``solver='BiCGStab'`` to use the
stabilized biconjugate gradient method (``BiCGStab``) is generally recommended for Ehrenfest
dynamics calculations for any system more complex than simple dimers or dimers.

H2 dissociation example::

   from gpaw.tddft import TDDFT
   from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
   from gpaw.tddft.laser import CWField
   from ase.units import Hartree, Bohr, AUT
   from ase.parallel import parprint

   name = 'h2_diss'
   
   # Ehrenfest simulation parameters
   timestep = 10.0 # Timestep given in attoseconds
   ndiv = 10 # Write trajectory every 10 timesteps
   niter = 500 # Run for 100 timesteps 
   
   # TDDFT calculator with an external potential emulating an intense harmonic laser field
   # aligned (CWField uses by default the z axis) along the H2 molecular axis.
   tdcalc = TDDFT(name + '_gs.gpw', txt=name + '_td.txt', propagator='EFSICN',
                  solver='BiCGStab', td_potential=CWField(1000 * Hartree, 1 * AUT, 10))

   # For Ehrenfest dynamics, we use this object for the Velocity Verlet dynamics.   
   ehrenfest = EhrenfestVelocityVerlet(tdcalc)

   # Trajectory to save the dynamics.
   traj = Trajectory(name + '_td.traj', 'w', tdcalc.get_atoms())
   
   # Propagates the dynamics for niter timesteps.
   for i in range(1, niter + 1):
   ehrenfest.propagate(timestep)

   if atoms.get_distance(0,1) > 2.0: # Stop simulation if H-H distance is greater than 2 A.
       parprint('Dissociated!')
       break
       
    # Every ndiv timesteps, save an image in the trajectory file.
    if i % ndiv == 0:
    # Currently needed with Ehrenfest dynamics to save energy, forces and velocitites.
      epot = tdcalc.get_td_energy() * Hartree
      F_av = ehrenfest.F * Hartree / Bohr # Forces converted from atomic units
      v_av = ehrenfest.v * Bohr / AUT # Velocities converted from atomic units
      atoms = tdcalc.atoms.copy()
      atoms.set_velocities(v_av) # Needed to save the velocities to the trajectory
   
      traj.write(atoms, energy=epot, forces=F_av)
   
   traj.close()

The distance between the H atoms at the end of the dynamics is more than 2 A and thus 
their bond has been broken by the intense laser field.

--------------------------------
=======
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
>>>>>>> b647503a12ad5c3467091ce82b2dba99aad00151
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

<<<<<<< HEAD
The following script calculates the ground state of the projectile + target system. An external potential is used at the hydrogen
ion the converge the calculation. One might also have to change the default convergence parameters depending on the projectile used. 
Here, slightly less strict convergence criteria are used. The impact point in this case is the center of a carbon hexagon, but this
can be modified by changing the x-y position of the H atom (`projpos`).

Projectile + target example::

   import ase.io as io
   import numpy as np
   
   from ase.lattice.hexagonal import Graphene
   from gpaw.mixer import Mixer, MixerSum
   from gpaw import GPAW
   from ase import Atom, Atoms
   from gpaw.utilities import h2gpts
   from ase.units import Bohr
   from gpaw.occupations import FermiDirac
   from gpaw.external import ConstantElectricField
   from gpaw.mpi import world

   def gaussian(x, x0, A):
      E = np.linalg.norm(x-x0)
      return A*np.exp(-E**2)
   
   name = 'graphene_h'
   
   # 5 x 5 supercell of graphene
   index1 = 5
   index2 = 5
   a = 2.45
   c = 3.355

   gra = Graphene(symbol = 'C',latticeconstant={'a':a,'c':c},
                  size=(index1,index2,1), pbc = (1,1,0))

   gra.center(vacuum=15.0, axis=2)
   gra.center()

   # Starting position of the projectile with an impact point at the center of a hexagon
   projpos = [[gra[15].position[0], gra[15].position[1]+1.41245, 25.0]]

   H = Atoms('H', cell=gra.cell, positions=projpos)

   # Combine target and projectile
   atoms = gra + H
   atoms.set_pbc(True)

   conv_fast = {'energy':1.0, 'density': 1.0, 'eigenstates':1.0}
   conv_par = {'energy':0.001, 'density': 1e-3, 'eigenstates':1e-7}
   const_pot = ConstantPotential(1.0)
   mixer= Mixer(0.1,5,weight=100.0)

   calc = GPAW(gpts=h2gpts(0.2, gra.get_cell(), idiv=8),
               nbands = 110, xc='LDA',charge=1, txt=name + '_gs.txt',
               convergence=conv_fast, external=const_pot)

   atoms.set_calculator(calc)
   atoms.get_potential_energy()

   A = 1.0
   X0 = atoms.positions[50] / Bohr
   rcut = 3.0 / Bohr
   vext = calc.hamiltonian.vext
   gd = calc.hamiltonian.finegd
   n_c = gd.n_c
   h_c = gd.get_grid_spacings()
   b_c = gd.beg_c
   vext.vext_g[:] = 0.0
   for i in range(n_c[0]):
      for j in range(n_c[1]):
         for k in range(n_c[2]):
            x = h_c[0]*(b_c[0] + i)
            y = h_c[1]*(b_c[1] + j)
            z = h_c[2]*(b_c[2] + k)
            X = np.array([x,y,z])
            dist = np.linalg.norm(X-X0)
            if(dist < rcut):
               vext.vext_g[i,j,k] += gaussian(X,X0,A)

   calc.set(convergence=conv_par, eigensolver=RMMDIIS(5), external=vext)
   
   atoms.get_potential_energy()
   calc.write(name + '.gpw', mode='all')

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

----------------------
TDDFT reference manual
----------------------

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
=======
The following script calculates the ground state of the projectile + target
system. An external potential is used at the hydrogen ion the converge the
calculation. One might also have to change the default convergence parameters
depending on the projectile used. Here, slightly less strict convergence
criteria are used. The impact point in this case is the center of a carbon
hexagon, but this can be modified by changing the x-y position of the H atom
(``projpos``).

Projectile + target example:

.. literalinclude:: graphene_h_gs.py

Finally, the following script can be used for performing an electronic
stopping calculation for a hydrogen ion impacting graphene with the initial
velocity being 40 keV. The external potential is automatically set to zero
when the TDDFT object is initialized and hence does not affect the
calculation. The calculation ends when the distance between the projectile
and the bottom of the supercell is less than 3 Å. (Note: this is a fairly
demanding calculation even with 32 cores.)

Electronic stopping example:

.. literalinclude:: graphene_h_prop.py
>>>>>>> b647503a12ad5c3467091ce82b2dba99aad00151


----------
References
----------

.. [#Ojanpera2012] A. Ojanpera, V. Havu, L. Lehtovaara, M. Puska,
                   "Nonadiabatic Ehrenfest molecular dynamics within
                   the projector augmented-wave method",
                   *J. Chem. Phys.* **136**, 144103 (2012).

.. [#Ojanpera2014] A. Ojanpera, Arkady V. Krasheninnikov, M. Puska,
                   "Electronic stopping power from first-principles
                   calculations with account for core
                   electron excitations and projectile ionization",
                   *Phys. Rev. B* **89**, 035120 (2014).

.. [#Brand2019] C. Brand et al., to be published.
