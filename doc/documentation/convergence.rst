.. _convergence:

==================
Convergence Issues
==================

*Try to use default parameters for the calculator. Simple but
sometimes useful.*

Here you find a list of suggestions that should be considered when
encountering convergence problems:

1) Make sure the geometry and spin-state is physically sound.
    
   Remember that ASE uses Angstrom and not Bohr or nm!
   For spin polarized systems, make sure you have sensible initial magnetic
   moments.

2) Use less aggressive density mixing.

   Try something like ``mixer=Mixer(0.02, 5, 100)`` or
   ``mixer=MixerSum(0.02, 5, 100)`` for spin-polarized calculations
   and remember to import the mixer class::
       
       from gpaw import Mixer  # or MixerSum

3) Solve the eigenvalue problem more accurately at each scf-step.

   Import the Davidson eigensolver::
       
       from gpaw import Davidson
       
    and use three iterations per scf-step instead of the default of only one::
        
       eigensolver=Davidson(3)
       
4) Use a smoother distribution function for the occupation numbers.

   Remember that for systems without periodic boundary conditions
   (molecules) the Fermi temperature is set to zero by default.
   You might want to specify a finite Fermi temperature as described
   :ref:`here <manual_occ>` and check the convergence of
   the results with respect to the temperature!

5) Try adding more empty states.

   If you are specifying the :ref:`number of bands <manual_nbands>`
   manually, try to increase the number of empty states. You might also
   let GPAW choose the default number, which is in general large
   enough.
  
6) Use enough k-points.

   Try something like ``kpts={'density': 3.5, 'even': True}``
   (see :ref:`manual_kpts`).

7) Don't let your structure optimization algorithm take too large steps.

8) Solve the Poisson equation more accurately.

   Sometimes for metallic systems of large dimensions (thick slabs or
   large clusters), one can have the situation that the wave functions
   converge nicely, but the density does not.  For such cases it can
   help to solve the Poisson equation more accurately between each SCF
   step.  Try something like ``poissonsolver=PoissonSolver(eps=1e-12)``.

9) Better initial guess for the wave functions.

   The initial guess for the wave functions is always calculated
   using the LCAO scheme, with a default single-zeta basis, i.e. one
   orbital for each valence electron. You can try to make a better
   initial guess by enlarging the :ref:`manual_basis`. This can be done
   by setting ``basis='szp'`` if you want to use a
   single-zeta-polarized basis. Note that you first need to generate
   the basis file, as described in :ref:`LCAO mode <lcao>`. 
   It is also possible to use ``basis='szp(dzp)'`` to extract
   the single-zeta polarization basis set from the double-zeta
   polarization basis sets that are distributed together with
   the latest PAW datasets. 
 
