How to turn off things (and maker development and debugging simpler)
====================================================================

When developing new features or debugging it can be an advantage to simplify the problem by turning things off.


No XC functional
----------------

Use ``xc={'name': 'null'}``.


No Coulomb interactions
-----------------------

::

  from gpaw.poisson import NoInteractionPoissonSolver
  calc = GPAW(...,
              poissonsolver=NoInteractionPoissonSolver())


No PAW corrections
------------------

* For hydrogen we have an "all-electron" potential (bare Coulomb potential).
  Use ``setups='ae'``.  See :ref:`ae hydrogen atom` and
  :git:`gpaw/ae.py`.

* For aluminium we have the Appelbaum-Hamann local pseudo potential.
  Use ``setups='ah'``.  See :git:`gpaw/test/pseudopotential/test_ah.py`
  and :git:`gpaw/ah.py`.

* PP's ...


H-example here ...
