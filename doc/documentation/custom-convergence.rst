.. _custom-convergence:

===========================
Custom convergence criteria
===========================

You can set any criteria you like by writing a custom convergence criterion.
There are also pre-defined criteria (actually, currently only one), that you can use like this::

  from gpaw import GPAW
  from gpaw.scf import WorkFunction

  calc = GPAW(...,
              convergence={...,
                           custom=[WorkFunction(tol=0.001, n_old=3)]},
             )

The :code:`custom` list can contain as many custom criteria as you like, and you can stil use standard convergence keywords like :code:`'density'`.

You can write your own custom convergence criteria if you structure it like this::

  class MyCriterion:
      name = 'my criterion'  # must be a unique name
      tablename = 'crit1'  # this prints as the header in the SCF table

      def __init__(self, ...):
          ...  # your code here
          self.description = 'My custom criterion with tolerance ...'
          # the above line will print at the top of the log file

      def __call__(self, context):
          ...  # your code here
          converged = ...  # True or False if your criterion is met
          entry = ...  # a string with up to 5 characters to print in SCF table
          return converged, entry

       def todict(self):
           # code to convert your criterion to a dictionary, to save in
           # .gpw files
           return {'name': self.name,
                   ...
                  }


Help for the built-in criterion follows.

.. autoclass:: gpaw.scf.WorkFunction
