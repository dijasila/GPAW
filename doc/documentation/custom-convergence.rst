.. _custom-convergence:

===========================
Custom convergence criteria
===========================

If you'd like to adjust how the SCF cycle decides when it is complete, you can set a custom convergence criterion (in addition to the :ref:`default convergence criteria <manual_convergence>`).
There are also pre-defined custom criteria (actually, currently only one), that you can use like this::

  from gpaw import GPAW
  from gpaw.scf import WorkFunction

  convergence={...,
               custom=[WorkFunction(tol=0.001, n_old=3)]},

  calc = GPAW(...,
              convergence=convergence)

The above example will make sure that the work function changes by no more than 0.001 eV across the last three SCF iterations.
The :code:`custom` list can contain as many custom criteria as you like, and you can still use standard convergence keywords like :code:`'density'` or :code:`'energy'`.

You can also use this syntax to change things about how the default criteria work.
For example, if you'd like the energy convergence criterion to only examine the changes in the last two values of the energy, instead of the default three, you can define the convergence dictionary as::

  from gpaw.scf import Energy

  convergence = {'energy': Energy(tol=0.0005, n_old=2)}

instead of::

  convergence = {'energy': 0.0005}

(The last line is equivalent to :code:`convergence={'energy': Energy(0.0005)}`; :code:`0.0005` is taken as the first positional argument and the default value of :code:`n_old=3` is assumed.)
You can call any pre-defined convergence criterion by its :code:`name` attribute even if it's not in the default convergence dictionary, for example :code:`convergence={'work function': 0.001}` is equivalent to the first example above.

You can write your own custom convergence criteria if you structure it like this::

  from gpaw.scf import Criterion


  class MyCriterion(Criterion):
      name = 'my criterion'  # must be a unique name
      tablename = 'mycri'  # <=5 char, prints as header in the SCF table
      calc_last = False  # if True, waits until all other criteria are met
                         # before checking (for expensive criteria)

      def __init__(self, ...):
          ...  # your code here; note if you save all variables directly
               # (as self.a, self.b, ...) then todict() and __repr__ methods
               # will work automatically.
          # The next line prints at the top of the log file.
          self.description = 'My custom criterion with tolerance ...'

      def __call__(self, context):
          ...  # your code here
          converged = ...  # True or False if your criterion is met
          entry = ...  # a string with up to 5 characters to print in SCF table
          return converged, entry

      def reset(self):
          ...  # your code here to clear anything saved whenever
               # the SCF restarts



  calc = GPAW(...,
              convergence={'custom': [MyCriterion(0.01, 4)]}
             )


Help for the built-in criteria follows.

.. autoclass:: gpaw.scf.Energy
.. autoclass:: gpaw.scf.Density
.. autoclass:: gpaw.scf.Eigenstates
.. autoclass:: gpaw.scf.Forces
.. autoclass:: gpaw.scf.WorkFunction
