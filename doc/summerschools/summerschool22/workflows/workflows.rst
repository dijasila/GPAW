========================================
Workflows with Atomic Simulation Recipes
========================================

In this exercise we will write and run computational workflows
using the atomic simulation recipes, ASR.

The basic unit of computation in ASR is a *task*.  A task is a Python
function along with a specification of the inputs to that function.
The inputs can be either concrete values like lists, strings, or numbers,
or references to the outputs of other tasks.
Tasks that depend on one another and form a graph.  An
ASR workflow is a Python class which defines such a graph, along with
metadata about how the tasks should run and be stored.  Workflows can
then be parametrized so they run on a collection of materials,
for example.

When using ASR, we define workflows and tasks using Python.
However, the tools used to *run* workflows and tasks are command-line
tools.  Therefore, for this exercise we will be using the terminal
rather than the usual notebooks.  Basic knowledge of shell commands
is an advantage.


This exercise consists of three parts.  Specifically, we will:

 * Define simple tasks and run them as a workflow
   to obtain a band structure
 * Parametrize the workflow to apply it to multiple materials
 * Import and run already-made recipes from ASR


Part 1: Define a workflow
=========================

First we will write a workflow to calculate the band structure
of a material, which will be silicon.  There will be three
steps: Structure optimization, ground state, and
band structure.

We need to write code for the tasks in each of the three steps, and
also to write a workflow which generates those tasks.

First, go to a clean directory and create an ASR repository::

 askhl@alberich:~$ mkdir myworkflow
 askhl@alberich:~$ cd myworkflow
 askhl@alberich:~/myworkflow$ asr init
 Created repository in /home/askhl/myworkflow

The repository will store calculations under the newly created,
currently empty folder named ``tree/``.  The ``asr info`` command
will tell us a few basic things about the repository::

 askhl@alberich:~/myworkflow$ asr info
 Root:     /home/askhl/myworkflow
 Tree:     /home/askhl/myworkflow/tree
 db-file:  /home/askhl/myworkflow/registry.dat (0 entries)
 Tasks:    /home/askhl/myworkflow/tasks.py (not created)

Let's perform a structure optimization of bulk Si.
We write a function which performs such an optimization:

.. literalinclude:: tasks.py
   :end-before: end-snippet-1

This function uses a cell filter to expose the cell degrees of freedom
for the standard BFGS optimizer (see the ASE documentation on optimizers
and cell filters if interested).

Since workflows run on the local computer whereas computational tasks
(generally) run on compute nodes, we generally workflow code and task
code in different files.  ASR can load user-defined functions from the
special file ``tasks.py`` mentioned in the info command.
Create that file and save the function to it.

Next, we write a workflow:

.. literalinclude:: workflow.py
   :end-before: end-snippet-1

Explanation:

* The ``@asr.workflow`` decorator tells ASR to regard the class as a
  workflow.  In particular, it equips the class with a constructor
  with appropriate input arguments.

* ``asr.var()`` is used to declare input variables.  The names
  ``atoms`` and ``calculator`` imply
  that we want this workflow to take atoms and calculator parameters
  as input.

* The method ``relax()`` defines our task.
  By naming the method ``relax()``, we choose that the task will run
  in a directory called ``tree/relax``.

* The method returns ``asr.node(...)``, which is a specification of
  the *actual* calculation: The name of the task
  (in ``tasks.py``) as a string, followed by inputs.
  The attribute ``self.atoms`` will become the input variable
  of the same name.

  When defining a node, ASR calculates a hash (i.e. checksum) of the inputs;
  the hash will become different if the inputs are changed.

* The decorator ``@asr.task`` tells ASR to define a task from the
  node returned by the method.
  This decorator can also be equipped with additional information,
  for example computational resources.
  We currently do not specify any such information.


Save the code to a file, e.g. named ``workflow.py``.  Then execute the
workflow by issuing the command::

  asr workflow workflow.py

The command executes the workflow and creates a folder under the ``tree/``
directory for each task.
We can run ``asr ls`` to see a list of the tasks we generated::

  541d427c new      tree/relax                     relax(atoms=…, calculator=…)

Feel free to look at the contents of the ``tree/relax`` directory.
The task is listed as "new" because we did not run it yet
— we only *created* it, so far.  While developing workflows,
we will often want to
create and inspect the tasks before we submit anything expensive.
If we made a mistake, we can correct it by removing
by e.g. ``asr remove tree/relax``, then fixing the running the workflow
again.

Let's the task on the local computer using ``asr run``::

  asr run tree/relax

If everything worked as intended, the task will now be "done",
which we can see by running ``asr ls`` again::

  541d427c done     tree/relax                     relax(atoms=…, calculator=…)


We can use the very handy standard command, ``tree``, to see the whole
directory tree::

  askhl@alberich:~/myworkflow$ tree tree/
  tree/
  └── relax
      ├── gpaw.txt
      ├── input.json
      ├── input.resolved.json
      ├── opt.log
      ├── opt.traj
      ├── output.json
      └── state.dat

  1 directory, 7 files

Be sure to open the trajectory file in (e.g. in ASE GUI) and check
``gpaw.txt`` and ``opt.log`` to verify that the job ran as expected.


Part 2: Ground state and band structure tasks
=============================================

After the relaxation, we want to run a ground state
calculation to save a ``.gpw`` file, which we subsequently want
to pass to a non-selfconsistent calculation to get the band structure.

Add a ``groundstate()`` function to ``tasks.py``:

.. literalinclude:: tasks.py
   :pyobject: groundstate

In order to "return" the gpw file, we actually return a ``Path`` object
pointing to it.  ASR will make sure that the path is resolved to the correct
location when other tasks are running.

Let's add a corresponding groundstate method to the workflow:

.. literalinclude:: workflow.py
   :pyobject: MyWorkflow.groundstate


By calling ``asr.node(..., atoms=self.relax)``, we are specifying
that the atoms should be taken as the *output* of the ``relax`` task,
creating a dependency.

We can now run the workflow again.  The old task still exists and
will remain unchanged, whereas the new task should now appear
in the ``tree/groundstate`` directory.

Run the ground state task and check that the .gpw file was created as
expected.

Finally, we write a band structure task and corresponding
workflow method:

.. literalinclude:: tasks.py
   :pyobject: bandstructure

.. literalinclude:: workflow.py
   :pyobject: MyWorkflow.bandstructure

Run the workflow and execute the task.
The code saves the Brillouin zone path and band structure separately to
ASE JSON files.  Once it runs, we can go to the directory and check
that it looks correct::

  asr reciprocal tree/bandstructure/bandpath.json

::

   asr bandstructure tree/bandstructure/bs.json

You can delete all of them with ``asr remove tree/`` and run them from
scratch by ``asr run tree/``, ``asr run tree/*``, or simply ``asr run
tree/bandstructure``.  The run command always executes tasks in
topological order, i.e., each task runs only when its dependencies
are done.

The ``asr ls`` command can also be used to list tasks in topological
order following the dependency graph::

  askhl@alberich:~/myworkflow$ asr ls --parents tree/bandstructure/
  541d427c done     tree/relax                     relax(atoms=…, calculator=…)
  5ca14caa done     tree/groundstate               groundstate(atoms=<541d427c>, calculator=…)
  b5875ebd done     tree/bandstructure             bandstructure(gpw=<5ca14caa>)


This way, we can comfortably work with larger numbers of tasks.


Part 2: Run workflow on multiple materials
==========================================

The current workflow creates directories right under the repository root.
For a materials workflow, we'd probably find it helpful to work
with a structure that nests the tasks by material.


Part 3: Use recipe workflows from ASR
=====================================

...
