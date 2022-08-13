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


Part 1: Write a basic workflow
==============================

First we will write a workflow which performs a structure optimization
of Si followed by a ground-state calculation and then band structure.

We need to write code for the tasks in each of the three steps, and
then we need to write a workflow which generates those tasks.


To get started, go to a clean directory and create an ASR repository::

  mkdir myworkflow
  cd myworkflow
  asr init

The repository will store calculations under the newly created,
currently empty folder named ``tree/``.  The ``asr info`` command
will tell us a few basic things about the repository::

   asr info

Let's perform a structure optimization of bulk Si.
We write a function which performs such an optimization:

.. literalinclude:: tasks.py
   :pyobject: relax

Since workflows run on the local computer whereas tasks (generally)
run on compute nodes, we generally put them in different files.
Save the function to ``tasks.py`` in the repository root directory.
That file is special, as ASR will look up functions inside the file
when running workflows.
Next, we need to write a workflow.


.. literalinclude: workflow.py

Save the code to a file, e.g. named ``workflow.py``.  Then execute the
workflow by issuing the command::

  asr workflow workflow.py

The command executes the workflow and creates a folder for each task.
The tasks did not run yet, which gives us a chance to inspect what is there
before we run the thousands of tasks that we might have defined if we
were running a real materials workflow.

::

  asr ls

A directory has been created under the tree with the same name as the
method which we defined on the workflow task.
To run the workflow, we can use the run command::

  asr run tree/

Alternatively, we can submit it using::

  asr submit tree/


Typically we will make mistakes and need to adapt the tasks several
times before finally submitting thousands of expensive calculations.
To rerun a task, we first



  asr unrun tree/   # delete outputs of tasks
  asr remove tree/  # delete all tasks entirely



a number of calculations, still zero, 


Part 2: Run workflow on multiple materials
==========================================

...

Part 3: Use recipe workflows from ASR
=====================================

...
