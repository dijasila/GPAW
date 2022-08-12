==================
Workflows with ASR
==================

In this exercise we will write and run computational workflows
using the atomic simulation recipes, ASR.

The basic unit of computation in ASR is a *task*.  A task is a Python
function along with a specification of the inputs to that function.
The input of a task may be the output of another task, even if the other
task's output does not exist yet --- a kind of future reference.
Hence, tasks depend on one another forming a graph.  An
ASR workflow is a Python class which defines such a graph, along with
metadata about how the tasks should run and be stored.

When using ASR, we define workflows and tasks using Python.
However, the tools used to *run* workflows and tasks are command-line
tools.  Therefore:

.. note::

   For this exercise we will use the GBar terminal rather than a notebook.
   The exercise hence requires basic knowledge of shell commands.


This exercise consist of three parts were we will:

 * Define simple tasks and use run them with a simple workflow
   to obtain a band structure
 * Parametrize the workflow, applying it to multiple materials
 * Import and run already-made recipes from ASR


Part 1: Write a basic workflow
==============================

We will write a simple workflow which performs a structure optimization
of Si followed by a ground-state calculation and band structure.

To get started, go to a clean directory and create an ASR repository::

:: bash

  mkdir myworkflow
  cd myworkflow
  asr init

The repository will store calculations under the newly created,
currently empty folder named tree/.  The file registry.dat is a
database used for efficient indexing of tasks.

:: bash

   asr info

The info command tells us what there is to know about our repository.

Let's perform a structure optimization of bulk Si.
This function performs such an optimization::

.. literalinclude: tasks.py

We need to save the code to the special tasks.py file in the repository
root.  In ASR, user-defined tasks can be stored in tasks.py such that they
do not need to exist inside an installed library.

.. literalinclude: workflow.py

Save the code to a file, e.g. named workflow.py.  Then execute the
workflow by issuing the command::

  asr workflow workflow.py

The command executes the workflow and creates a folder for each task.
The tasks did not run yet, which gives us a chance to inspect what is there
before we run the thousands of tasks that we might have defined if we
were running a real materials workflow.

.. highlight:: bash

  asr ls

A directory has been created under the tree with the same name as the
method which we defined on the workflow task.
To run the workflow, we can use the run command::

  asr run tree/

Alternatively, we can submit it using::

  asr submit tree/

Typically we will make mistakes and need to adapt the tasks several
times before finally submitting thousands of expensive calculations.  We can
remove the tasks using the

  asr unrun tree/   # delete outputs of tasks
  asr remove tree/  # delete all tasks entirely

a number of calculations, still zero, 


Part 2: Run workflow on multiple materials
==========================================

...

Part 3: Use recipe workflows from ASR
=====================================

...
