.. _instructors:

============================
Instructions for instructors
============================

The DTU databar
===============

The summer school exercises will be running in the DTU Databar, not on
Niflheim.  You can log in with your DTU username and password.  Like
Niflheim, it has a login node and a number of compute node, but unlike
Niflheim it also has *interactive compute nodes* where the students
and you can run e.g. your Jupyter notebooks.

Logging in
----------

::
   
   ssh -X login.gbar.dtu.dk

Then proceed from the login node to an interactive compute node with
the command::
  
  linuxsh -X


Setting up the Virtual Environment etc
--------------------------------------

To prepare your account, run the script (you only need to do this
once)::

  bash ~jasc/CAMD2022/setup2022_teacher

(Note that this is an extended version of the students' setup script
which will give you an editable version of GPAW and its documentation).

This will create a folder ``CAMD2022``, a "slim" virtual environment in
``CAMD2022/venv``, and a MyQueue configuration.  The slim virtual
environment is a fake venv folder where most of the contents is links
to a master environment.  If this is causing trouble, please contact
Jakob to get it fixed.

Remember to activate the virtual environment - this may be done
automatically for the students (XXX fix later)::

  source ~/CAMD2022/venv/bin/activate


Summerschool Notebooks
======================

The summerschool notebooks exist in two versions, the **student
version** and the **teacher version**.  The student version is
somewhat censored, the teacher version also contain a working
solution.

The student version is autogenerated from the teacher version and is
available from the :ref:`summerschool22` web page under the individual
projects.  Remember also to download associated ``.png`` files or they
will not display properly.

The teacher version is stored in the GPAW source code, but as a
``.py`` file with special comments to help automatic conversion to
Notebook format.  In this way, the teacher versions can be run as part
of the GPAW test suite, this is the only reason that we can expect the
notebooks to still work after four years of code development!

The .py files are found in the project folders under
``venv/gpaw/doc/summerschools/summerschool22``.

Student versions of all notebooks are extracted from the source Python
files when you build the GPAW documentation (``make`` in the ``doc``
folder).  Edit your notebooks in another folder, as **building the
documentation mayoverwrite all notebooks without warning.**

Making small modifications to the notebooks
-------------------------------------------

It is probably easiest to first make the modification in the
downloaded Notebook file, and then make the same modification in the
.py file.  Then submit a merge request.

Teacher versus student version
------------------------------

In the ``.py`` file, a cell can be marked to be left out in the
student version like this::

  # %%
  # teacher:
  print('N2 bond length:', slabN2.get_distance(8, 9))

The first line marks that a new cell should be made, the second that
the cell should be left out of the student version.

You can also make a single line appear different in the teacher and
student version::

  vib = Vibrations(slab,
                   name='vib',
                   indices=[8, 9],  # student: indices=[?, ?],
                   nfree=4)

Here, one line reads ``indices=[8, 9],`` but ``indices=[?, ?],`` in
the student version.


Making new notebooks or large modifications
-------------------------------------------

When making a new notebook, make it as a normal notebook, use the
``teacher`` / ``student`` markup in the comments as needed, and then
get help converting it to a .py file once it is ready.

To make major changes to an existing notebook, first extract the
"teacher version" from the .py file with the command::

  cd doc/summerschools/summerschool22
  python convert.py --teacher

**move it somewhere else to update it**, and then get help converting
it back once it is ready.  Note that rebuilding the documentation or
rerunning the command above **will cause all notebooks to be
overwritten**.

The extracted notebook will start with a cell::

  # teacher
  import ase.visualize as viz
  viz.view = lambda atoms, repeat=None: None

It disables the ASE gui when the source script is run as part of the GPAW
test suite, but should probably be deleted while you develop the notebook.

Running Notebooks in the Databar
--------------------------------

Notebooks can developed on Niflheim or elsewhere, but *all notebooks
should be testet in the DTU Databar*.

Instructions on how to run notebooks in the databar can be found here:

* :ref:`accesswin`
* :ref:`accesslinmac`


Updating the text of the project pages
--------------------------------------

Remember to also update the web pages associated with the projects.
This is done by editing the associated ``.rst`` files in
``doc/summerschools/summeschool22``, and then submitting a merge request.



Notes on how to set this up
===========================

These notes are intended as a starting point for setting this up in
2024 !

* Build a GPAW venv called venv-master

* Upgrade sphinx, otherwise it will not work::

    pip install --upgrade  sphinx sphinx-rtd-theme
