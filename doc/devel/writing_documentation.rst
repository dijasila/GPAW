=====================
Writing documentation
=====================

.. highlight:: bash

We use the Sphinx_ tool to generate the GPAW documentation.

First, you should take a look at the documentation for Sphinx_ and
reStructuredText_.  Also, read carefully the
:ref:`Writing documentation for ASE <ase:writing_documentation_ase>`
page.

.. _reStructuredText: http://docutils.sf.net/rst.html
.. _Sphinx: http://sphinx.pocoo.org


Getting started
===============

If you don't already have your own copy of the GPAW package, then
perform a :ref:`developer_installation`.

Then :command:`cd` to the :file:`doc` directory and build the html-pages::

  $ cd ~/gpaw/doc
  $ make

.. Note::

   Make sure that you build the Sphinx documentation using the corresponding
   GPAW version by setting the environment variables :envvar:`PYTHONPATH`,
   :envvar:`PATH` (described at :ref:`developer_installation`) and
   the location of setups (decribed at :ref:`installationguide_setup_files`).

Make your changes to the ``.rst`` files, run the
:command:`make` command again, check the results and if things
looks ok, commit::

  $ emacs index.rst
  $ make
  $ firefox build/html/index.html
  $ svn ci -m "..." index.rst

  
Adding figures and tables
=========================

We don't want to have png and csv files committed to SVN.  Instead, you should
add the Python scripts that generate the figures and table data so that we can
always generate them again if needed.
    
For quick scripts (no more than 5 seconds), see :ref:`ase:generated`.  For
more expensive scripts you can use :ref:`AGTS <agts>` for running long jobs
that create figures or table data for this web-page.  For an example, look at
the source code :trac:`here <doc/tutorials/stm>` which will produce this:
:ref:`stm tutorial`.
