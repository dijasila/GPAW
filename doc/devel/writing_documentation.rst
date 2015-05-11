.. _reStructuredText: http://docutils.sf.net/rst.html
.. _Sphinx: http://sphinx.pocoo.org
.. _PDF: ../GPAW.pdf

Writing documentation
=====================

.. highlight:: bash

We use the Sphinx_ tool to generate the GPAW documentation (both HTML
and PDF_).

First, you should take a look at the documentation for Sphinx_ and
reStructuredText_.

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

More tricks to be found at ASE `Writing documentation
<https://wiki.fysik.dtu.dk/ase/development/writing_documentation_ase.html>`_
page. Note that GPAW has `PSTricks <http://tug.org/PSTricks>`_ as an
additional requirement, which is usually provided by ``texlive-pstricks``.
