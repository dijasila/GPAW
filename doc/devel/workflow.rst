.. _development workflow:

====================
Development workflow
====================

.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _venv: https://docs.python.org/3/library/venv.html#module-venv
.. _pip: https://pip.pypa.io/
.. _git: https://git-scm.com/
.. _GitLab issues: https://gitlab.com/gpaw/gpaw/issues
.. _pytest: https://docs.pytest.org/en/6.2.x/

.. contents::


Setting up your development environment
=======================================

Make a venv_::

 $ mkdir devel
 $ cd devel
 $ python3 -m venv venv
 $ source venv/bin/activate  # venv/bin/ is now first in $PATH
 $ pip install --upgrade pip

Install ASE_::

 $ git clone git@gitlab.com:ase/ase
 $ pip install --editable ase/

Install GPAW::

 $ git clone git@gitlab.com:gpaw/gpaw
 $ echo "noblas = True; nolibxc = True" > gpaw/siteconfig.py
 $ pip install -e gpaw

.. note::

    ``noblas = True``: Use the BLAS library built into  NumPy_
    (usually OpenBLAS).

    ``nolibxc = True``: Use GPAW's own XC-functionals
    (only LDA, PBE, revPBE, RPBE and PW91).

Get PAW datasets::

 $ gpaw install-data --register ~/PAWDATA

Run the tests
=============

::

 $ pip install pytest-xdist
 $ pytest -n4

With MPI (2, 4 and 8 cores)::

 $ mpiexec -n 2 pytest


Creating a merge request
========================

Request to become a member of the ``gpaw`` group on GitLab.

::

 $ cd gpaw
 $ git switch -c fix-something

.. note::

   ``git switch -c fix-something`` is the same as any of these:

   * ``git branch fix-something && git switch fix-something``
   * ``git branch fix-something && git checkout fix-something``
   * ``git checkout -b fix-something``

   :xkcd:`More git-tricks <1597>`__.

Make changes and commit::

 $ git add file1 file2 ...
 $ git commit -m "Short summary of changes"

Push your branch to GitLab::

 $ git push --uspstream origin fix-somthing

and click the link to create a merge-request (MR).  Mark the MR as DRAFT to
signal that it is work-in-progress and remove the DRAFT-marker once the MR
is ready for code review.

Every time you push your local repository changes upstream to the remote
repository, you will trigger a continuous integration (CI) runner on the
GitLab servers.  The script that runs in CI is :git:`.gitlab-ci.yml`.
Here is a short summary of what happens in CI:

* install the code
* ``pytest -m ci``: small selection of fast tests
* ``mypy -p gpaw``: `Static code analysis`_ (type hints)
* ``flake8``: pyflakes + pycodestyle (pep8) = flake8_

If CI fails, you will have to fix things and push your changes.

It's a good idea to also run the CI-checks locally::

 $ pip install flake8 mypy
 $ flake8 ...
 $ mypy ...
 $ pytest ...
 $ # fix things
 $ git add ...
 $ git commit ...
 $ git push  # Git now knows your upstream

.. _Static code analysis: https://mypy.readthedocs.io/en/stable/
.. _flake8: https://flake8.pycqa.org/en/latest/
